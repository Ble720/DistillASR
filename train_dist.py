import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import os
import math

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from miniASR import MiniS2T
import aiohttp


def setup_ddp():
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

def ddp_average(value, device):
    """Reduce scalar value across GPUs and return global average."""
    tensor = torch.tensor(value, device=device, dtype=torch.float)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

def get_cosine_warmup_scheduler_ddp(optimizer, dataset_size, batch_size_per_gpu, num_epochs, base_lr, warmup_ratio=0.05, min_lr_ratio=0.05):
    """
    Cosine LR scheduler with linear warmup for multi-GPU (DDP) training.

    Args:
        optimizer: torch optimizer
        dataset_size: total number of examples in dataset
        batch_size_per_gpu: batch size on each GPU
        num_epochs: total epochs to train
        warmup_ratio: fraction of total steps used for warmup (default 0.05)
        min_lr_ratio: ratio of base_lr for floor (e.g. 0.05 → 5% of base)
    """
    # Compute steps per epoch using global batch size
    global_batch_size = batch_size_per_gpu * torch.distributed.get_world_size()
    steps_per_epoch = math.ceil(dataset_size / global_batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr = base_lr * min_lr_ratio

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay with floor
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

    return LambdaLR(optimizer, lr_lambda), total_steps

def distill_loss(student_hidden_states, teacher_hidden_states, proj_layers):
    total_loss = 0.0
    choosen_layers_student = [1, 8, 12]
    choosen_layers_teacher = [1, 8, 12]

    if len(teacher_hidden_states) == 25:
        choosen_layers_teacher = [2, 16, 24]

    if isinstance(proj_layers, DDP):
        proj_layers = proj_layers.module


    for i in range(len(choosen_layers_student)):
        # Project teacher hidden states to student dimension
        teacher_id = choosen_layers_teacher[i]
        projected_teacher = proj_layers[str(i)](teacher_hidden_states[teacher_id].detach())
        
        # Compute MSE loss between student and projected teacher hidden states
        total_loss += F.mse_loss(student_hidden_states[choosen_layers_student[i]], projected_teacher)
    
    avg_loss = total_loss / len(choosen_layers_student)
    return avg_loss

def kld_loss(flat_logits_student, flat_logits_teacher, temp=1):
    return F.kl_div(
        F.log_softmax(flat_logits_student / temp, dim=-1),
        F.softmax(flat_logits_teacher.detach() / temp, dim=-1),
        reduction="batchmean"
    )  * (temp ** 2)

def preprocess(batch):
    batch["audio"] = batch["audio"]["array"]
    return batch

def collate_fn(batch, processor):
    audio = [example["audio"] for example in batch]
    texts = [example["text"] for example in batch]
    inputs = processor(audio, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True)
    labels = [torch.tensor(processor.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
    return inputs.input_values, inputs.attention_mask, labels

def cascading_distillation_ddp(
    students,
    is_frozen, 
    teacher,
    projections,
    processor, 
    dataset,
    optimizer, 
    alphas, 
    betas, 
    deltas,
    batch_size=16,
    scheduler=None,
    kl_temp=1,
    save_path='.',
    epochs=100,
    resume=0,
):
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')

    # Move teacher and students to device
    if teacher is not None:
        teacher.to(device)
        teacher.eval()

    for student_id, s in students.items():
        s.to(device)
        s.train(not is_frozen[student_id])
        students[student_id] = DDP(s, device_ids=[local_rank], output_device=local_rank)

    for proj_id, module_dict in projections.items():
        projections[proj_id] = module_dict.to(device)

    for proj_id, module in projections.items():
        module.to(device)
        projections[proj_id] = DDP(module, device_ids=[local_rank], output_device=local_rank)

    # === Create Weight Folders ===
    if local_rank == 0:
        for id in students.keys():
            path = f'./{save_path}/{id}'
            if not os.path.exists(path):
                os.makedirs(path)

        path = f'./{save_path}/proj'
        if not os.path.exists(path):
            os.makedirs(path)

    # Distributed sampler
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    ctc_loss_fn = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
    ids = [0] + list(students.keys()) if teacher is not None else list(students.keys())

    for epoch in range(resume, epochs):
        sampler.set_epoch(epoch)
        avg_kld, avg_ctc, avg_distill = {i: [] for i in ids}, {i: [] for i in ids}, {i: [] for i in ids}
        
        for batch_num, (waveforms, attention_mask, labels) in enumerate(dataloader, 1):
            waveforms, attention_mask = waveforms.to(device), attention_mask.to(device)

            logits, hiddens = {}, {}

            # Teacher forward
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(waveforms, attention_mask=attention_mask, output_hidden_states=True)
                    logits[0], hiddens[0] = teacher_out.logits, teacher_out.hidden_states

            # Students forward
            for student_id, s in students.items():
                s_logits, s_hiddens = s(waveforms, attention_mask, output_hidden_states=True, epoch=epoch)
                logits[student_id], hiddens[student_id] = s_logits, s_hiddens

            # Align lengths
            min_len = min(l.size(1) for l in logits.values())
            for k in logits:
                logits[k] = logits[k][:, :min_len, :]
                hiddens[k] = [layer[:, :min_len, :] for layer in hiddens[k]]

            # Flatten for KL
            flat_logits = {k: l.reshape(-1, l.size(-1)) for k, l in logits.items()}

            # Compute losses
            batch_kld, batch_distill, batch_ctc = 0.0, 0.0, 0.0
            for student_id in ids[1:]:
                if is_frozen.get(student_id, False):
                    continue

                # KL
                student_kld_loss = betas[student_id] * kld_loss(flat_logits[student_id], flat_logits[0], temp=kl_temp)
                batch_kld += student_kld_loss

                # Intermediate MSE
                student_distill_loss = 0.0
                for j in ids[:student_id]:
                    proj_id = f'{j}_{student_id}'
                    if proj_id in projections:
                        student_distill_loss += deltas[proj_id] * distill_loss(hiddens[student_id], hiddens[j], projections[proj_id])
                batch_distill += student_distill_loss

                # CTC
                input_lengths = torch.full((waveforms.size(0),), min_len, dtype=torch.long, device=device)
                targets = torch.cat(labels).to(device)
                target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long, device=device)
                log_probs = F.log_softmax(logits[student_id], dim=-1).transpose(0, 1)
                student_ctc_loss = alphas[student_id] * ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                batch_ctc += student_ctc_loss

                # ---- DDP Averaging and Logging ----
                kld_global = ddp_average(student_kld_loss.item(), device)
                ctc_global = ddp_average(student_ctc_loss.item(), device)
                distill_global = ddp_average(student_distill_loss.item(), device)

                avg_kld[student_id].append(kld_global)
                avg_ctc[student_id].append(ctc_global)
                avg_distill[student_id].append(distill_global)

            loss = batch_kld + batch_distill + batch_ctc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if local_rank == 0:
                print(f'epoch/batch num {epoch+1}/{batch_num}')

                print_out = ''
                for i in ids:
                    if is_frozen[i]:
                        continue
                    print_out += f'Student_{i} : Distill, CTC, KLD, Total : {avg_distill[i][-1]:.4f}, {avg_ctc[i][-1]:.4f}, {avg_kld[i][-1]:.4f}, {loss.item():.4f}\n'

                print(print_out)

        # Save checkpoints only on rank 0
        if local_rank == 0:
            for student_id, s in students.items():
                torch.save(s.module.state_dict(), f'{save_path}/{student_id}/{epoch+1}.pt')
            for k, v in projections.items():
                to_save = v.module.state_dict() if isinstance(v, DDP) else v.state_dict()
                torch.save(to_save, f'{save_path}/proj/proj_{k}-{epoch+1}.pt')

            with open(f'{save_path}/metrics_cascade.txt', "a") as f:
                log_out = f'Epoch {epoch+1}:\n'
                for i in ids:
                    if is_frozen[i]:
                        continue
                    log_out += f'\tStudent_{i} : Distill, CTC, KLD : \
                        {sum(avg_distill[i])/len(avg_distill[i]):.4f}\tCTC:{sum(avg_ctc[i])/len(avg_ctc[i]):.4f}\tKL:{sum(avg_kld[i])/len(avg_kld[i]):.4f}\n'
                log_out += '\n'
                f.write(log_out)



if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    # Load teacher model
    teacher_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    d_model_1 = 512
    d_model_2 = 384
    d_model_3 = 256
    model = MiniS2T(vocab_size=32, d_model=d_model_1, n_head=8, num_layers=12, dim_feedforward=2048, hidden_dim=512).to(device)  # match teacher hidden size for distillation
    #intermediate2 = MiniS2T(vocab_size=32, d_model=d_model_2, n_head=8, num_layers=12, dim_feedforward=1536, hidden_dim=384)
    #final_student = MiniS2T(vocab_size=32, d_model=d_model_3, n_head=8, num_layers=12, dim_feedforward=768, hidden_dim=256).to(device)
    
    proj_layers = {}
   
    proj_layers['0_1'] = nn.ModuleDict({
        str(i): nn.Linear(1024, d_model_1).to(device)
            for i in [0, 1, 2]
    })

    student_models = {1:model}#, 2:intermediate2}#, 3:final_student}#, final_student]
    frozen = {0:True, 1: False, 2:False}#, 3:False}

    alphas = {1:0.8, 2:0.5}
    betas = {1:0.15, 2:0.3}
    deltas = {'0_1': 1.0, '0_2': 0.75, '1_2': 0.25}

    model.load_state_dict(torch.load('./weights_cascade_p1_large_SpecAug/1/130.pt', map_location=device)) 
    proj_layers['0_1'].load_state_dict(torch.load('./weights_cascade_p1_large_SpecAug/proj/proj_0_1-130.pt', map_location=device))
    
    params = list(model.parameters()) + list(proj_layers['0_1'].parameters()) #+ list(intermediate2.parameters())  + \
    #    list(proj_layers['0_2'].parameters()) + list(proj_layers['1_2'].parameters())
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    batch_size = 16

    dataset = load_dataset("librispeech_asr", "clean", split="train.360", trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}, cache_dir='./librispeech')
    dataset = dataset.map(preprocess)

    dataset_size = len(dataset)
    #count_parameters(final_student)
    resume = 0
    num_epochs = 150

    resume_step = 0

    
    setup_ddp()


    scheduler, total_steps = get_cosine_warmup_scheduler_ddp(
        optimizer, dataset_size, batch_size, num_epochs, learning_rate, warmup_ratio=0.05, min_lr_ratio=0.05
    )

    for _ in range(resume_step):
        scheduler.step()

    # === Log Parameters === 


    cascading_distillation_ddp(
        students=student_models,
        is_frozen=frozen, 
        teacher=teacher_model, 
        projections=proj_layers, 
        processor=processor, 
        dataset=dataset, 
        optimizer=optimizer,
        alphas=alphas, 
        betas=betas, 
        deltas=deltas,
        batch_size=batch_size,
        scheduler=scheduler,
        kl_temp=1.5,
        save_path='./weights', 
        epochs=num_epochs,
        resume=resume
    ) 