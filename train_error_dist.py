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
from minis2t_50 import MiniS2T
from error_correction_model import ErrorCorrectionModel
import aiohttp


def setup_ddp():
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

def ddp_average(value, device):
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

def kld_loss(student_logits, teacher_logits, temp=1):
    B, T_s, V = student_logits.shape
    T_t = teacher_logits.size(1)

    # If time dims differ, interpolate student -> teacher length
    if T_s != T_t:
        student_logits = F.interpolate(
            student_logits.transpose(1, 2),  # (B, V, T_s)
            size=T_t,
            mode="linear",
            align_corners=False
        ).transpose(1, 2)  # (B, T_t, V)

    # Flatten for KL (batch * time, vocab)
    flat_student = student_logits.contiguous().view(-1, V)
    flat_teacher = teacher_logits.contiguous().view(-1, V)

    return F.kl_div(
        F.log_softmax(flat_student / temp, dim=-1),
        F.softmax(flat_teacher.detach() / temp, dim=-1),
        reduction="batchmean"
    ) * (temp ** 2)

def preprocess(batch):
    batch["audio"] = batch["audio"]["array"]
    return batch

def collate_fn(batch, processor):
    audio = [example["audio"] for example in batch]
    texts = [example["text"] for example in batch]
    inputs = processor(audio, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True)
    labels = [torch.tensor(processor.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
    return inputs.input_values, inputs.attention_mask, labels

def training_step(
    student, 
    teacher, 
    error_model, 
    processor, 
    optimizer, 
    batch_examples, 
    ctc_loss_fn, 
    alpha, 
    beta, 
    kl_temp=1, 
    scheduler=None,
    device='cpu'
):
    waveforms, attention_mask, labels = collate_fn(batch_examples, processor)
    waveforms, attention_mask = waveforms.to(device), attention_mask.to(device)

    logits = {}

    # Teacher and Student Forward
    with torch.no_grad():
        teacher_out = teacher(waveforms, attention_mask=attention_mask, output_hidden_states=False)
        logits[0] = teacher_out.logits

        student_logits = student(waveforms, attention_mask, output_hidden_states=False)

    #Error Model Forward
    logits[1] = error_model(student_logits, attention_mask=attention_mask)

    # Compute losses
    # KL
    error_kld_loss = beta * kld_loss(logits[1], logits[0], temp=kl_temp)

    # CTC
    input_lengths = torch.full(
        (waveforms.size(0),),
        logits[1].size(1),   # student/error model length
        dtype=torch.long,
        device=device
    )

    targets = torch.cat(labels).to(device)
    target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long, device=device)

    log_probs = F.log_softmax(logits[1], dim=-1).transpose(0, 1)
    error_ctc_loss = alpha * ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    # ---- DDP Averaging and Logging ----
    kld_global = ddp_average(error_kld_loss.item(), device)
    ctc_global = ddp_average(error_ctc_loss.item(), device)
    
    loss = error_kld_loss + error_ctc_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return kld_global, ctc_global, loss


def train_ddp(
    student,
    teacher,
    error_model,
    processor, 
    dataset,
    optimizer,
    alpha, 
    beta, 
    batch_size=16,
    scheduler=None,
    kl_temp=1,
    save_path='.',
    epochs=100,
    resume=0,
):
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    world_size = torch.distributed.get_world_size()

    # Move teacher and students to device
    teacher.to(device)
    teacher.eval()

    student.to(device)
    student.eval()

    for p in student.parameters():
        p.requires_grad = False

    error_model.to(device)
    error_model.train()
    error_model = DDP(error_model, device_ids=[local_rank], output_device=local_rank)

    # === Create Weight Folders ===
    if local_rank == 0:
        path = f'./{save_path}/error'
        if not os.path.exists(path):
            os.makedirs(path)

    ctc_loss_fn = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    for epoch in range(resume, epochs):
        avg_kld, avg_ctc = [], []

        batch_examples = []
        batch_num = 1

        for idx, example in enumerate(dataset):
            # Each rank processes its slice
            if idx % world_size != local_rank:
                continue

            # Preprocess and accumulate
            batch_examples.append(preprocess(example))

            if len(batch_examples) == batch_size:
                kld_global, ctc_global, loss = training_step(
                    student=student,
                    teacher=teacher,
                    error_model=error_model,
                    processor=processor,
                    optimizer=optimizer,
                    batch_examples=batch_examples,
                    ctc_loss_fn=ctc_loss_fn,
                    alpha=alpha,
                    beta=beta,
                    kl_temp=kl_temp,
                    scheduler=scheduler,
                    device=device
                )

                avg_kld.append(kld_global)
                avg_ctc.append(ctc_global)
                
                if local_rank == 0:
                    print(f'epoch/batch num {epoch+1}/{batch_num}')
                    print(f'Error: CTC, KLD, Total : {avg_ctc[-1]:.4f}, {avg_kld[-1]:.4f}, {loss.item():.4f}')
                batch_examples = [] #Reset batch examples
                batch_num += 1

        if len(batch_examples) > 0:
            kld_global, ctc_global, loss = training_step(
                student=student,
                teacher=teacher,
                error_model=error_model,
                processor=processor,
                optimizer=optimizer,
                batch_examples=batch_examples,
                ctc_loss_fn=ctc_loss_fn,
                alpha=alpha,
                beta=beta,
                kl_temp=kl_temp,
                scheduler=scheduler,
                device=device
            )

            avg_kld.append(kld_global)
            avg_ctc.append(ctc_global)
            
            if local_rank == 0:
                print(f'epoch/batch num {epoch+1}/{batch_num}')
                print(f'Error: CTC, KLD, Total : {avg_ctc[-1]:.4f}, {avg_kld[-1]:.4f}, {loss.item():.4f}')


        # Save checkpoints only on rank 0
        if local_rank == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": error_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            if scheduler:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(ckpt, f'{save_path}/error/{epoch+1}.pt')

            with open(f'{save_path}/metrics_error.txt', "a") as f:
                log_out = f'Epoch {epoch+1}:\n'
                log_out += f'\tError : CTC, KLD : CTC:{sum(avg_ctc)/len(avg_ctc):.4f}\tKL:{sum(avg_kld)/len(avg_kld):.4f}\n'   
                log_out += '\n'
                f.write(log_out)



if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    # Load teacher model
    teacher_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    intermediate1 = MiniS2T(vocab_size=32, d_model=512, n_head=8, num_layers=12, dim_feedforward=2048, hidden_dim=512).to(device)
    error_model = ErrorCorrectionModel(vocab_size=32, zero_init=True).to(device)

    
    student_model = intermediate1

    alpha, beta = 1, 0

    intermediate1.load_state_dict(torch.load('./model_150.pt', map_location=device)) 

    params = list(error_model.parameters())
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    batch_size = 16

    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.360",
        trust_remote_code=True,
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
        cache_dir='./librispeech',
    )

    dataset_size = 104014
    #count_parameters(final_student)
    resume = 0
    num_epochs = 30

    setup_ddp()

    scheduler, total_steps = get_cosine_warmup_scheduler_ddp(
        optimizer, dataset_size, batch_size, num_epochs, learning_rate, warmup_ratio=0.05, min_lr_ratio=0.05
    )

    # === Log Parameters === 
    train_ddp(
        student=student_model,
        teacher=teacher_model, 
        error_model=error_model, 
        processor=processor, 
        dataset=dataset, 
        optimizer=optimizer,
        alpha=alpha, 
        beta=beta, 
        batch_size=batch_size,
        scheduler=scheduler,
        kl_temp=2,
        save_path='./weights', 
        epochs=num_epochs,
        resume=resume
    ) 