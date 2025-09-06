#0.093
from datasets import load_dataset
from transformers import Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import torch
from jiwer import wer
import aiohttp
import json

from miniASR import MiniS2T
from ErrorCorrection import ErrorCorrectionModel


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")


librispeech_eval = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}, cache_dir='./librispeech')
device = 'cuda'
model = MiniS2T(vocab_size=32, d_model=512, n_head=8, num_layers=12, dim_feedforward=2048, hidden_dim=512).to(device)
error_model = ErrorCorrectionModel(vocab_size=32).to(device)

model.load_state_dict(torch.load('./model_150.pt'))
err_ckpt = torch.load('./error_10.pt')
error_model.load_state_dict(err_ckpt["model_state_dict"])

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

model.eval()
error_model.eval()
vocab = list(processor.tokenizer.get_vocab().keys())

with open("unigrams.json") as f:
    unigrams = json.load(f)

a, b = 0.55, 1.55

decoder = build_ctcdecoder(
    labels=vocab,
    kenlm_model_path="wiki_en_token.arpa.bin",
    unigrams=unigrams,
    alpha=a,
    beta=b
)

def map_to_pred(batch):
    input = processor(batch["audio"][0]["array"], return_tensors="pt", padding=True, sampling_rate=16000, return_attention_mask=True)
    input_values = input.input_values
    mask = input.attention_mask

    with torch.no_grad():
        logits = model(input_values.to("cuda"), mask.to("cuda"))
        logits = error_model(logits, mask.to("cuda"))

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()[0]
    transcription = decoder.decode(log_probs, beam_width=50)

    return {
        "text": [batch["text"][0]],             # list of refs
        "transcription": [transcription],       # list of preds
    }


result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])
print(f"a={a}, b={b}, WER:", wer(result["text"], result["transcription"]))