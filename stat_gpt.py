import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from math import log, exp
from tqdm import tqdm
import time

# Load fine-tuned model and tokenizer
model_path = "/local/data2/home/thawe276/Documents/gpt2-finetuned-wikitext103{seed}"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load evaluation dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
eval_texts = dataset["validation"]["text"]

def tokenize_function(text, max_length=512):
    return tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

loss_fn = CrossEntropyLoss(reduction="none")

total_nll = 0.0
total_tokens = 0
total_model_time = 0.0  # track only model forward+loss time

with torch.no_grad():
    for text in tqdm(eval_texts, desc="Evaluating"):
        if not text.strip():
            continue

        inputs = tokenize_function(text)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = None  # match your requested style

        if input_ids.size(1) < 2:
            continue

        # START model computation timing
        start = time.time()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        end = time.time()
        # END model computation timing

        total_model_time += (end - start)
        total_nll += loss.sum().item()
        total_tokens += shift_labels.numel()

# Compute metrics
avg_nll = total_nll / total_tokens
bpt = avg_nll / log(2)
ppl = exp(avg_nll)
tokens_per_sec = total_tokens / total_model_time

print(f"\nEvaluation Results:")
print(f"  - Average NLL         : {avg_nll:.4f}")
print(f"  - Bits Per Token (BPT): {bpt:.4f}")
print(f"  - Perplexity (PPL)    : {ppl:.4f}")
print(f"  - Total Tokens        : {total_tokens}")
print(f"  - Model Time (s)      : {total_model_time:.2f}")
print(f"  - Speed (tokens/sec)  : {tokens_per_sec:.2f}")

# Generate sample predictions
print("\nSample generations from model:")
sample_prompts = [
    "The history of artificial intelligence began",
    "In a distant future, humanity has",
    "Quantum computing offers the potential to",
]

for prompt in sample_prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=50,
        num_return_sequences=1,
        do_sample=False
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}\n→ {generated_text}")
