from datasets import load_dataset
from transformers import GPT2TokenizerFast
import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, GPT2LMHeadModel

# Set a seed for reproducibility
seed = 2000
torch.manual_seed(seed)

# Load the WikiText-103 dataset (using the raw version)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Load the GPT-2 tokenizer (fast version)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# GPT2 does not have a pad token by default; we set it to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Define a tokenization function that truncates or pads sequences to a fixed max_length.
def tokenize_function(examples, max_length=512):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=max_length)
    return tokenized

# Tokenize the training and validation splits
tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples), batched=True)

# Remove the "text" column, we only need input_ids and attention_mask
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Filter out examples that result in empty input_ids
tokenized_datasets = tokenized_datasets.filter(lambda example: len(example["input_ids"]) > 0)

# Set the dataset format to PyTorch tensors
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

output_dir = f"./gpt2-finetuned-wikitext103-seed{seed}"
# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext103",  # Directory to save model checkpoints
    overwrite_output_dir=True,
    num_train_epochs=3,                          # Adjust number of epochs as needed
    per_device_train_batch_size=16,               # Adjust based on your GPU/CPU memory
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",                 # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=200,
    fp16=True if torch.cuda.is_available() else False,  # Mixed precision if GPU available
    seed=seed,                                   # Add seed to TrainingArguments
)

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id

# Initialize data collator. For GPT-2, we need to specify that it should pad using the eos token.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # mlm=False for autoregressive language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# Save the model and tokenizer
print("Saving fine-tuned model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

