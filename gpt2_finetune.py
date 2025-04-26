import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Initialize Weights & Biases run
wandb.init(project="gpt2-wikitext2", name="gpt2_finetuen")

# Load and prepare dataset
df = pd.read_csv("data/cleaned_dataset.csv")
texts = df["cleaned_text"].dropna().tolist()
dataset = Dataset.from_dict({"text": texts})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Padding token fix
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True).train_test_split(test_size=0.1)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2-wikitext2",  # Your original directory
    per_device_train_batch_size=8,  # Max for T4 GPU (2x faster)
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    fp16=True,                      # Mixed precision (2x speed)
    logging_steps=100,
    save_strategy="epoch",          # Saves checkpoints
    save_total_limit=2,
    eval_strategy="epoch",          # Evaluation enabled
    load_best_model_at_end=True,    # Keeps best model
    warmup_steps=100,               # Faster convergence
    report_to="wandb",              # Enables wandb logging
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,          # Required for eval_strategy
)

# Train the model
trainer.train()

# Save final model (same structure as original)
model.save_pretrained("./gpt2-wikitext2/final_model")
tokenizer.save_pretrained("./gpt2-wikitext2/final_model")


