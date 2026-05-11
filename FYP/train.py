# ----------------------------------------------
# 1️⃣ LOAD DATASET
# ----------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/db_drug_interactions.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

df["input_text"] = "Generate DDI between " + df["Drug 1"] + " and " + df["Drug 2"]
df["target_text"] = df["Interaction Description"]

print("✅ Loaded dataset with", len(df), "entries")
print(df.sample(3))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ----------------------------------------------
# 2️⃣ LOAD MODEL
# ----------------------------------------------
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ----------------------------------------------
# 3️⃣ TOKENIZE DATA
# ----------------------------------------------
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_fn(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length",
                       truncation=True, max_length=64)
    targets = tokenizer(batch["target_text"], padding="max_length",
                        truncation=True, max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_tokenized = train_dataset.map(tokenize_fn, batched=True)
test_tokenized = test_dataset.map(tokenize_fn, batched=True)

# ----------------------------------------------
# 4️⃣ TRAINING ARGUMENTS
# ----------------------------------------------
from transformers import TrainingArguments, Trainer
import os

checkpoint_dir = "checkpoints"

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    num_train_epochs=4,
    learning_rate=3e-5,
    save_total_limit=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized
)

# ----------------------------------------------
# 5️⃣ TRAIN MODEL
# ----------------------------------------------
last_checkpoint = None

if os.path.isdir(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
    last_checkpoint = max(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1])
    )
    last_checkpoint = f"{checkpoint_dir}/{last_checkpoint}"
    print(" Resuming from checkpoint:", last_checkpoint)
else:
    print(" Starting fresh training...")

trainer.train(resume_from_checkpoint=last_checkpoint)

model.save_pretrained("drug_interaction_generator")
tokenizer.save_pretrained("drug_interaction_generator")

print("💾 Model saved successfully!")


