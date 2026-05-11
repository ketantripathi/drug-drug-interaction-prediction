# import torch
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from tqdm import tqdm
# import evaluate

# # -----------------------------
# # 1️⃣ Load ROUGE metric
# # -----------------------------
# metric = evaluate.load("rouge")

# # -----------------------------
# # 2️⃣ Load full dataset
# # -----------------------------
# DATA_PATH = "data/db_drug_interactions.csv"

# df = pd.read_csv(DATA_PATH)
# df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

# # Recreate EXACT same 80/20 split used during training
# _, test_df = train_test_split(df, test_size=0.2, random_state=42)

# print("📌 Test samples:", len(test_df))

# # -----------------------------
# # 3️⃣ Load trained model
# # -----------------------------
# MODEL_DIR = "drug_interaction_generator"
# tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# predictions = []
# references = []

# def generate(ddi_input):
#     inputs = tokenizer(ddi_input, return_tensors="pt")
#     outputs = model.generate(
#         **inputs,
#         max_length=70,
#         num_beams=5
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # -----------------------------
# # 4️⃣ Evaluate on test set
# # -----------------------------
# print("\n🚀 Running evaluation on test dataset...\n")

# for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#     text = f"Generate DDI between {row['Drug 1']} and {row['Drug 2']}"
#     pred = generate(text)
#     actual = row["Interaction Description"]

#     predictions.append(pred)
#     references.append(actual)

# # -----------------------------
# # 5️⃣ Compute ROUGE Scores
# # -----------------------------
# results = metric.compute(predictions=predictions, references=references)

# print("\n🔍 Evaluation Results:")
# print("ROUGE-1:", results['rouge1'])
# print("ROUGE-2:", results['rouge2'])
# print("ROUGE-L:", results['rougeL'])

# # -----------------------------
# # 6️⃣ Save detailed results
# # -----------------------------
# test_df["Predicted Interaction"] = predictions
# test_df.to_csv("test_results.csv", index=False)

# print("\n✅ Saved evaluation results to test_results.csv")
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import evaluate

# -----------------------------
# 1️⃣ Load ROUGE metric
# -----------------------------
metric = evaluate.load("rouge")

# -----------------------------
# 2️⃣ Load full dataset
# -----------------------------
DATA_PATH = "data/db_drug_interactions.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

# Recreate EXACT same 80/20 split used during training
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("📌 Total Test samples:", len(test_df))

# -----------------------------
# 3️⃣ Split test dataset into Test A (eval) and Test B (manual)
# -----------------------------
test_eval, test_manual = train_test_split(test_df, test_size=0.5, random_state=42)

print("📌 Test-Eval samples:", len(test_eval))
print("📌 Test-Manual samples:", len(test_manual))

# Save Test Manual set for later manual analysis
test_manual.to_csv("test_manual.csv", index=False)

# -----------------------------
# 4️⃣ Load trained model
# -----------------------------
MODEL_DIR = "drug_interaction_generator"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

predictions = []
references = []

def generate(ddi_input):
    inputs = tokenizer(ddi_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=70,
        num_beams=5
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 5️⃣ Evaluate on test_eval dataset
# -----------------------------
print("\n🚀 Running evaluation on Test-Eval subset...\n")

for _, row in tqdm(test_eval.iterrows(), total=len(test_eval)):
    text = f"Generate DDI between {row['Drug 1']} and {row['Drug 2']}"
    pred = generate(text)
    actual = row["Interaction Description"]

    predictions.append(pred)
    references.append(actual)

# -----------------------------
# 6️⃣ Compute ROUGE Scores
# -----------------------------
results = metric.compute(predictions=predictions, references=references)

print("\n🔍 Evaluation Results (On Test-Eval Set Only):")
print("ROUGE-1:", results["rouge1"])
print("ROUGE-2:", results["rouge2"])
print("ROUGE-L:", results["rougeL"])

# -----------------------------
# 7️⃣ Save evaluation results
# -----------------------------
test_eval["Predicted Interaction"] = predictions
test_eval.to_csv("test_results.csv", index=False)

print("\n💾 Saved predictions → test_results.csv")
print("💾 Saved manual test dataset → test_manual.csv")
