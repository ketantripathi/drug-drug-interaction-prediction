# build_biobert_index.py

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

DATA_PATH = "data/db_drug_interactions.csv"
INDEX_EMB_PATH = "biobert_ddi_embeddings.npy"
INDEX_META_PATH = "biobert_ddi_metadata.csv"
DRUG_VOCAB_PATH = "drug_vocab.txt"

model_name = "dmis-lab/biobert-base-cased-v1.1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading BioBERT on", device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

# Text we will encode
df["pair_text"] = "Interaction between " + df["Drug 1"] + " and " + df["Drug 2"]

# Save unique drug vocabulary for validation
all_drugs = pd.concat([df["Drug 1"], df["Drug 2"]]).dropna().unique()
with open(DRUG_VOCAB_PATH, "w") as f:
    for d in sorted(set(all_drugs)):
        f.write(str(d).strip() + "\n")
print(f"✅ Saved drug vocabulary with {len(all_drugs)} unique drugs to {DRUG_VOCAB_PATH}")

def encode_texts(texts, batch_size=32):
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BioBERT"):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            # Use [CLS] token representation: outputs.last_hidden_state[:, 0, :]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

print("Encoding all interaction pairs...")
embeddings = encode_texts(df["pair_text"].tolist(), batch_size=32)

print("Saving embeddings and metadata...")
np.save(INDEX_EMB_PATH, embeddings)

meta_cols = ["Drug 1", "Drug 2", "Interaction Description"]
df[meta_cols].to_csv(INDEX_META_PATH, index=False)

print(f"✅ Saved embeddings to {INDEX_EMB_PATH}")
print(f"✅ Saved metadata to {INDEX_META_PATH}")
