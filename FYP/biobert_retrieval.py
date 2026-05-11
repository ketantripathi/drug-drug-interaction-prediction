# biobert_retrieval.py

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

model_name = "dmis-lab/biobert-base-cased-v1.1"
INDEX_EMB_PATH = "biobert_ddi_embeddings.npy"
INDEX_META_PATH = "biobert_ddi_metadata.csv"
DRUG_VOCAB_PATH = "drug_vocab.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading BioBERT on", device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("Loading DDI index...")
embeddings = np.load(INDEX_EMB_PATH)           # shape [N, 768]
meta_df = pd.read_csv(INDEX_META_PATH)

print("Loading drug vocabulary...")
with open(DRUG_VOCAB_PATH) as f:
    valid_drugs = {line.strip().lower() for line in f if line.strip()}

def is_valid_drug(name: str) -> bool:
    return str(name).strip().lower() in valid_drugs

def encode_pair(drug1: str, drug2: str) -> np.ndarray:
    text = f"Interaction between {drug1} and {drug2}"
    with torch.no_grad():
        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
        outputs = model(**enc)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return cls_emb  # shape [768]

def retrieve_interaction(drug1: str, drug2: str, top_k: int = 5, sim_threshold: float = 0.6):
    # 1) Drug validation
    if not is_valid_drug(drug1) or not is_valid_drug(drug2):
        return {
            "status": "invalid_drug",
            "message": "Unknown or unsupported drug name. Please enter valid existing drugs.",
            "results": []
        }

    query_emb = encode_pair(drug1, drug2).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]  # [N]

    # Get top-k indices
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_idx]

    # If best similarity is too low: abstain
    if top_scores[0] < sim_threshold:
        return {
            "status": "no_confident_match",
            "message": "No reliable interaction found in the knowledge base for this drug pair.",
            "results": []
        }

    results = []
    for idx, score in zip(top_idx, top_scores):
        row = meta_df.iloc[idx]
        results.append({
            "drug1": row["Drug 1"],
            "drug2": row["Drug 2"],
            "description": row["Interaction Description"],
            "similarity": float(score)
        })

    return {
        "status": "ok",
        "message": f"Found {len(results)} similar known interactions.",
        "results": results
    }

if __name__ == "__main__":
    # quick manual test
    out = retrieve_interaction("Aspirin", "Warfarin")
    print(out)