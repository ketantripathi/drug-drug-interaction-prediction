# import torch
# import pandas as pd
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# MODEL_DIR = "drug_interaction_generator"
# DATA_PATH = "data/db_drug_interactions.csv"

# # -----------------------------
# # 1️⃣ Load model + tokenizer
# # -----------------------------
# tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# # -----------------------------
# # 2️⃣ Build vocab of known drugs
# # -----------------------------
# df = pd.read_csv(DATA_PATH)
# df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

# def norm(s: str) -> str:
#     return str(s).strip().lower()

# KNOWN_DRUGS = set(
#     pd.concat([df["Drug 1"], df["Drug 2"]])
#       .dropna()
#       .map(norm)
#       .unique()
# )

# def is_valid_drug(name: str) -> bool:
#     return norm(name) in KNOWN_DRUGS

# # -----------------------------
# # 3️⃣ Prediction function
# # -----------------------------
# def predict_ddi(drug1: str, drug2: str) -> str:
#     # ✅ Only check *individual* drug validity
#     missing = []
#     if not is_valid_drug(drug1):
#         missing.append(f"'{drug1}'")
#     if not is_valid_drug(drug2):
#         missing.append(f"'{drug2}'")

#     if missing:
#         return (
#             "❌ Unknown drug name(s): "
#             + ", ".join(missing)
#             + ". Please enter valid drugs from the dataset."
#         )

#     # ✅ We do NOT require (drug1, drug2) pair to exist in df
#     text = f"Generate DDI between {drug1} and {drug2}"
#     inputs = tokenizer(text, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         max_length=70,
#         num_beams=5,
#         return_dict_in_generate=True,
#         output_scores=True,
#     )

#     seq = outputs.sequences[0]
#     result = tokenizer.decode(seq, skip_special_tokens=True)

#     confidence = torch.exp(outputs.sequences_scores[0]).item()
#     if confidence < 0.35:
#         return "No strong interaction signal found for this pair."

#     return result


# if __name__ == "__main__":
#     print(predict_ddi("Aspirin", "Warfarin"))
#     print(predict_ddi("abc", "xyz"))  # should now be rejected
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# -----------------------------
# 1️⃣ Load model
# -----------------------------
MODEL_DIR = "drug_interaction_generator"
DATA_PATH = "data/db_drug_interactions.csv"

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# -----------------------------
# 2️⃣ Load dataset & build indexes
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])

df["drug1_norm"] = df["Drug 1"].str.lower().str.strip()
df["drug2_norm"] = df["Drug 2"].str.lower().str.strip()

VALID_DRUGS = sorted(set(df["drug1_norm"]) | set(df["drug2_norm"]))

# Represent each pair as a string for similarity search
pair_strings = (df["drug1_norm"] + " " + df["drug2_norm"]).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
pair_matrix = vectorizer.fit_transform(pair_strings)


# -----------------------------
# 3️⃣ Helper functions
# -----------------------------
def _norm(name: str) -> str:
    return str(name).strip().lower()


def resolve_drug(name: str, cutoff: float = 0.9):
    """
    Check if drug is in the known list.
    If not, try fuzzy match and return suggestion.
    """
    norm = _norm(name)
    if norm in VALID_DRUGS:
        return norm, None  # no warning

    candidates = get_close_matches(norm, VALID_DRUGS, n=1, cutoff=cutoff)
    if candidates:
        return candidates[0], f"Interpreting '{name}' as '{candidates[0]}'"
    return None, f"'{name}' is not found in the known drug list."


def lookup_known_interaction(d1_norm: str, d2_norm: str):
    """
    Check if exact (or reversed) pair exists in dataset.
    """
    mask = (
        ((df["drug1_norm"] == d1_norm) & (df["drug2_norm"] == d2_norm))
        | ((df["drug1_norm"] == d2_norm) & (df["drug2_norm"] == d1_norm))
    )
    subset = df[mask]
    if subset.empty:
        return None
    descs = subset["Interaction Description"].dropna().unique().tolist()
    return descs


def retrieve_similar_pairs(d1_norm: str, d2_norm: str, k: int = 5):
    """
    Find top-k most similar drug pairs in dataset using TF-IDF + cosine similarity.
    """
    query = f"{d1_norm} {d2_norm}"
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, pair_matrix)[0]
    top_idx = sims.argsort()[-k:][::-1]
    rows = df.iloc[top_idx][["Drug 1", "Drug 2", "Interaction Description"]].copy()
    rows["similarity"] = sims[top_idx]
    return rows


# -----------------------------
# 4️⃣ Hybrid prediction
# -----------------------------
def predict_ddi_hybrid(drug1: str, drug2: str):
    messages = []

    d1_norm, msg1 = resolve_drug(drug1)
    d2_norm, msg2 = resolve_drug(drug2)

    for msg in (msg1, msg2):
        if msg:
            messages.append(msg)

    # ❌ Unknown drugs → refuse instead of hallucinating
    if d1_norm is None or d2_norm is None:
        return {
            "mode": "invalid",
            "status": "❌ One or both drug names are not recognised in the dataset.",
            "kb_interaction": "",
            "model_prediction": "",
            "similar_examples": "",
            "notes": "\n".join(messages),
        }

    # ✅ 1) Check for exact known interaction
    known_descs = lookup_known_interaction(d1_norm, d2_norm)
    if known_descs:
        kb_text = "\n\n".join(f"- {txt}" for txt in known_descs[:5])
        status = "✅ Known interaction found in dataset."
        return {
            "mode": "knowledge",
            "status": status,
            "kb_interaction": kb_text,
            "model_prediction": "",
            "similar_examples": "",
            "notes": "\n".join(messages),
        }

    # ℹ️ 2) No direct record → use similar pairs + generation
    similar_rows = retrieve_similar_pairs(d1_norm, d2_norm, k=5)

    sim_lines = []
    for _, r in similar_rows.iterrows():
        sim_lines.append(
            f"{r['Drug 1']} + {r['Drug 2']} "
            f"(sim={r['similarity']:.2f}): {r['Interaction Description']}"
        )
    similar_block = "\n".join(sim_lines)

    # Build prompt with similar examples as context
    context_snippet = "\n".join(
        f"- {r['Drug 1']} + {r['Drug 2']}: {r['Interaction Description']}"
        for _, r in similar_rows.iterrows()
    )

    prompt = (
        f"Generate DDI between {drug1} and {drug2}. "
        f"Use these similar known interactions as pharmacological guidance:\n"
        f"{context_snippet}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    outputs = model.generate(
        **inputs,
        max_length=80,
        num_beams=5,
        return_dict_in_generate=True,
        output_scores=True,
    )

    seq = outputs.sequences[0]
    confidence = torch.exp(outputs.sequences_scores[0]).item()
    pred_text = tokenizer.decode(seq, skip_special_tokens=True)

    if confidence < 0.35:
        pred_text = (
            "No strong pharmacological interaction is predicted based on similar known interactions."
        )

    status = (
        "ℹ️ No direct record found; prediction generated using similar known interactions."
    )

    messages.append(f"Approximate model confidence: {confidence:.2f}")
    messages.append(
        "⚠️ This system is for research only and NOT for clinical decision making."
    )

    return {
        "mode": "hybrid",
        "status": status,
        "kb_interaction": "",
        "model_prediction": pred_text,
        "similar_examples": similar_block,
        "notes": "\n".join(messages),
    }


# Optional: keep the old simple function if you still need it somewhere
def predict_ddi(drug1: str, drug2: str) -> str:
    result = predict_ddi_hybrid(drug1, drug2)
    if result["mode"] == "invalid":
        return result["notes"] or result["status"]
    if result["mode"] == "knowledge":
        return result["kb_interaction"]
    return result["model_prediction"]
