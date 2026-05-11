import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:
    sentence_bleu = None
    SmoothingFunction = None


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "test_results_with_similarity.csv"
OUT_PATH = ROOT / "text_generation_metrics.json"


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def token_f1(reference, prediction):
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref and not pred:
        return 1.0, 1.0, 1.0
    if not ref or not pred:
        return 0.0, 0.0, 0.0
    ref_counts = Counter(ref)
    pred_counts = Counter(pred)
    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_l(reference, prediction):
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0

    # Two-row dynamic programming for LCS length.
    prev = [0] * (len(pred) + 1)
    for token in ref:
        curr = [0]
        for j, pred_token in enumerate(pred, start=1):
            if token == pred_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


CATEGORY_PATTERNS = {
    "metabolism_decreased": r"metabolism .* decreased",
    "metabolism_increased": r"metabolism .* increased",
    "serum_increased": r"(serum concentration|serum level).* increased|higher serum",
    "serum_decreased": r"(serum concentration|serum level).* decreased|lower serum",
    "excretion_decreased": r"excretion rate .* decreased",
    "excretion_increased": r"excretion rate .* increased",
    "adverse_effects": r"risk or severity of adverse effects",
    "therapeutic_efficacy": r"therapeutic efficacy",
    "activity_increased": r"may increase .* activities",
    "activity_decreased": r"may decrease .* activities",
    "qtc": r"qtc-prolonging|qt prolong",
    "photosensitizing": r"photosensitizing",
}


def category(text):
    norm = normalize_text(text)
    for name, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, norm):
            return name
    return "other"


def contains_direction(text):
    norm = normalize_text(text)
    if "increase" in norm or "increased" in norm or "higher" in norm:
        if "decrease" in norm or "decreased" in norm or "lower" in norm:
            return "mixed"
        return "increase"
    if "decrease" in norm or "decreased" in norm or "lower" in norm:
        return "decrease"
    return "none"


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main():
    df = pd.read_csv(RESULTS_PATH)
    refs = df["Interaction Description"].fillna("").astype(str).tolist()
    preds = df["Predicted Interaction"].fillna("").astype(str).tolist()

    exact = [normalize_text(r) == normalize_text(p) for r, p in zip(refs, preds)]

    if "Similarity Score" in df.columns:
        similarities = df["Similarity Score"].astype(float).to_numpy()
    else:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(refs + preds)
        similarities = cosine_similarity(matrix[: len(refs)], matrix[len(refs) :]).diagonal()

    p_vals, r_vals, f_vals, rouge_vals, bleu_vals = [], [], [], [], []
    smoothie = SmoothingFunction().method1 if SmoothingFunction else None
    for ref, pred in zip(refs, preds):
        p, r, f = token_f1(ref, pred)
        p_vals.append(p)
        r_vals.append(r)
        f_vals.append(f)
        rouge_vals.append(rouge_l(ref, pred))
        if sentence_bleu:
            bleu_vals.append(sentence_bleu([tokenize(ref)], tokenize(pred), smoothing_function=smoothie))

    ref_categories = [category(x) for x in refs]
    pred_categories = [category(x) for x in preds]
    category_matches = [r == p for r, p in zip(ref_categories, pred_categories)]

    ref_directions = [contains_direction(x) for x in refs]
    pred_directions = [contains_direction(x) for x in preds]
    direction_matches = [r == p for r, p in zip(ref_directions, pred_directions)]

    drug1_mentioned = [
        normalize_text(row["Drug 1"]) in normalize_text(pred)
        for pred, (_, row) in zip(preds, df.iterrows())
    ]
    drug2_mentioned = [
        normalize_text(row["Drug 2"]) in normalize_text(pred)
        for pred, (_, row) in zip(preds, df.iterrows())
    ]

    actual_lengths = df.get("Actual Length", pd.Series([len(tokenize(x)) for x in refs])).astype(float)
    pred_lengths = df.get("Predicted Length", pd.Series([len(tokenize(x)) for x in preds])).astype(float)
    length_abs_error = (actual_lengths - pred_lengths).abs().to_numpy()

    metrics = {
        "records": int(len(df)),
        "exact_match_rate": float(np.mean(exact)),
        "tfidf_cosine_mean": float(np.mean(similarities)),
        "tfidf_cosine_median": float(np.median(similarities)),
        "tfidf_cosine_std": float(np.std(similarities, ddof=1)),
        "tfidf_cosine_min": float(np.min(similarities)),
        "tfidf_cosine_p05": percentile(similarities, 5),
        "tfidf_cosine_p25": percentile(similarities, 25),
        "tfidf_cosine_p75": percentile(similarities, 75),
        "tfidf_cosine_p95": percentile(similarities, 95),
        "token_precision_mean": float(np.mean(p_vals)),
        "token_recall_mean": float(np.mean(r_vals)),
        "token_f1_mean": float(np.mean(f_vals)),
        "rouge_l_f1_mean": float(np.mean(rouge_vals)),
        "bleu_mean": float(np.mean(bleu_vals)) if bleu_vals else None,
        "category_match_rate": float(np.mean(category_matches)),
        "direction_match_rate": float(np.mean(direction_matches)),
        "drug1_mention_rate": float(np.mean(drug1_mentioned)),
        "drug2_mention_rate": float(np.mean(drug2_mentioned)),
        "both_drugs_mention_rate": float(np.mean(np.logical_and(drug1_mentioned, drug2_mentioned))),
        "actual_length_mean": float(actual_lengths.mean()),
        "predicted_length_mean": float(pred_lengths.mean()),
        "length_abs_error_mean": float(np.mean(length_abs_error)),
        "length_abs_error_median": float(np.median(length_abs_error)),
        "reference_category_distribution": dict(Counter(ref_categories).most_common()),
        "prediction_category_distribution": dict(Counter(pred_categories).most_common()),
    }

    OUT_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nSaved metrics to {OUT_PATH}")


if __name__ == "__main__":
    main()
