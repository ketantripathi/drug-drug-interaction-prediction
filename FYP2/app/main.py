from fastapi import FastAPI
from difflib import get_close_matches
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.smiles import load_drug_smiles
from utils.graph import smiles_to_graph, fingerprint
from utils.graph_torch import smiles_to_torch

import torch
from models.gnn_ddi import DDINet

from utils.visualization import smiles_to_image_base64
from utils.side_effects import predict_side_effects
from utils.scoring import compute_severity, compute_harm_score
from utils.interaction_type import predict_interaction_type
from utils.mechanism import predict_mechanism
from utils.knowledge_graph import get_reasoning_path
from utils.explanation import generate_explanation
from utils.molecule_generator import (
    combine_molecules,
    should_generate_new_drug,
    analyze_generated
)

app = FastAPI()

# -------------------
# LOAD DATABASE
# -------------------
xml_path = os.path.join("data", "raw", "full_database.xml")
drug_db = load_drug_smiles(xml_path)

# -------------------
# SYNONYMS
# -------------------
synonyms = {
    "aspirin": "acetylsalicylic acid",
    "paracetamol": "acetaminophen"
}

reverse_synonyms = {v: k for k, v in synonyms.items()}

def normalize_name(name):
    return reverse_synonyms.get(name, name)

# -------------------
# LOAD GNN MODEL
# -------------------
gnn_model = DDINet()
gnn_model.load_state_dict(
    torch.load("checkpoints/gnn_model.pt", map_location=torch.device("cpu"))
)
gnn_model.eval()

# -------------------
# LOOKUP
# -------------------
@app.get("/lookup")
def lookup(drug: str):

    drug = drug.lower()
    drug = synonyms.get(drug, drug)

    if drug in drug_db:
        smiles = drug_db[drug]
        return {
            "drug": drug,
            "smiles": smiles,
            "graph": smiles_to_graph(smiles),
            "fingerprint": fingerprint(smiles)[:32]
        }

    match = get_close_matches(drug, drug_db.keys(), n=1, cutoff=0.6)

    if match:
        best = match[0]
        return {
            "drug": best,
            "smiles": drug_db[best],
            "note": "closest match used"
        }

    return {"error": "drug not found"}


# -------------------
# PREDICT
# -------------------
@app.get("/predict")
def predict(d1: str, d2: str):

    d1 = synonyms.get(d1.lower(), d1.lower())
    d2 = synonyms.get(d2.lower(), d2.lower())

    if d1 not in drug_db or d2 not in drug_db:
        return {"error": "drug not found"}

    # -------------------
    # GNN Prediction
    # -------------------
    g1 = smiles_to_torch(drug_db[d1])
    g2 = smiles_to_torch(drug_db[d2])

    if g1 is None or g2 is None:
        return {"error": "invalid molecule"}

    with torch.no_grad():
        out = gnn_model(g1, g2)
        probs = torch.softmax(out, dim=1)

        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    # -------------------
    # NORMALIZATION
    # -------------------
    d1_norm = normalize_name(d1)
    d2_norm = normalize_name(d2)

    # -------------------
    # SYMBOLIC REASONING
    # -------------------
    interaction_type = predict_interaction_type(d1_norm, d2_norm)
    mechanism = predict_mechanism(interaction_type)
    reasoning = get_reasoning_path(d1_norm, d2_norm)

    effects = predict_side_effects(d1_norm, d2_norm)

    severity = compute_severity(effects, interaction_type)
    harm_score = compute_harm_score(pred, confidence, effects, interaction_type)

    risk = (
        "High Risk" if harm_score < -40
        else "Moderate Risk" if harm_score < -10
        else "Low Risk"
    )

    explanation = generate_explanation(
        d1_norm, d2_norm, interaction_type, effects
    )

    # -------------------
    # NEW DRUG GENERATION
    # -------------------
    from utils.molecule_generator import generate_new_drug

new_drug = {
    "generated": False,
    "candidates": []
}

try:
    if should_generate_new_drug(pred, confidence):

        candidates = generate_new_drug(drug_db[d1], drug_db[d2])

        if candidates:

            analyzed = []

            for smi in candidates:
                analysis = analyze_generated(smi, gnn_model)
                analyzed.append({
                    "smiles": smi,
                    "analysis": analysis
                })

            new_drug = {
                "generated": True,
                "candidates": analyzed
            }

except Exception as e:
    new_drug = {"generated": False, "error": str(e)}
    
    # -------------------
    # MOLECULAR VISUALIZATION (NEW)
    # -------------------

    drug1_img = smiles_to_image_base64(drug_db[d1])
    drug2_img = smiles_to_image_base64(drug_db[d2])

    new_drug_img = None
    if new_drug.get("generated") and new_drug.get("smiles"):
        new_drug_img = smiles_to_image_base64(new_drug["smiles"])

    # -------------------
    # RESPONSE
    # -------------------
    return {
        "drug1": d1,
        "drug2": d2,
        "prediction": int(pred),
        "confidence": round(confidence, 3),

        "interaction_type": interaction_type,
        "mechanism": mechanism,

        "side_effects": effects,
        "severity": severity,
        "harm_score": harm_score,
        "risk_level": risk,

        "reasoning_path": reasoning,
        "explanation": explanation,

        "new_drug": new_drug,

        # 🔥 NEW FIELD
        "visualization": {
            "drug1": drug1_img,
            "drug2": drug2_img,
            "new_drug": new_drug_img
        }
    }