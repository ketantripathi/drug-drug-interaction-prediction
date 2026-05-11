from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from utils.graph_torch import smiles_to_torch


from utils.reactions import generate_products
from utils.filters import is_valid, passes_drug_likeness


def generate_new_drug(smiles1, smiles2):

    candidates = generate_products(smiles1, smiles2)

    valid = []

    for smi in candidates:
        if is_valid(smi) and passes_drug_likeness(smi):
            valid.append(smi)

    return valid[:3]  # return top 3 candidates


def should_generate_new_drug(pred, confidence):
    return pred == 1 and confidence > 0.5


def analyze_generated(smiles, gnn_model):

    g = smiles_to_torch(smiles)

    if g is None:
        return None

    with torch.no_grad():
        out = gnn_model(g, g)
        probs = torch.softmax(out, dim=1)

    return {
        "interaction_score": float(probs[0][1])
    }