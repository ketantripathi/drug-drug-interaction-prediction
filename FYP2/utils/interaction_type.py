import json

with open("data/processed/drug_properties.json") as f:
    prop_db = json.load(f)

def predict_interaction_type(d1, d2):

    p1 = set(prop_db.get(d1, []))
    p2 = set(prop_db.get(d2, []))

    # generic rules

    if "blood_thinner" in p1 and "blood_thinner" in p2:
        return "synergistic"

    if "nsaid" in p1 and "nsaid" in p2:
        return "antagonistic"

    if p1 & p2:
        return "additive"

    return "neutral"