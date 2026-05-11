import json

with open("data/processed/drug_properties.json") as f:
    prop_db = json.load(f)

with open("data/processed/property_effects.json") as f:
    effect_db = json.load(f)

def predict_side_effects(d1, d2):

    effects = set()

    for drug in [d1, d2]:
        props = prop_db.get(drug, [])

        for p in props:
            effects.update(effect_db.get(p, []))

    return list(effects)