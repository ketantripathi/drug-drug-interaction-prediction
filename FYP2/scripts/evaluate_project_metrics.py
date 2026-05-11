import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DDI_PATH = ROOT / "data/raw/ddi_kaggle/ddi.csv"
XML_PATH = ROOT / "data/raw/full_database.xml"
PROCESSED_DIR = ROOT / "data/processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUT_PATH = ROOT / "results/project_metrics.json"


def norm(name):
    return str(name).strip().lower()


def canonical_pair(a, b):
    a, b = norm(a), norm(b)
    return tuple(sorted((a, b)))


def load_drugbank_smiles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}"

    all_drugs = set()
    smiles = {}
    for drug in root.findall(f"{ns}drug"):
        name_el = drug.find(f"{ns}name")
        if name_el is None or not name_el.text:
            continue
        name = norm(name_el.text)
        all_drugs.add(name)

        found_smiles = None
        for prop in drug.findall(f".//{ns}calculated-properties/{ns}property"):
            kind = prop.find(f"{ns}kind")
            value = prop.find(f"{ns}value")
            if kind is not None and value is not None and kind.text == "SMILES":
                found_smiles = value.text
                break
        if found_smiles:
            smiles[name] = found_smiles

    return all_drugs, smiles


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


def interaction_category(text):
    text = norm(text)
    for name, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, text):
            return name
    return "other"


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main():
    random.seed(42)
    df = pd.read_csv(DDI_PATH).dropna(subset=["Drug 1", "Drug 2", "Interaction Description"])
    df["drug1_norm"] = df["Drug 1"].map(norm)
    df["drug2_norm"] = df["Drug 2"].map(norm)
    df["canonical_pair"] = [canonical_pair(a, b) for a, b in zip(df["drug1_norm"], df["drug2_norm"])]
    df["category"] = df["Interaction Description"].map(interaction_category)

    ddi_drugs = set(df["drug1_norm"]) | set(df["drug2_norm"])
    all_drugbank, smiles = load_drugbank_smiles(XML_PATH)
    smiles_drugs = set(smiles)

    both_in_drugbank = df["drug1_norm"].isin(all_drugbank) & df["drug2_norm"].isin(all_drugbank)
    both_have_smiles = df["drug1_norm"].isin(smiles_drugs) & df["drug2_norm"].isin(smiles_drugs)
    one_has_smiles = df["drug1_norm"].isin(smiles_drugs) | df["drug2_norm"].isin(smiles_drugs)

    side_effects = load_json(PROCESSED_DIR / "side_effects.json")
    drug_properties = load_json(PROCESSED_DIR / "drug_properties.json")
    interaction_types = load_json(PROCESSED_DIR / "interaction_types.json")
    property_effects = load_json(PROCESSED_DIR / "property_effects.json")

    side_effect_drugs = set(map(norm, side_effects.keys()))
    property_drugs = set(map(norm, drug_properties.keys()))
    rows_with_side_effect_rule = df["drug1_norm"].isin(side_effect_drugs) | df["drug2_norm"].isin(side_effect_drugs)
    rows_with_property_rule = df["drug1_norm"].isin(property_drugs) | df["drug2_norm"].isin(property_drugs)

    # Estimate random negative collision with known positives. This checks whether
    # the training script's random-negative strategy is likely to sample hidden positives.
    canonical_positives = set(df["canonical_pair"])
    drug_list = sorted(ddi_drugs)
    trials = 20000
    collisions = 0
    for _ in range(trials):
        if canonical_pair(random.choice(drug_list), random.choice(drug_list)) in canonical_positives:
            collisions += 1

    metrics = {
        "ddi_rows": int(len(df)),
        "unique_ordered_pairs": int(df[["drug1_norm", "drug2_norm"]].drop_duplicates().shape[0]),
        "unique_unordered_pairs": int(len(canonical_positives)),
        "duplicate_ordered_rows": int(len(df) - df[["drug1_norm", "drug2_norm"]].drop_duplicates().shape[0]),
        "unique_ddi_drugs": int(len(ddi_drugs)),
        "drugbank_total_drugs": int(len(all_drugbank)),
        "drugbank_drugs_with_smiles": int(len(smiles_drugs)),
        "ddi_drug_name_coverage_in_drugbank": float(len(ddi_drugs & all_drugbank) / len(ddi_drugs)),
        "ddi_drug_smiles_coverage": float(len(ddi_drugs & smiles_drugs) / len(ddi_drugs)),
        "ddi_rows_both_drugs_in_drugbank": int(both_in_drugbank.sum()),
        "ddi_rows_both_drugs_in_drugbank_rate": float(both_in_drugbank.mean()),
        "ddi_rows_at_least_one_smiles": int(one_has_smiles.sum()),
        "ddi_rows_at_least_one_smiles_rate": float(one_has_smiles.mean()),
        "ddi_rows_both_smiles": int(both_have_smiles.sum()),
        "ddi_rows_both_smiles_rate": float(both_have_smiles.mean()),
        "interaction_category_distribution": dict(Counter(df["category"]).most_common()),
        "side_effect_rule_drugs": int(len(side_effect_drugs)),
        "property_rule_drugs": int(len(property_drugs)),
        "interaction_type_rules": int(len(interaction_types)),
        "property_effect_rules": int(len(property_effects)),
        "ddi_rows_with_side_effect_rule": int(rows_with_side_effect_rule.sum()),
        "ddi_rows_with_side_effect_rule_rate": float(rows_with_side_effect_rule.mean()),
        "ddi_rows_with_property_rule": int(rows_with_property_rule.sum()),
        "ddi_rows_with_property_rule_rate": float(rows_with_property_rule.mean()),
        "gnn_checkpoint_exists": (CHECKPOINT_DIR / "gnn_model.pt").exists(),
        "random_forest_checkpoint_exists": (CHECKPOINT_DIR / "ddi_model.pkl").exists(),
        "random_negative_trials": trials,
        "random_negative_known_positive_collision_rate": float(collisions / trials),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nSaved metrics to {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
