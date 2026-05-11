from utils.graph_torch import smiles_to_torch

def build_dataset(pairs, drug_db):

    dataset = []

    for d1, d2, label in pairs:

        g1 = smiles_to_torch(drug_db[d1])
        g2 = smiles_to_torch(drug_db[d2])

        if g1 is None or g2 is None:
            continue

        dataset.append((g1, g2, label))

    return dataset