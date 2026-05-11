import pandas as pd
import random
from utils.graph import fingerprint

def load_kaggle_ddi(path, drug_db):

    df = pd.read_csv(path)

    # auto-detect first two columns
    col1, col2 = df.columns[:2]
    print("Using columns:", col1, col2)

    data = []
    skipped = 0

    # -------------------
    # Positive samples
    # -------------------
    for _, row in df.iterrows():

        d1 = str(row[col1]).lower()
        d2 = str(row[col2]).lower()

        if d1 not in drug_db or d2 not in drug_db:
            continue

        fp1 = fingerprint(drug_db[d1])
        fp2 = fingerprint(drug_db[d2])

        # skip invalid molecules
        if fp1 is None or fp2 is None:
            skipped += 1
            continue

        data.append((fp1 + fp2, 1))

    print("Positive samples:", len(data))
    print("Skipped invalid molecules:", skipped)

    # -------------------
    # Negative samples
    # -------------------
    drugs = list(drug_db.keys())
    neg = []

    while len(neg) < len(data):

        d1 = random.choice(drugs)
        d2 = random.choice(drugs)

        fp1 = fingerprint(drug_db[d1])
        fp2 = fingerprint(drug_db[d2])

        if fp1 is None or fp2 is None:
            continue

        neg.append((fp1 + fp2, 0))

    # combine
    data.extend(neg)
    random.shuffle(data)

    X = [x for x, y in data]
    y = [y for x, y in data]

    print("Final dataset size:", len(X))

    return X, y

def load_ddi_pairs(path, drug_db):

    df = pd.read_csv(path)
    col1, col2 = df.columns[:2]

    pairs = []

    for _, row in df.iterrows():
        d1 = str(row[col1]).lower()
        d2 = str(row[col2]).lower()

        if d1 in drug_db and d2 in drug_db:
            pairs.append((d1, d2, 1))

    # negative sampling
    drugs = list(drug_db.keys())

    while len(pairs) < 2 * len(pairs):
        d1 = random.choice(drugs)
        d2 = random.choice(drugs)
        pairs.append((d1, d2, 0))

    random.shuffle(pairs)

    return pairs
