import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from utils.smiles import load_drug_smiles

# load drug database
drug_db = load_drug_smiles("data/raw/full_database.xml")

# load Kaggle dataset
df = pd.read_csv("data/raw/ddi_kaggle/ddi.csv")

col1, col2 = df.columns[:2]

valid_pairs = 0

for _, row in df.iterrows():
    d1 = str(row[col1]).lower()
    d2 = str(row[col2]).lower()

    if d1 in drug_db and d2 in drug_db:
        valid_pairs += 1

# negatives are generated to match positives
neg_pairs = valid_pairs

labels = ["Known Interactions", "Generated Non-Interactions"]
counts = [valid_pairs, neg_pairs]

plt.bar(labels, counts)
plt.title("DDI Dataset Distribution (Real Counts)")
plt.ylabel("Number of Samples")

for i, v in enumerate(counts):
    plt.text(i, v + 5, str(v), ha="center")

plt.savefig("results/dataset_distribution_real.png")
print("Saved results/dataset_distribution_real.png")
print("Valid positive pairs:", valid_pairs)
print("Generated negatives:", neg_pairs)
