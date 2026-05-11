import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random

from models.gnn_ddi import DDINet
from utils.smiles import load_drug_smiles
from utils.graph_torch import smiles_to_torch

print("Loading DrugBank...")
drug_db = load_drug_smiles("data/raw/full_database.xml")

print("Loading CSV...")
df = pd.read_csv("data/raw/ddi_kaggle/ddi.csv")

col1, col2 = df.columns[:2]
print("Using columns:", col1, col2)

model = DDINet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

drugs = list(drug_db.keys())

print("Training GNN (true streaming)...")

for epoch in range(3):

    total_loss = 0
    count = 0

    for i, row in df.iterrows():

        # LIMIT DATA
        if i > 1500:
            break

        d1 = str(row[col1]).lower()
        d2 = str(row[col2]).lower()

        # positive sample
        if d1 in drug_db and d2 in drug_db:

            g1 = smiles_to_torch(drug_db[d1])
            g2 = smiles_to_torch(drug_db[d2])

            if g1 is None or g2 is None:
                continue

            optimizer.zero_grad()
            out = model(g1, g2)
            loss = loss_fn(out, torch.tensor([1]))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        # negative sample (random)
        d1n = random.choice(drugs)
        d2n = random.choice(drugs)

        g1 = smiles_to_torch(drug_db[d1n])
        g2 = smiles_to_torch(drug_db[d2n])

        if g1 is None or g2 is None:
            continue

        optimizer.zero_grad()
        out = model(g1, g2)
        loss = loss_fn(out, torch.tensor([0]))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Samples: {count}")

torch.save(model.state_dict(), "checkpoints/gnn_model.pt")

print("GNN training complete.")