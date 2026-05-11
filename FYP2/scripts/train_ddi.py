import os
import sys

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.smiles import load_drug_smiles
from utils.ddi_dataset import load_kaggle_ddi
from models.ddi_model import train_model

print("Loading DrugBank...")
drug_db = load_drug_smiles("data/raw/full_database.xml")

print("Loading DDI dataset...")
X, y = load_kaggle_ddi("data/raw/ddi_kaggle/ddi.csv", drug_db)

print("Training model...")
model = train_model(X, y)

print("Training complete.")
