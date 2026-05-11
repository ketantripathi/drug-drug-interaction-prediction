from rdkit import Chem
from rdkit.Chem import Draw

drugs = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
}

mols = [Chem.MolFromSmiles(s) for s in drugs.values()]

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=2,
    subImgSize=(300,300),
    legends=list(drugs.keys())
)

img.save("results/molecules.png")
print("Saved molecules.png")
