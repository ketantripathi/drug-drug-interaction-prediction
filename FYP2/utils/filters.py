from rdkit import Chem
from rdkit.Chem import Descriptors

def is_valid(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def passes_drug_likeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Simple Lipinski-like rules
    if mw > 600:
        return False
    if logp > 5:
        return False

    return True