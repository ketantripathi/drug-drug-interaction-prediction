from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    bonds = []
    for bond in mol.GetBonds():
        bonds.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ))

    return {
        "atoms": atoms,
        "bonds": bonds
    }


def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    # skip invalid molecules
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    return list(fp)