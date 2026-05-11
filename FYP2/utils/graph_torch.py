from rdkit import Chem
import torch
from torch_geometric.data import Data

def smiles_to_torch(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atoms = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        return None

    x = torch.tensor(atoms, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
