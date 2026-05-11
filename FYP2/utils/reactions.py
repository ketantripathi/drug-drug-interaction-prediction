from rdkit import Chem
from rdkit.Chem import AllChem

# Define simple realistic reaction templates
REACTIONS = [
    # Esterification (very common)
    "[C:1](=O)[O:2].[O:3]>>[C:1](=O)[O:3]",
    
    # Amide formation
    "[C:1](=O)[O:2].[N:3]>>[C:1](=O)[N:3]",
]

def generate_products(smiles1, smiles2):

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return []

    products = []

    for rxn_smarts in REACTIONS:
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            results = rxn.RunReactants((mol1, mol2))

            for r in results:
                try:
                    smi = Chem.MolToSmiles(r[0])
                    products.append(smi)
                except:
                    pass
        except:
            pass

    return list(set(products))