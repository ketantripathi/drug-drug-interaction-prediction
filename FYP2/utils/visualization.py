from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO


def smiles_to_image_base64(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(300, 300))

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str