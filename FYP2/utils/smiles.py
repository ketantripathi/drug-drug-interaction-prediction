import xml.etree.ElementTree as ET

def load_drug_smiles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = root.tag.split("}")[0] + "}"

    drug_dict = {}

    for drug in root.findall(f"{ns}drug"):
        name = drug.find(f"{ns}name")

        smiles = None

        # search calculated properties
        props = drug.findall(f".//{ns}calculated-properties/{ns}property")

        for prop in props:
            kind = prop.find(f"{ns}kind")
            value = prop.find(f"{ns}value")

            if kind is not None and value is not None:
                if kind.text == "SMILES":
                    smiles = value.text
                    break

        if name is not None and smiles is not None:
            drug_dict[name.text.lower()] = smiles

    return drug_dict
