import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol = Chem.MolFromSmiles(smiles)

G = nx.Graph()

for atom in mol.GetAtoms():
    G.add_node(atom.GetIdx(), label=atom.GetSymbol())

for bond in mol.GetBonds():
    G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

pos = nx.spring_layout(G)

labels = nx.get_node_attributes(G, 'label')

nx.draw(G, pos, labels=labels, node_color='lightblue')
plt.title("Molecular Graph Representation")

plt.savefig("results/molecular_graph.png")
print("Saved molecular_graph.png")
