import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from utils.graph import fingerprint

smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",      # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # ibuprofen
    "CC(=O)NC1=CC=C(O)C=C1"          # paracetamol
]

fps = [fingerprint(s) for s in smiles]

n = len(fps)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        matrix[i][j] = sum(a == b for a, b in zip(fps[i], fps[j])) / len(fps[i])

plt.imshow(matrix, cmap="coolwarm")
plt.colorbar()
plt.title("Drug Similarity Matrix")

plt.savefig("results/similarity_matrix.png")
print("Saved results/similarity_matrix.png")