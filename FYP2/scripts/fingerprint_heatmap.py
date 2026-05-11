import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from utils.graph import fingerprint

smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
]

fps = [fingerprint(s) for s in smiles]

plt.imshow(fps, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("Chemical Fingerprint Heatmap")
plt.xlabel("Fingerprint bits")
plt.ylabel("Drug index")

plt.savefig("results/fingerprint_heatmap.png")
print("Saved fingerprint_heatmap.png")
