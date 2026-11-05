import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# --- ADD THESE LINES TO FIX LEGACY NUMPY & SCIPY ISSUES ---
if not hasattr(np, 'float'):
    np.float = float

_original_solve = scipy.linalg.solve
def _patched_solve(a, b, **kwargs):
    if 'sym_pos' in kwargs:
        kwargs.pop('sym_pos')
        kwargs['assume_a'] = 'pos'
    return _original_solve(a, b, **kwargs)
scipy.linalg.solve = _patched_solve
# ----------------------------------------------------------

from megaman.geometry import Geometry
from megaman.embedding import (
    Isomap,
    LocallyLinearEmbedding,
    LTSA,
    SpectralEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "africa" / "uniform_5000"
RESULTS_DIR = ROOT / "results" / "africa"

# 1. Load the Africa dataset
data_path = os.path.join(DATA_DIR, "land_X.npy")
if not os.path.exists(data_path):
    data_path = os.path.join(DATA_DIR, "land_X.csv")
    X = np.loadtxt(data_path, delimiter=",")
else:
    X = np.load(data_path)

print(f"Loaded dataset: shape={X.shape}, dtype={X.dtype}")

# 2. Color code by original coordinates to track distortion
colors = X[:, 1]  # North-South latitude gradient

# 3. Create Geometry objects correctly using megaman syntax
# For Isomap, LLE, and LTSA, a k-nearest neighbor adjacency is standard.
geom_knn = Geometry(
    adjacency_method='auto',
    adjacency_kwds={'n_neighbors': 20}
)

# Spectral Embedding is highly sensitive to density, so megaman's creators 
# recommend a radius graph with a Geometric Laplacian.
# (Adjust radius based on the dataset's scale. 0.05 is a starting guess for scaled data).
radius = 0.05
geom_radius = Geometry(
    adjacency_method='auto',
    adjacency_kwds={'radius': radius},
    affinity_method='gaussian',
    affinity_kwds={'radius': radius},
    laplacian_method='geometric'
)

# 4. Initialize algorithms using the `geom` parameter AND explicit eigen_solvers
algorithms = {
    "Original 2D Africa": None,
    "Isomap": Isomap(
        n_components=2, geom=geom_knn, eigen_solver='dense'
    ),
    "Spectral Embedding": SpectralEmbedding(
        n_components=2, geom=geom_radius, eigen_solver='arpack'
    ),
    "Locally Linear": LocallyLinearEmbedding(
        n_components=2, geom=geom_knn, eigen_solver='arpack'
    ),
    "LTSA": LTSA(
        n_components=2, geom=geom_knn, eigen_solver='arpack'
    ),
}

# 5. Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# 6. Fit, transform, and plot
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for idx, (name, model) in enumerate(algorithms.items()):
    ax = axes[idx]
    print(f"Running {name}...")

    if model is None:
        Y = X
    else:
        Y = model.fit_transform(X)

    # Subplot in combined comparison
    ax.scatter(Y[:, 0], Y[:, 1], c=colors, cmap="viridis", s=3, alpha=0.8)
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")

    # Save individual high-res plot
    fig_single, ax_single = plt.subplots(figsize=(6, 6))
    ax_single.scatter(
        Y[:, 0], Y[:, 1], c=colors, cmap="viridis", s=4, alpha=0.8
    )
    ax_single.set_title(name, fontsize=14)
    ax_single.axis("equal")
    ax_single.set_xticks([])
    ax_single.set_yticks([])
    
    filename = name.lower().replace(" ", "_") + ".png"
    fig_single.savefig(
        os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight"
    )
    plt.close(fig_single)

# Save the full side-by-side figure
fig.tight_layout()
fig.savefig(
    os.path.join(RESULTS_DIR, "comparison_all_4.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nDone! Check the 'results/africa/' folder for your PNG outputs.")