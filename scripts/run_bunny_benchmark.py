import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# --- LEGACY NUMPY & SCIPY PATCHES ---
if not hasattr(np, 'float'):
    np.float = float

_original_solve = scipy.linalg.solve
def _patched_solve(a, b, **kwargs):
    if 'sym_pos' in kwargs:
        kwargs.pop('sym_pos')
        kwargs['assume_a'] = 'pos'
    return _original_solve(a, b, **kwargs)
scipy.linalg.solve = _patched_solve
# ------------------------------------

from megaman.geometry import Geometry
from megaman.embedding import (
    Isomap,
    LocallyLinearEmbedding,
    LTSA,
    SpectralEmbedding,
)

# 1. Load the Bunny dataset
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data" / "bunny"
RESULTS_DIR = ROOT / "results" / "bunny"
data_path = os.path.join(folder, "X.npy")
if not os.path.exists(data_path):
    X = np.loadtxt(os.path.join(folder, "X.csv"), delimiter=",")
else:
    X = np.load(data_path)

print(f"Loaded Bunny dataset: shape={X.shape}, dtype={X.dtype}")

# Load colors (README says rgb.npy exists)
color_path = os.path.join(folder, "rgb.npy")
if os.path.exists(color_path):
    colors = np.load(color_path)
else:
    # Fallback to height.npy or raw Y-axis if rgb is missing
    colors = X[:, 1] 

# 2. Create Geometry objects
# Using k=20 for a 5000-point manifold is standard.
geom_knn = Geometry(
    adjacency_method='auto',
    adjacency_kwds={'n_neighbors': 20}
)

# A radius of 0.1 is a safe starting point for the normalized (0-1) bunny.
radius = 0.1
geom_radius = Geometry(
    adjacency_method='auto',
    adjacency_kwds={'radius': radius},
    affinity_method='gaussian',
    affinity_kwds={'radius': radius},
    laplacian_method='geometric'
)

# 3. Initialize algorithms
algorithms = {
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

# 4. Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# 5. Fit, transform, and plot
fig = plt.figure(figsize=(25, 5))

# Plot Original 3D Bunny first
ax3d = fig.add_subplot(1, 5, 1, projection='3d')
ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=3, alpha=0.8)
ax3d.set_title("Original 3D Bunny", fontsize=14, fontweight="bold")
# Adjust viewing angle for the Stanford Bunny
ax3d.view_init(elev=15, azim=-90)
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])

# Plot the 2D Embeddings
for idx, (name, model) in enumerate(algorithms.items()):
    ax = fig.add_subplot(1, 5, idx + 2)
    print(f"Running {name}...")

    # For Spectral, fallback to KNN if the radius is too small and disconnects the graph
    try:
        Y = model.fit_transform(X)
    except ValueError as e:
        print(f"  Warning: {name} failed with radius. Switching to KNN geometry...")
        model.geom = geom_knn
        Y = model.fit_transform(X)

    # Subplot in combined comparison
    ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=3, alpha=0.8)
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")

    # Save individual high-res plot
    fig_single, ax_single = plt.subplots(figsize=(6, 6))
    ax_single.scatter(
        Y[:, 0], Y[:, 1], c=colors, s=4, alpha=0.8
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

print("\nDone! Check the 'results/bunny/' folder for your PNG outputs.")