import numpy as np
from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding

# 1. Generate synthetic data (100 samples, 10 features)
np.random.seed(42)
X = np.random.randn(100, 10)

# 2. Configure embedding geometry
radius = 5.0
n_components = 2  # target dimension (e.g., 2D projection)

geom = Geometry(
    adjacency_method='cyflann',
    adjacency_kwds={'radius': radius},
    affinity_method='gaussian',
    affinity_kwds={'radius': radius},
    laplacian_method='geometric',
    laplacian_kwds={'scaling_epps': radius}
)

spectral = SpectralEmbedding(n_components=n_components, eigen_solver='arpack', geom=geom)
embed_spectral = spectral.fit_transform(X)

print("Original shape:", X.shape)
print("Embedded shape:", embed_spectral.shape)
print("First 3 embedded rows:\n", embed_spectral[:3])
