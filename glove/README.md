# GloVe vectors for manifold learning

The original `glove.6B.50d.txt` is kept in `data/glove/`. The conversion keeps
every token and the original row order, including punctuation and numbers.
No manifold embedding has been fitted; these are the 50-dimensional inputs.

```text
glove/
├── convert_glove.py          Reproducible converter; requires only NumPy
├── load_glove.py             Loader returning features and matching tokens
└── README.md

data/glove/                   (local only, not committed)
├── glove.6B.50d.txt          Original source (unchanged)
└── glove.6B.50d/
    ├── X.npy                 Original vectors, float32, shape (400000, 50)
    ├── X_unit.npy            L2-normalized vectors, float32, same shape
    ├── tokens.txt            One UTF-8 token per row
    └── metadata.json         Shape, preprocessing, and source SHA-256
```

The data files are too large for GitHub. Download `glove.6B.zip` from
<https://nlp.stanford.edu/projects/glove/>, place `glove.6B.50d.txt` in
`data/glove/`, then run the converter below.

Each matrix takes about 80 MB (76.3 MiB). NumPy `.npy` files preserve shape and
dtype and support memory mapping, so you can select rows without loading the
entire matrix. The numeric matrices contain features only; token labels are
stored separately. Row `i` corresponds to line `i + 1` in `tokens.txt`.

## Load and select a sample

From the project root, using the existing `.venv` environment:

```python
import numpy as np
from glove.load_glove import load_glove

X_all, tokens = load_glove()  # normalized vectors, read-only memory map
print(X_all.shape)           # (400000, 50)
print(tokens[0])             # the

# A reproducible, uniformly sampled subset of vocabulary entries.
# This samples vocabulary rows, not occurrences in the training corpus.
rng = np.random.default_rng(42)
indices = np.sort(rng.choice(len(tokens), size=5000, replace=False))
X = np.asarray(X_all[indices])
labels = [tokens[i] for i in indices]
# Keep indices if you save a fitted embedding, to recover the source rows.
```

For original, unnormalized vectors, use `load_glove(normalized=False)`.
To load without the helper:

```python
X = np.load("data/glove/glove.6B.50d/X_unit.npy", mmap_mode="r")
```

## Choosing the geometry

`X.npy` retains the learned vector magnitudes. `X_unit.npy` places every vector
on the unit sphere, a useful starting point for semantic neighborhood studies.
For unit vectors, squared Euclidean distance equals `2 * cosine_distance`, so
Euclidean and cosine distances produce the same nearest-neighbor ranking
(apart from numerical ties). Their distance values and downstream graph weights
still differ. Normalizing changes the geometry; choose the version that matches
your experiment. No standardization, PCA, token filtering, or sampling is baked
into either matrix.

Start with a subset such as 5,000 rows. A dense pairwise distance matrix for all
400,000 rows alone takes about 640 GB at float32 or 1.28 TB at float64. Memory
mapping the inputs does not remove this cost for algorithms that build dense
pairwise matrices. Full-vocabulary experiments should use methods with scalable
neighbor searches and sparse graphs.

## Reproduce the conversion

From the project root:

```sh
.venv/bin/python glove/convert_glove.py
```

The default destination must not already exist, to prevent accidental replacement.
For a separate regenerated copy:

```sh
.venv/bin/python glove/convert_glove.py --output /tmp/glove-reconverted
```

The converter checks row widths, numeric values, finite coordinates, and nonzero
vector lengths. It stages files before publishing the finished dataset and records
a SHA-256 of the source bytes. Values are stored as float32, introducing only the
normal rounding associated with that dtype. The source text is never modified.
