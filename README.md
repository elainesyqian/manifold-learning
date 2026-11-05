# Manifold Learning

Exploring how non-linear dimensionality reduction (manifold learning) algorithms
recover — or distort — the hidden geometry of data. The project compares
**Isomap**, **Spectral Embedding (Laplacian Eigenmaps / Diffusion Maps)**,
**Locally Linear Embedding (LLE)** and **LTSA** on datasets with known geometry
(the Africa and Americas landmasses, the Stanford Bunny), and is moving toward
higher-dimensional real data such as GloVe word embeddings.

Guiding questions:

- When is each algorithm the right choice, and which are most prone to artifacts?
- How can the Riemannian metric be used to measure how an embedding stretches or
  shrinks the data locally?
- What geometric structure is hidden in complex datasets like word vectors?

## Results so far

**Africa (2D, 5,000 area-uniform points)** — colored by latitude

![Africa comparison](results/africa/comparison_all_4.png)

**Stanford Bunny (3D surface, 5,000 points)** — colored by height

![Bunny comparison](results/bunny/comparison_all_4.png)

Individual high-resolution plots for each algorithm are in [`results/`](results/).

## Repository layout

```text
manifold-learning/
├── scripts/
│   ├── run_africa_benchmark.py   megaman: 4 algorithms on the Africa dataset
│   ├── run_bunny_benchmark.py    megaman: 4 algorithms on the Stanford Bunny
│   └── megaman_smoke_test.py     minimal check that megaman is installed
├── glove/                        convert + load GloVe vectors (see glove/README.md)
├── notebooks/                    scikit-learn / UMAP warm-ups: swiss roll, s-curve, digits, iris
├── results/                      generated figures (africa/, bunny/)
├── docs/notes/                   research notes (algorithm summaries, progress, open questions)
├── data/                         datasets — local only, not committed
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install cython && pip install git+https://github.com/mmp2/megaman.git
```

[megaman](https://github.com/mmp2/megaman) predates current NumPy/SciPy, so the
benchmark scripts patch `np.float` and `scipy.linalg.solve(sym_pos=...)` at import time.

## Data

Datasets are **not included** in this repository. Scripts expect them under `data/`:

| Path | Contents | Source |
|---|---|---|
| `data/africa/uniform_5000/` | 5,000 area-uniform mainland points (N×2, units of 1,000 km) | Natural Earth land/country shapefiles |
| `data/americas/{uniform,balanced,combined_10,combined_20}_5000/` | Four sampling profiles of the American mainland | Natural Earth |
| `data/bunny/` | 5,000 area-uniform surface samples (N×3, normalized) | [Stanford Bunny](https://graphics.stanford.edu/data/3Dscanrep/) |
| `data/glove/` | GloVe 6B 50d vectors converted to `.npy` | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

## Running

From the repository root:

```bash
python scripts/run_africa_benchmark.py   # writes results/africa/
python scripts/run_bunny_benchmark.py    # writes results/bunny/
```

## Notes

Research notes live in [`docs/notes/`](docs/notes/):

- [Dimension reduction from the user's perspective](docs/notes/dimension-reduction-users-perspective.md) — when to use manifold learning, algorithm walkthroughs, distortions
- [Work completed](docs/notes/work-completed.md) — progress log, next steps and open questions
