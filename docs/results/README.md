# Results

One page per dataset, each figure with a short description:

| Dataset | Page | Input | Why it's interesting |
|---|---|---|---|
| Africa | [africa.md](africa.md) | 5,000 pts, 2D map | Flat, compact shape: every algorithm *should* be able to recover it |
| Americas | [americas.md](americas.md) | 5,000 pts, 2D map, 2 sampling profiles | Long, thin, with a narrow isthmus: a stress test for neighbourhood graphs |
| Stanford Bunny | [bunny.md](bunny.md) | 5,000 pts on a 3D surface | Curved, closed surface with thin ears: cannot be flattened without distortion |
| GloVe | [glove.md](glove.md) | 5,000 words, 50D | Real high-dimensional data with unknown geometry |

All figures are produced by:

```bash
python scripts/run_africa_benchmark.py    # results/africa/*.png      (first-pass comparison)
python scripts/run_bunny_benchmark.py     # results/bunny/*.png
python scripts/riemannian_metric.py       # results/<dataset>/riemannian/  (everything else)
```

## Method: the dual Riemannian metric

An embedding `f` maps each data point `p` to `f(p)` in 2D. Near `p`, `f` acts like a
linear map. It may stretch some directions and shrink others. The **dual Riemannian
metric** `H(p)` is a 2×2 symmetric matrix that measures this. It is estimated from the
embedding coordinates and a Laplacian built on the *original* data (Perrault-Joncas &
Meilă, 2013; `megaman.geometry.rmetric.RiemannMetric`):

```
H(p)_ij = ½ [ L(f_i f_j) − f_i L(f_j) − f_j L(f_i) ](p)
```

Its eigen-decomposition `H = λ_long v_long v_longᵀ + λ_short v_short v_shortᵀ` gives:

| Quantity | Meaning |
|---|---|
| `v_long`, `v_short` | Directions in the embedding that get stretched most and least |
| `√λ_long`, `√λ_short` | How long a unit step on the data becomes in the embedding along those directions |
| `√(λ_long / λ_short)` (**stretch ratio**) | Shape distortion. 1 means angles are preserved (a circle stays a circle); 10 means a circle becomes a 10:1 ellipse |
| `√(λ_long · λ_short)` (**area scale**) | Local area magnification |

**Reading the ellipses.** Each ellipse is `H(p)` drawn at `f(p)`: its axes lie along the
eigenvectors and have length `√λ`. Colour is `log10(stretch ratio)`: dark purple is
isotropic, yellow is ≥10:1. All ellipses in a panel share one scale factor, printed in
the corner. If one panel's ellipses look small, its embedding has a small overall
scale, not necessarily less distortion. Ellipses more than 5× the median length are
capped, keeping their shape, and drawn dashed.

**Sanity check.** For the 2D maps, the "Original" panel embeds the data as itself, so
`H` should be the identity. We measure a median stretch ratio of 1.07 and an area
scale of 0.94–0.95. This is the estimator's noise floor; values within ~10% of it are
"no distortion". Ellipses at the coastline are slightly flattened because the
Laplacian only sees neighbours on one side.

**Important.** Spectral Embedding, LLE and LTSA return unit-norm eigenvectors, so their
overall scale (and therefore the *absolute* area scale) is arbitrary. Compare them by
**stretch ratio** and by **area-scale spread** (how much the area scale varies across
the dataset), not by raw `λ`.

## Settings

| | Africa | Americas | Bunny | GloVe |
|---|---|---|---|---|
| k-NN graph (Isomap, LLE, LTSA) | k = 20 | k = 20 | k = 20 | k = 20 |
| Laplacian kernel bandwidth `r` | 0.25 | 0.28 | 0.07 | 0.6 |
| Points with ellipses | 80 | 80 | 80 | 80 |

`r` is about 1.8× the median distance to the 10th nearest neighbour. Every graph is
connected. Sample points are picked by farthest-point sampling on the input data, so
they cover it evenly. The same 80 points (and the same first 12 in the detail figures)
are used for every algorithm within a dataset.

## Files per dataset (`results/<dataset>/riemannian/`)

| File | Contents |
|---|---|
| `embeddings.png` | All embeddings side by side, coloured by a ground-truth variable |
| `ellipses.png` | `H` as ellipses at 80 sampled points, one panel per algorithm |
| `distortion_map.png` | Stretch ratio (top) and area scale (bottom) at **every** point |
| `detail_<algorithm>.png` | 12 numbered points with a table of `λ_long`, `λ_short`, `v_long`, stretch ratio |
| `samples.csv` | Same numbers for all 80 sampled points × all algorithms (both eigenvectors) |
| `summary.csv` | Per-algorithm distortion statistics (the tables in each page) |
| `metric.npz` | Full embeddings `Y_*` and dual metrics `H_*` for all 5,000 points |

## Cross-dataset summary

Median stretch ratio (1.0 = shape preserved), lower is better:

| | Africa | Americas (uniform) | Americas (balanced) | Bunny | GloVe† |
|---|---:|---:|---:|---:|---:|
| Isomap | **1.10** | **3.65** | 6.71 | **1.55** | 1.36 |
| Spectral Embedding | 5.58 | 33.7 | 24.9 | 3.42 | 1.32 |
| LLE | 8.21 | 118 | 398 | 2.98 | 1.19 |
| LTSA | 4.07* | 3.75* | **1.18*** | 2.88 | collapsed |

\* LTSA's ratio is nearly the same at every point (spread 1.32–1.63, identical to the
Original panel). Its distortion is one global linear map, so it is not truly a local
distortion (see [africa.md](africa.md)).
† GloVe has many more than 2 intrinsic dimensions, so these numbers describe only the
2D slice each method keeps (see [glove.md](glove.md)).

**Takeaways so far**

1. **Isomap is the most faithful on flat, convex-ish shapes** (Africa: ratio 1.10, area
   spread 1.58, close to the noise floor). It fails when geodesics run through a
   bottleneck (Americas: South America is squashed).
2. **LTSA distorts globally but uniformly.** One linear map, the same everywhere, can be
   undone. Isomap/LLE/Spectral distortions vary from place to place and cannot.
3. **LLE and Spectral Embedding put most of their distortion at the boundary and in thin
   regions.** On the Americas they collapse the continent onto curves, with stretch
   ratios of 100–1000+.
4. **The Bunny cannot be flattened isometrically.** Every method distorts the ears and
   the rim most, and Isomap distorts least.
5. **Two megaman pitfalls** affected the first-pass results; both are fixed in
   `riemannian_metric.py`:
   - The `dense`/`arpack` solvers treat the non-symmetric geometric Laplacian as
     symmetric and return noise for Spectral Embedding, so it now uses `lobpcg`.
   - A kernel radius smaller than the point spacing disconnects the graph.
