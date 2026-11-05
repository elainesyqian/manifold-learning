# Results

## Parameter sweep (k = 8, 16, 32 × three kernel radii)

Made by `scripts/parameter_sweep.py`. Each dataset has a table that maps every
`(k, r)` setting to its figures:

| Dataset | Table | Radii `r` tried |
|---|---|---|
| Africa | [africa/experiments/README.md](africa/experiments/README.md) | 0.15, 0.25, 0.4 |
| Americas (uniform) | [americas/experiments/uniform_5000/README.md](americas/experiments/uniform_5000/README.md) | 0.18, 0.28, 0.45 |
| Americas (balanced) | [americas/experiments/balanced_5000/README.md](americas/experiments/balanced_5000/README.md) | 0.18, 0.28, 0.45 |
| Stanford Bunny | [bunny/experiments/README.md](bunny/experiments/README.md) | 0.045, 0.07, 0.11 |
| GloVe | [glove/experiments/README.md](glove/experiments/README.md) | 0.45, 0.6, 0.9 |

What each parameter controls:

| Parameter | Affects |
|---|---|
| `k` (neighbours) | Isomap, LLE and LTSA: the k-nearest-neighbour graph they are built on |
| `r` (kernel bandwidth) | Spectral Embedding (Gaussian kernel graph, cut off at 3r), and the Laplacian used to estimate the Riemannian metric of **every** embedding |

So within one `k` folder only Spectral Embedding and the ellipses change with `r`.
Across `k` folders with the same `r`, only Isomap, LLE and LTSA change.

### What the sweep shows

Numbers are median stretch ratios (1 = no local distortion), taken from the tables.

- **Isomap is stable but needs a connected graph.** On Africa its ratio is 1.10–1.22
  for every setting. On the uniform Americas, **k = 8 splits the k-NN graph into
  two pieces** (North/South America, 2,684 / 2,316 points) and Isomap fails. With
  k = 16 → 32 the ratio drops from about 4 to about 3.
- **LTSA is very sensitive to k.** Africa: 7.0 / 2.1 / 5.9 for k = 8 / 16 / 32.
  Americas (balanced): collapsed at k = 8, 20 at k = 16, 4.7 at k = 32. On GloVe it
  collapses for every k.
- **LLE has a sweet spot.** Bunny is best at k = 16 (2.2–3.0, compared with 3.9–5.3 at
  k = 8 or 32). On the uniform Americas, k = 32 is about 10× less distorted than k = 16.
- **Spectral Embedding improves with a larger r on elongated shapes** (uniform
  Americas 45 → 34 → 25, balanced 35 → 25 → 17 as r grows). On GloVe it **collapses at r = 0.45**, where the
  kernel graph is too sparse in 50D.
- **The metric estimate itself depends on r.** A larger r smooths the Laplacian and
  lowers measured ratios a little for every algorithm, Original included. Compare
  algorithms at the same r.

Layout:

```text
<dataset>/experiments/[<profile>/]
  README.md              (k, r) -> figure table, with median stretch ratios
  parameters.csv         the same table as CSV
  k8/  k16/  k32/
    r<r>/
      comparison.png             all embeddings
      metrics/ellipses.png       dual Riemannian metric at 80 sampled points
      metrics/distortion_map.png stretch ratio and area scale at every point
      summary.csv                distortion statistics per algorithm
```

## Other results

| Folder | Contents |
|---|---|
| `africa/*.png`, `bunny/*.png` | First-pass comparisons from `run_africa_benchmark.py` / `run_bunny_benchmark.py` (k = 20) |
| `<dataset>/riemannian/` | Detailed single-setting analysis (k = 20, default r) with numbered point tables; written up in [docs/results/](../docs/results/README.md) |
