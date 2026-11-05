"""Sweep the neighbourhood size k and kernel radius r for every dataset.

For each dataset, each k in K_VALUES and each radius r in that dataset's RADII:

  * Isomap, LLE and LTSA use a k-nearest-neighbour graph (depend on k only);
  * Spectral Embedding uses a Gaussian kernel of bandwidth r (depends on r only);
  * the dual Riemannian metric of every embedding is estimated with the same
    r-bandwidth Laplacian (depends on r).

Output, following the layout of the shared figure collection:

    results/<dataset>/experiments/[<profile>/]k<k>/r<r>/
        comparison.png             all embeddings, coloured by ground truth
        metrics/ellipses.png       dual-metric ellipses at 80 sampled points
        metrics/distortion_map.png stretch ratio and area scale at every point
        summary.csv                distortion statistics per algorithm
    results/<dataset>/experiments/[<profile>/]README.md
                                   table: (k, r) -> figure files + key numbers
    results/<dataset>/experiments/[<profile>/]parameters.csv
                                   the same table, machine-readable

    python scripts/parameter_sweep.py                 # every dataset
    python scripts/parameter_sweep.py africa bunny    # a subset
"""

import argparse
import csv
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

import riemannian_metric as rm

K_VALUES = (8, 16, 32)
# Kernel bandwidths: about 0.6x, 1x and 1.6x the default used in riemannian_metric.py
RADII = {
    "africa": (0.15, 0.25, 0.4),
    "americas_uniform": (0.18, 0.28, 0.45),
    "americas_balanced": (0.18, 0.28, 0.45),
    "bunny": (0.045, 0.07, 0.11),
    "glove": (0.45, 0.6, 0.9),
}
OUT = {
    "africa": "africa/experiments",
    "americas_uniform": "americas/experiments/uniform_5000",
    "americas_balanced": "americas/experiments/balanced_5000",
    "bunny": "bunny/experiments",
    "glove": "glove/experiments",
}
ALGORITHMS = ("Isomap", "Spectral Embedding", "LLE", "LTSA")
N_SAMPLES = 80

plt.rcParams["savefig.dpi"] = 110   # many figures: keep the repo small


def summarise(evals):
    ratio = np.sqrt(evals[:, 1] / evals[:, 0])
    area = np.sqrt(evals[:, 0] * evals[:, 1])
    return dict(
        median_stretch_ratio=float(np.median(ratio)),
        p90_stretch_ratio=float(np.percentile(ratio, 90)),
        frac_ratio_over_2=float(np.mean(ratio > 2)),
        area_scale_spread_p90_p10=float(np.percentile(area, 90) / np.percentile(area, 10)),
    )


def knn_pieces(X, k):
    """Sizes of the connected pieces of the (symmetrised) k-NN graph."""
    geom = rm.knn_geometry(k)
    geom.set_data_matrix(X)
    A = geom.compute_adjacency_matrix()
    _, labels = connected_components(A.maximum(A.T))
    return sorted(np.bincount(labels), reverse=True)


def collapsed(Y, frac=0.02):
    """True when 98% of points sit in a tiny fraction of the embedding's range."""
    core = np.ptp(np.percentile(Y, [1, 99], axis=0), axis=0).max()
    return core < frac * np.ptp(Y, axis=0).max()


def try_fit(name, model, X, failures):
    try:
        return rm.fit_embedding(name, model, X)
    except Exception as error:                      # record and keep going
        failures[name] = f"{type(error).__name__}: {str(error).splitlines()[0][:80]}"
        print(f"  {name} FAILED: {failures[name]}")
        traceback.print_exc(limit=1)
        return None


def run(key):
    cfg = rm.DATASETS[key]
    print(f"\n== {cfg['title']}")
    X, colour, colour_label, _ = cfg["load"]()
    out_root = rm.ROOT / "results" / OUT[key]
    idx = rm.farthest_point_sample(X, N_SAMPLES)
    rows = []

    for k in K_VALUES:
        print(f"-- k = {k}")
        knn_failures = {}
        pieces = knn_pieces(X, k)
        if len(pieces) > 1:
            knn_failures["k-NN graph"] = (f"splits into {len(pieces)} pieces "
                                          f"({' / '.join(map(str, pieces))} pts)")
        knn_embeddings = {name: try_fit(name, model, X, knn_failures)
                          for name, model in rm.knn_models(k).items()}

        for r in RADII[key]:
            print(f"   r = {r}")
            t = time.time()
            folder = out_root / f"k{k}" / f"r{r:g}"
            (folder / "metrics").mkdir(parents=True, exist_ok=True)
            failures = dict(knn_failures)

            geom = rm.radius_geometry(r)
            geom.set_data_matrix(X)
            n_comp = connected_components(geom.compute_adjacency_matrix())[0]
            laplacian = geom.compute_laplacian_matrix()

            spectral = try_fit("Spectral Embedding", rm.spectral_model(r), X, failures)
            embeddings = {
                "Isomap": knn_embeddings["Isomap"], "Spectral Embedding": spectral,
                "LLE": knn_embeddings["LLE"], "LTSA": knn_embeddings["LTSA"],
            }
            embeddings = {n: Y for n, Y in embeddings.items() if Y is not None}
            for name, Y in embeddings.items():
                if collapsed(Y):
                    failures[name] = "collapsed (98% of points on ~one spot)"
            spaces = {"Original": X} | embeddings
            metric_spaces = spaces if cfg["map_2d"] else embeddings
            metrics = {}
            for name, Y in metric_spaces.items():
                try:
                    metrics[name] = rm.dual_metric(Y, laplacian)
                except Exception as error:
                    failures[f"{name} metric"] = type(error).__name__

            title = f"{cfg['title']} · k = {k}, r = {r:g}"
            rm.plot_embeddings(spaces, colour, colour_label, title, folder / "comparison.png")
            rm.plot_ellipse_grid(spaces, metrics, colour, idx, title,
                                 folder / "metrics" / "ellipses.png")
            rm.plot_distortion_map(X, spaces,
                                   {n: m for n, m in metrics.items() if n != "Original"},
                                   cfg["map_2d"], title, folder / "metrics" / "distortion_map.png")

            stats = {name: summarise(m[1]) for name, m in metrics.items()}
            with open(folder / "summary.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["algorithm", *next(iter(stats.values()))])
                for name, s in stats.items():
                    w.writerow([name, *(f"{v:.4g}" for v in s.values())])

            rows.append(dict(
                k=k, r=r, adjacency_cutoff=3 * r, knn_graph_pieces=len(pieces),
                kernel_graph_components=n_comp,
                folder=folder.relative_to(out_root).as_posix(),
                **{f"median_ratio_{n}": stats[n]["median_stretch_ratio"] if n in stats else None
                   for n in ALGORITHMS},
                failures="; ".join(f"{n}: {e}" for n, e in failures.items()),
            ))
            print(f"   -> {folder.relative_to(rm.ROOT)} ({time.time() - t:.0f}s)")

    write_index(cfg, out_root, rows)


def write_index(cfg, out_root, rows):
    with open(out_root / "parameters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: "–" if v is None else f"{v:.3g}"
    lines = [
        f"# {cfg['title']}: parameter sweep",
        "",
        "Generated by `scripts/parameter_sweep.py`. Each row is one setting and the",
        "figures it produced. `k` = neighbours in the k-NN graph (Isomap, LLE, LTSA);",
        "`r` = Gaussian kernel bandwidth for Spectral Embedding and for the Riemannian",
        "metric of every embedding (graph cut off at 3r). The four algorithm columns are",
        "the median stretch ratio √(λ_long/λ_short) of each embedding (1 = no local",
        "distortion; – = algorithm failed). Notes explain failures and collapsed embeddings.",
        "",
        "| k | r | 3r cutoff | comparison | ellipses | distortion map | stats "
        "| Isomap | Spectral | LLE | LTSA | notes |",
        "|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row["folder"]
        notes = row["failures"]
        if row["kernel_graph_components"] > 1:
            notes = f"kernel graph has {row['kernel_graph_components']} components; " + notes
        lines.append(
            f"| {row['k']} | {row['r']:g} | {row['adjacency_cutoff']:g} "
            f"| [comparison]({d}/comparison.png) | [ellipses]({d}/metrics/ellipses.png) "
            f"| [distortion]({d}/metrics/distortion_map.png) | [csv]({d}/summary.csv) "
            + "".join(f"| {fmt(row[f'median_ratio_{n}'])} " for n in ALGORITHMS)
            + f"| {notes or ''} |"
        )
    lines += ["", "Machine-readable version: [`parameters.csv`](parameters.csv).", ""]
    (out_root / "README.md").write_text("\n".join(lines))
    print(f"   index -> {(out_root / 'README.md').relative_to(rm.ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="*", metavar="dataset",
                        help=f"any of: {', '.join(RADII)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(RADII)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for key in args.datasets or RADII:
        run(key)


if __name__ == "__main__":
    main()
