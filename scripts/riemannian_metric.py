"""Estimate and plot the dual Riemannian metric of manifold-learning embeddings.

For each dataset this script:

1. embeds the data in 2D with Isomap, Spectral Embedding, LLE and LTSA (megaman);
2. estimates the dual Riemannian metric H at every point (Perrault-Joncas &
   Meila, 2013) from the embedding and a geometric Laplacian of the input data;
3. draws H as an ellipse at an evenly spread sample of points. The ellipse axes
   point along the eigenvectors of H and have length sqrt(eigenvalue), i.e. how
   much a unit step on the data manifold is stretched (long axis) or shrunk
   (short axis) by the embedding at that point. A circle means no distortion.

Outputs go to results/<dataset>/riemannian/:

    embeddings.png        all embeddings, coloured by a ground-truth variable
    ellipses.png          dual-metric ellipses at sampled points, per algorithm
    distortion_map.png    anisotropy and area scale at every point
    detail_<algo>.png     12 numbered points with their eigenvalues/eigenvectors
    samples.csv           metric values at every sampled point
    summary.csv           per-algorithm distortion statistics
    metric.npz            embeddings Y and dual metrics H for all points

Run from anywhere (all datasets, or a subset):

    python scripts/riemannian_metric.py
    python scripts/riemannian_metric.py africa bunny
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

# --- LEGACY NUMPY & SCIPY PATCHES (megaman predates both) ---
if not hasattr(np, "float"):
    np.float = float

_original_solve = scipy.linalg.solve
def _patched_solve(a, b, **kwargs):
    if "sym_pos" in kwargs:
        kwargs.pop("sym_pos")
        kwargs["assume_a"] = "pos"
    return _original_solve(a, b, **kwargs)
scipy.linalg.solve = _patched_solve

_original_eigh = scipy.linalg.eigh
def _patched_eigh(a, b=None, **kwargs):
    if "eigvals" in kwargs:
        kwargs["subset_by_index"] = kwargs.pop("eigvals")
    return _original_eigh(a, b, **kwargs)
scipy.linalg.eigh = _patched_eigh
# ------------------------------------------------------------

from megaman.embedding import LTSA, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from megaman.geometry import Geometry
from megaman.geometry.rmetric import RiemannMetric
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

N_NEIGHBORS = 20      # k-NN graph for Isomap, LLE, LTSA
N_DETAIL = 12         # numbered points in the detail figures
ANISO_MAX = 1.0       # colour scale cap: log10(long/short axis) = 1 means 10x


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

def _continent(folder, title, radius, out):
    def load():
        X = np.load(DATA / folder / "countries_X.npy")
        ids = np.load(DATA / folder / "countries_country_id.npy")
        names = json.loads((DATA / folder / "countries_country_names.json").read_text())
        labels = [names[str(i)] for i in ids]
        return X, X[:, 1], "north–south position", labels
    return dict(title=title, radius=radius, load=load, out=out, map_2d=True)


def _bunny():
    X = np.load(DATA / "bunny" / "X.npy")
    return X, np.load(DATA / "bunny" / "height.npy"), "height", None


def _glove(n=5000):
    from glove.load_glove import load_glove

    X_all, tokens = load_glove(normalized=True)
    # GloVe rows are sorted by corpus frequency: keep the n most frequent words.
    rows = [i for i, t in enumerate(tokens) if t.isalpha()][:n]
    X = np.asarray(X_all[rows], dtype=np.float64)
    return X, np.log10(np.arange(1, n + 1)), "log10 frequency rank", [tokens[i] for i in rows]


DATASETS = {
    "africa": _continent("africa/uniform_5000", "Africa (uniform, 5,000 pts)", 0.25,
                         "africa"),
    "americas_uniform": _continent("americas/uniform_5000", "Americas (uniform, 5,000 pts)",
                                   0.28, "americas/uniform_5000"),
    "americas_balanced": _continent("americas/balanced_5000",
                                    "Americas (balanced, 5,000 pts)", 0.28,
                                    "americas/balanced_5000"),
    "bunny": dict(title="Stanford Bunny (5,000 surface pts)", radius=0.07, load=_bunny,
                  out="bunny", map_2d=False),
    "glove": dict(title="GloVe 50d (5,000 most frequent words)", radius=0.6, load=_glove,
                  out="glove", map_2d=False),
}


# ----------------------------------------------------------------------------
# Embedding + metric
# ----------------------------------------------------------------------------

def radius_geometry(radius):
    # Gaussian kernel of bandwidth `radius`, truncated at 3 bandwidths;
    # renormalised geometric Laplacian, so H is in the units of the embedding.
    return Geometry(
        adjacency_method="brute",
        adjacency_kwds={"radius": 3 * radius},
        affinity_method="gaussian",
        affinity_kwds={"radius": radius},
        laplacian_method="geometric",
        laplacian_kwds={"scaling_epps": radius},
    )


def knn_geometry():
    return Geometry(adjacency_method="brute", adjacency_kwds={"n_neighbors": N_NEIGHBORS})


def embed_all(X, radius):
    models = {
        "Isomap": Isomap(n_components=2, geom=knn_geometry(), eigen_solver="dense"),
        # lobpcg works on the symmetrised Laplacian; megaman's dense/arpack paths
        # treat the (non-symmetric) geometric Laplacian as symmetric and return noise.
        "Spectral Embedding": SpectralEmbedding(n_components=2, geom=radius_geometry(radius),
                                                eigen_solver="lobpcg", random_state=0,
                                                solver_kwds={"maxiter": 500}),
        "LLE": LocallyLinearEmbedding(n_components=2, geom=knn_geometry(),
                                      eigen_solver="arpack"),
        "LTSA": LTSA(n_components=2, geom=knn_geometry(), eigen_solver="arpack"),
    }
    embeddings = {}
    for name, model in models.items():
        t = time.time()
        try:
            Y = model.fit_transform(X)
        except ValueError:
            # ARPACK can fail on a singular weight matrix (e.g. LTSA on GloVe)
            print(f"  {name}: ARPACK failed, retrying with the dense solver")
            model.eigen_solver = "dense"
            Y = model.fit_transform(X)
        embeddings[name] = np.asarray(Y, dtype=np.float64)
        print(f"  {name:<20s} {time.time() - t:5.1f}s")
    return embeddings


def dual_metric(Y, laplacian):
    H = RiemannMetric(Y, laplacian, n_dim=2).get_dual_rmetric()
    H = 0.5 * (H + np.transpose(H, (0, 2, 1)))           # symmetrise
    evals, evecs = np.linalg.eigh(H)                       # ascending
    evals = np.clip(evals, 1e-12, None)
    return H, evals, evecs


def farthest_point_sample(X, n, seed=0):
    """Evenly spread sample: each new point is the one farthest from those chosen."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(X)))]
    dist = np.linalg.norm(X - X[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(X - X[chosen[-1]], axis=1))
    return np.array(chosen)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#cccccc",
})
ANISO_CMAP = plt.get_cmap("plasma")
POINT_CMAP = plt.get_cmap("viridis")


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_anchor("C")


def robust_view(ax, Y, q=1.0):
    """Zoom to the central (100 - 2q)% of points so a few far outliers don't
    shrink the rest of the embedding to a dot. Returns the view width."""
    lo, hi = np.percentile(Y, q, axis=0), np.percentile(Y, 100 - q, axis=0)
    pad = 0.08 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    outside = np.any((Y < lo - pad) | (Y > hi + pad), axis=1).sum()
    if outside:
        ax.text(0.98, 0.98, f"{outside} outlying pts off view", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#888888")
    return (hi - lo).max() + 2 * pad


def ellipse_scale(Y, width, evals, idx, frac=0.4):
    """Factor so the median long semi-axis is `frac` of the median spacing
    between neighbouring sampled points. The spacing is floored at a fraction
    of the view so ellipses stay visible when many samples pile up together."""
    P = Y[idx]
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    spacing = max(np.median(d.min(axis=1)), 0.3 * width / np.sqrt(len(idx)))
    return frac * spacing / np.median(np.sqrt(evals[idx, 1]))


def collapse_warning(ax, Y, width):
    full = np.ptp(Y, axis=0).max()
    if width < 1e-4 * full:
        ax.text(0.5, 0.5, f"embedding collapsed:\n98% of points lie within {width:.1g}\n"
                          f"of each other (full range {full:.2g})",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color="#b00020", fontweight="bold",
                bbox=dict(fc="white", ec="#b00020", alpha=0.85))


def draw_ellipses(ax, Y, evals, evecs, idx, scale, lw=1.0, axes_lines=True, cap=5.0):
    """Draw H at the points idx. Ellipses longer than `cap` x the median are
    shrunk to that length (keeping their shape) and drawn dashed."""
    aniso = 0.5 * np.log10(evals[:, 1] / evals[:, 0])
    norm = Normalize(0, ANISO_MAX)
    max_long = cap * scale * np.median(np.sqrt(evals[idx, 1]))
    for i in idx:
        a_long, a_short = scale * np.sqrt(evals[i, 1]), scale * np.sqrt(evals[i, 0])
        clipped = a_long > max_long
        if clipped:
            a_short *= max_long / a_long
            a_long = max_long
        v = evecs[i][:, 1]
        angle = np.degrees(np.arctan2(v[1], v[0]))
        color = ANISO_CMAP(norm(aniso[i]))
        ax.add_patch(Ellipse(Y[i], 2 * a_long, 2 * a_short, angle=angle,
                             facecolor=(*color[:3], 0.35), edgecolor=color, lw=lw,
                             linestyle="--" if clipped else "-", zorder=3))
        if axes_lines:
            ax.plot(*np.c_[Y[i] - a_long * v, Y[i] + a_long * v], color=color, lw=0.6, zorder=4)
    return aniso


def plot_embeddings(spaces, colour, colour_label, title, path):
    n = len(spaces)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    for ax, (name, Y) in zip(axes, spaces.items()):
        if Y.shape[1] > 3:
            Yc = Y - Y.mean(axis=0)
            Y = Yc @ np.linalg.svd(Yc, full_matrices=False)[2][:2].T
            name = f"{name} (PCA of {Yc.shape[1]}d)"
        if Y.shape[1] == 3:
            ax.remove()
            ax = fig.add_subplot(1, n, list(spaces).index("Original") + 1, projection="3d")
            ax.scatter(*Y.T, c=colour, cmap=POINT_CMAP, s=2)
            ax.view_init(elev=15, azim=-90)
            ax.set_axis_off()
        else:
            ax.scatter(*Y.T, c=colour, cmap=POINT_CMAP, s=2, linewidths=0)
            clean(ax)
        ax.set_title(name)
    fig.suptitle(f"{title} — coloured by {colour_label}", fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ellipse_grid(spaces, metrics, colour, idx, title, path):
    names = [n for n in spaces if n in metrics]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.0))
    for ax, name in zip(np.atleast_1d(axes), names):
        Y = spaces[name]
        _, evals, evecs = metrics[name]
        ax.scatter(*Y.T, c=colour, cmap=POINT_CMAP, s=2, alpha=0.25, linewidths=0)
        clean(ax)
        width = robust_view(ax, Y)
        scale = ellipse_scale(Y, width, evals, idx)
        draw_ellipses(ax, Y, evals, evecs, idx, scale)
        collapse_warning(ax, Y, width)
        ax.set_title(name)
        med = np.median(np.sqrt(evals[idx]), axis=0)
        ax.text(0.02, 0.02, f"median √λ: long {med[1]:.3g}, short {med[0]:.3g}\n"
                            f"ellipses drawn ×{scale:.3g} (dashed = capped at 5× median)",
                transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    sm = plt.cm.ScalarMappable(cmap=ANISO_CMAP, norm=Normalize(0, ANISO_MAX))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("anisotropy  log10(long / short axis)")
    fig.suptitle(f"{title} — dual Riemannian metric at {len(idx)} sampled points",
                 fontsize=15)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distortion_map(X, spaces, metrics, map_2d, title, path):
    """Anisotropy and area scale at every point, drawn on the input map (2D data)
    or on each embedding (otherwise)."""
    names = list(metrics)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8.4))
    for col, name in enumerate(names):
        _, evals, _ = metrics[name]
        aniso = 0.5 * np.log10(evals[:, 1] / evals[:, 0])
        area = 0.5 * np.log10(evals[:, 0] * evals[:, 1])
        area -= np.median(area)
        base = X if map_2d else spaces[name]
        for row, (vals, cmap, norm, lab) in enumerate([
            (aniso, ANISO_CMAP, Normalize(0, ANISO_MAX), "anisotropy log10(long/short)"),
            (area, plt.get_cmap("RdBu_r"), Normalize(-1, 1), "log10 area scale vs median"),
        ]):
            ax = axes[row, col]
            sc = ax.scatter(*base[:, :2].T, c=vals, cmap=cmap, norm=norm, s=3, linewidths=0)
            clean(ax)
            if row == 0:
                ax.set_title(name)
            if col == len(names) - 1:
                fig.colorbar(sc, ax=axes[row, :], fraction=0.015, pad=0.01).set_label(lab)
    where = "on the original map" if map_2d else "on each embedding"
    fig.suptitle(f"{title} — local distortion at every point, {where}", fontsize=15)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_detail(name, Y, evals, evecs, colour, idx, labels, title, path):
    fig = plt.figure(figsize=(15, 6.6))
    ax = fig.add_axes([0.02, 0.05, 0.44, 0.85])
    ax.scatter(*Y.T, c=colour, cmap=POINT_CMAP, s=3, alpha=0.3, linewidths=0)
    clean(ax)
    width = robust_view(ax, Y)
    scale = ellipse_scale(Y, width, evals, idx, frac=0.4)
    draw_ellipses(ax, Y, evals, evecs, idx, scale, lw=1.6)
    collapse_warning(ax, Y, width)
    for k, i in enumerate(idx, 1):
        tag = f"{k}" if labels is None else f"{k} {labels[i]}"
        ax.annotate(tag, Y[i], xytext=(5, 5), textcoords="offset points", fontsize=9,
                    fontweight="bold", color="#222222", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_title(f"{name}: {len(idx)} numbered points (ellipses ×{scale:.3g})")

    rows = []
    for k, i in enumerate(idx, 1):
        v_long = evecs[i][:, 1]
        rows.append([
            str(k), *([] if labels is None else [str(labels[i])[:14]]),
            f"{evals[i, 1]:.3g}", f"{evals[i, 0]:.3g}",
            f"({v_long[0]:+.2f}, {v_long[1]:+.2f})",
            f"{np.sqrt(evals[i, 1] / evals[i, 0]):.2f}",
        ])
    tax = fig.add_axes([0.5, 0.05, 0.48, 0.85])
    tax.axis("off")
    table = tax.table(
        cellText=rows,
        colLabels=["#", *([] if labels is None else ["label"]), "λ long", "λ short",
                   "v long (x, y)", "√(λl/λs)"],
        loc="center", cellLoc="center", colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    aniso = 0.5 * np.log10(evals[idx, 1] / evals[idx, 0])
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#333333")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif c == len(rows[0]) - 1:
            color = ANISO_CMAP(Normalize(0, ANISO_MAX)(aniso[r - 1]))
            cell.set_facecolor((*color[:3], 0.45))
    tax.set_title("Eigenvalues λ of the dual metric H and the long-axis direction",
                  fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=15)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(key, n_samples):
    cfg = DATASETS[key]
    print(f"\n== {cfg['title']}")
    X, colour, colour_label, labels = cfg["load"]()
    out = ROOT / "results" / cfg["out"] / "riemannian"
    out.mkdir(parents=True, exist_ok=True)

    geom = radius_geometry(cfg["radius"])
    geom.set_data_matrix(X)
    n_comp = connected_components(geom.compute_adjacency_matrix())[0]
    if n_comp > 1:
        raise RuntimeError(f"radius {cfg['radius']} leaves {n_comp} graph components")
    laplacian = geom.compute_laplacian_matrix()

    embeddings = embed_all(X, cfg["radius"])
    spaces = ({"Original": X} | embeddings)
    metric_spaces = spaces if cfg["map_2d"] else embeddings
    metrics = {name: dual_metric(Y, laplacian) for name, Y in metric_spaces.items()}

    idx = farthest_point_sample(X, n_samples)
    detail_idx = idx[:N_DETAIL]

    plot_embeddings(spaces, colour, colour_label, cfg["title"], out / "embeddings.png")
    plot_ellipse_grid(spaces, metrics, colour, idx, cfg["title"], out / "ellipses.png")
    plot_distortion_map(X, spaces, {k: v for k, v in metrics.items() if k != "Original"},
                        cfg["map_2d"], cfg["title"], out / "distortion_map.png")
    for name, (_, evals, evecs) in metrics.items():
        slug = name.lower().replace(" ", "_")
        plot_detail(name, spaces[name], evals, evecs, colour, detail_idx, labels,
                    cfg["title"], out / f"detail_{slug}.png")

    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "sample_rank", "point_index", "label", "y1", "y2",
                    "lambda_long", "lambda_short", "v_long_x", "v_long_y",
                    "v_short_x", "v_short_y", "stretch_ratio", "area_scale"])
        for name, (_, evals, evecs) in metrics.items():
            Y = spaces[name]
            for rank, i in enumerate(idx, 1):
                values = [*Y[i, :2], *evals[i, ::-1], *evecs[i][:, 1], *evecs[i][:, 0],
                          np.sqrt(evals[i, 1] / evals[i, 0]), np.sqrt(evals[i, 0] * evals[i, 1])]
                w.writerow([name, rank, i, "" if labels is None else labels[i],
                            *(f"{v:.6g}" for v in values)])

    summary = []
    for name, (_, evals, _) in metrics.items():
        ratio = np.sqrt(evals[:, 1] / evals[:, 0])
        area = np.sqrt(evals[:, 0] * evals[:, 1])
        summary.append(dict(
            algorithm=name,
            median_stretch_ratio=round(float(np.median(ratio)), 3),
            p90_stretch_ratio=round(float(np.percentile(ratio, 90)), 3),
            frac_ratio_over_2=round(float(np.mean(ratio > 2)), 3),
            median_area_scale=float(f"{np.median(area):.4g}"),
            area_scale_spread_p90_p10=round(float(np.percentile(area, 90)
                                                  / np.percentile(area, 10)), 3),
        ))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for row in summary:
        print("  ", row)

    np.savez_compressed(
        out / "metric.npz", sample_index=idx,
        **{f"Y_{k.replace(' ', '_')}": v for k, v in embeddings.items()},
        **{f"H_{k.replace(' ', '_')}": v[0].astype(np.float32) for k, v in metrics.items()},
    )
    print(f"  -> {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="*", metavar="dataset",
                        help=f"any of: {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--n-samples", type=int, default=80,
                        help="points that get an ellipse (default 80)")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for key in args.datasets or DATASETS:
        run(key, args.n_samples)


if __name__ == "__main__":
    main()
