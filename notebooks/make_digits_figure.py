"""Slide figure from digits.ipynb: 64-dimensional digit images and three 2D embeddings.

Same data and settings as the notebook (Isomap k=15, t-SNE and UMAP with
random_state=42), laid out as one figure for a presentation slide.

    python notebooks/make_digits_figure.py
    -> notebooks/figures/digits_overview.png (and .pdf)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn import datasets, manifold

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

digits = datasets.load_digits()
X, y = digits.data, digits.target  # (1797, 64), labels 0-9

embeddings = {
    "Isomap": manifold.Isomap(n_components=2, n_neighbors=15),
    "t-SNE": manifold.TSNE(n_components=2, random_state=42),
    "UMAP": umap.UMAP(n_components=2, random_state=42),
}
for name, model in embeddings.items():
    print(f"fitting {name}...")
    embeddings[name] = model.fit_transform(X)

# --- figure -----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans"})
colors = plt.get_cmap("tab10")
fig = plt.figure(figsize=(12, 10), facecolor="white")
top, bottom = fig.subfigures(2, 1, height_ratios=[4.6, 5.0], hspace=0.02)

# Top: one example of each digit; its frame colour doubles as the legend below
top.suptitle("Each image is 8 × 8 = 64 pixels → a point in 64-dimensional space",
             fontsize=15, fontweight="bold", y=0.97)
first_of_each = [int(np.flatnonzero(y == d)[0]) for d in range(10)]
digit_axes = top.subplots(2, 5, gridspec_kw={"wspace": 0.12, "hspace": 0.12}).ravel()
for d, (ax, i) in enumerate(zip(digit_axes, first_of_each)):
    ax.imshow(digits.images[i], cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors(d))
        spine.set_linewidth(3)
top.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.02)

# Bottom: the embeddings
bottom.suptitle(f"↓  the same {len(X):,} images squeezed into 2D by three algorithms (colour = digit)  ↓",
                fontsize=13, color="#444444", y=0.99)
for ax, (name, Y) in zip(bottom.subplots(1, 3, gridspec_kw={"wspace": 0.08}),
                         embeddings.items()):
    ax.scatter(Y[:, 0], Y[:, 1], c=[colors(v) for v in y], s=9, alpha=0.75, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.set_title(name, fontsize=15, fontweight="bold", pad=8)
bottom.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"digits_overview.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
print(f"saved {OUT / 'digits_overview.png'}")
