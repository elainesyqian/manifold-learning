"""Load the converted matrix and its aligned token labels."""

from pathlib import Path

import numpy as np


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "glove" / "glove.6B.50d"


def load_glove(data_dir=DEFAULT_DATA, *, normalized=True, mmap_mode="r"):
    """Return (X, tokens), with tokens[i] labeling X[i].

    X is read-only and memory-mapped by default. Set mmap_mode=None to load
    into memory, or call X.copy() if an estimator needs a writable matrix.
    """
    data_dir = Path(data_dir)
    filename = "X_unit.npy" if normalized else "X.npy"
    X = np.load(data_dir / filename, mmap_mode=mmap_mode, allow_pickle=False)
    tokens = (data_dir / "tokens.txt").read_text(encoding="utf-8").splitlines()
    if X.ndim != 2 or X.shape[0] != len(tokens):
        raise ValueError("Feature matrix and token labels are not aligned")
    return X, tokens
