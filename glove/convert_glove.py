"""Convert GloVe text to aligned, memory-mappable NumPy feature matrices."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "glove"
DEFAULT_OUTPUT = DATA_DIR / "glove.6B.50d"


def convert(source: Path, output: Path, dimensions: int = 50) -> dict:
    """Preserve every source row; publish the output only after validation."""
    source, output = source.resolve(), output.resolve()
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose another --output")
    with source.open("rb") as stream:
        rows = sum(1 for _ in stream)
    if not rows:
        raise ValueError("The input file is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".glove-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        X = np.lib.format.open_memmap(
            staging / "X.npy", mode="w+", dtype=np.float32, shape=(rows, dimensions)
        )
        digest = hashlib.sha256()
        with source.open("rb") as stream, (staging / "tokens.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as tokens:
            parsed_rows = 0
            for index, line in enumerate(stream):
                digest.update(line)
                fields = line.decode("utf-8").split()
                if len(fields) != dimensions + 1:
                    raise ValueError(
                        f"Line {index + 1}: expected a token and {dimensions} values, "
                        f"got {len(fields)} fields"
                    )
                if index >= rows:
                    raise ValueError("Input row count changed during conversion")
                try:
                    X[index] = np.asarray(fields[1:], dtype=np.float32)
                except ValueError as error:
                    raise ValueError(f"Line {index + 1}: invalid numeric value") from error
                tokens.write(fields[0] + "\n")
                parsed_rows += 1
        if parsed_rows != rows:
            raise ValueError("Input row count changed during conversion")

        unit = np.lib.format.open_memmap(
            staging / "X_unit.npy", mode="w+", dtype=np.float32, shape=X.shape
        )
        for start in range(0, rows, 10000):
            block = X[start : start + 10000]
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite vector in rows {start + 1}–{start + len(block)}")
            norms = np.linalg.norm(block.astype(np.float64), axis=1)
            if np.any(norms == 0):
                raise ValueError("Cannot unit-normalize a zero-length vector")
            unit[start : start + len(block)] = block / norms[:, None]
        X.flush()
        unit.flush()
        del X, unit

        metadata = {
            "source_file": source.name,
            "source_sha256": digest.hexdigest(),
            "rows": rows,
            "dimensions": dimensions,
            "dtype": "float32",
            "row_order": "unchanged from source; tokens.txt line i+1 labels matrix row i",
            "files": {
                "X.npy": "Original vectors rounded to float32; no centering or scaling",
                "X_unit.npy": "Each original vector divided by its Euclidean (L2) norm",
                "tokens.txt": "UTF-8 tokens, one per line, including punctuation and numbers",
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "glove.6B.50d.txt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimensions", type=int, default=50)
    args = parser.parse_args()
    metadata = convert(args.input, args.output, args.dimensions)
    print(f"Saved {metadata['rows']:,} × {metadata['dimensions']} float32 matrices to {args.output}")
    print(f"Source SHA-256: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()
