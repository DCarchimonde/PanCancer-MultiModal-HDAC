from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected LINCS Level 5 signatures from a GCTX file into a "
            "memory-mapped NumPy cache for fast repeated training."
        )
    )
    parser.add_argument("--gctx", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cache/revision")
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the full cache.",
    )
    return parser.parse_args()


def decode_array(values: np.ndarray) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return output


def find_dataset(handle: h5py.File, candidates: list[str]) -> h5py.Dataset:
    for candidate in candidates:
        if candidate in handle:
            object_ = handle[candidate]
            if isinstance(object_, h5py.Dataset):
                return object_
    raise KeyError(f"Could not find any GCTX dataset among: {candidates}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    if "sig_id" not in metadata.columns:
        raise ValueError("metadata CSV must contain a sig_id column")
    metadata = metadata.drop_duplicates(subset=["sig_id"], keep="first").reset_index(drop=True)
    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()

    requested_ids = metadata["sig_id"].astype(str).tolist()

    with h5py.File(args.gctx, "r") as handle:
        row_ids_ds = find_dataset(
            handle,
            ["/0/META/ROW/id", "0/META/ROW/id"],
        )
        col_ids_ds = find_dataset(
            handle,
            ["/0/META/COL/id", "0/META/COL/id"],
        )
        matrix = find_dataset(
            handle,
            ["/0/DATA/0/matrix", "0/DATA/0/matrix"],
        )

        row_ids = decode_array(row_ids_ds[:])
        col_ids = decode_array(col_ids_ds[:])
        col_index = {identifier: index for index, identifier in enumerate(col_ids)}

        missing = [identifier for identifier in requested_ids if identifier not in col_index]
        if missing:
            preview = ", ".join(missing[:10])
            raise KeyError(
                f"{len(missing)} requested sig_id values were not found in the GCTX. "
                f"First missing IDs: {preview}"
            )

        selected_indices = np.asarray([col_index[x] for x in requested_ids], dtype=np.int64)
        n_rows = len(row_ids)
        n_cols = len(col_ids)

        if matrix.shape == (n_cols, n_rows):
            orientation = "columns_by_rows"
        elif matrix.shape == (n_rows, n_cols):
            orientation = "rows_by_columns"
        else:
            raise ValueError(
                "Unexpected GCTX matrix shape. "
                f"matrix={matrix.shape}, row_ids={n_rows}, col_ids={n_cols}"
            )

        cache_path = args.output_dir / "expression_float32.npy"
        cache = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(requested_ids), n_rows),
        )

        for start in tqdm(
            range(0, len(selected_indices), args.chunk_size),
            desc="Caching LINCS signatures",
            unit="chunk",
        ):
            stop = min(start + args.chunk_size, len(selected_indices))
            original_indices = selected_indices[start:stop]
            sort_order = np.argsort(original_indices)
            sorted_indices = original_indices[sort_order]

            if orientation == "columns_by_rows":
                sorted_block = np.asarray(matrix[sorted_indices, :], dtype=np.float32)
            else:
                sorted_block = np.asarray(matrix[:, sorted_indices], dtype=np.float32).T

            inverse_order = np.argsort(sort_order)
            cache[start:stop] = sorted_block[inverse_order]

        cache.flush()

    metadata = metadata.copy()
    metadata.insert(0, "cache_row", np.arange(len(metadata), dtype=np.int64))
    metadata.to_csv(args.output_dir / "cache_metadata.csv", index=False)
    pd.Series(row_ids, name="gene_id").to_csv(
        args.output_dir / "gene_ids.csv", index=False
    )

    manifest = {
        "gctx": str(args.gctx.resolve()),
        "metadata_csv": str(args.metadata_csv.resolve()),
        "cache_file": str(cache_path.resolve()),
        "shape": [len(requested_ids), len(row_ids)],
        "dtype": "float32",
        "chunk_size": args.chunk_size,
        "matrix_orientation": orientation,
        "smoke_test_limit": args.limit,
    }
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Expression cache completed successfully.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
