from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add exact observed-gene masks to an existing screening bundle so that "
            "the original Spearman ranking can be reproduced without treating "
            "unmatched genes as observed zeroes."
        )
    )
    parser.add_argument("--disease-dir", type=Path, required=True)
    parser.add_argument("--gene-ids", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/screening_input"),
    )
    parser.add_argument("--zip-output", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gene_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if frame.empty:
        raise ValueError(f"Gene-ID file is empty: {path}")
    preferred = ["gene_id", "id", "rid", "entrez_id"]
    column = next((name for name in preferred if name in frame.columns), frame.columns[0])
    values = frame[column].astype(str).str.strip().tolist()
    if len(values) != len(set(values)):
        raise ValueError("Gene IDs must be unique and preserve model-output order")
    return values


def write_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "disease_manifest.csv"
    matrix_path = args.output_dir / "disease_matrix.npy"
    bundle_manifest_path = args.output_dir / "screening_bundle_manifest.json"

    for required in [manifest_path, matrix_path, bundle_manifest_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing existing screening-bundle file: {required}")

    gene_ids = load_gene_ids(args.gene_ids)
    gene_index = pd.Index(gene_ids, dtype=str)
    disease_manifest = pd.read_csv(manifest_path)
    cancer_names = disease_manifest["cancer"].astype(str).tolist()
    disease_matrix = np.load(matrix_path)

    if disease_matrix.shape != (len(cancer_names), len(gene_ids)):
        raise ValueError(
            f"Disease matrix shape {disease_matrix.shape} does not match "
            f"({len(cancer_names)}, {len(gene_ids)})"
        )

    masks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for cancer in cancer_names:
        path = args.disease_dir / f"signature_{cancer}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing disease signature: {path}")
        frame = pd.read_csv(path, index_col=0)
        if frame.empty:
            raise ValueError(f"Disease signature is empty: {path}")
        value_column = "logFC" if "logFC" in frame.columns else None
        if value_column is None:
            numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError(f"No numeric signature column found in {path}")
            value_column = numeric[0]
        series = pd.to_numeric(frame[value_column], errors="coerce")
        series.index = series.index.astype(str)
        series = series.groupby(level=0).mean()
        aligned = series.reindex(gene_index)

        # The original code intersected gene identifiers first and only then filled
        # missing numeric values with zero. Therefore a source-present gene remains
        # observed even when its logFC value is NaN.
        mask = gene_index.isin(series.index)
        masks.append(mask)
        observed_values = aligned.iloc[np.flatnonzero(mask)].fillna(0.0)
        rows.append({
            "cancer": cancer,
            "source_file": str(path),
            "observed_model_genes": int(mask.sum()),
            "observed_fraction": float(mask.mean()),
            "source_present_nan_filled_zero": int(
                aligned.iloc[np.flatnonzero(mask)].isna().sum()
            ),
            "observed_zero_values_after_fill": int((observed_values == 0).sum()),
        })

    mask_matrix = np.stack(masks)
    mask_path = args.output_dir / "disease_observed_mask.npy"
    np.save(mask_path, mask_matrix)
    mask_manifest = pd.DataFrame(rows)
    mask_manifest.to_csv(
        args.output_dir / "disease_observed_mask_manifest.csv",
        index=False,
    )

    unique_masks = len({row.tobytes() for row in mask_matrix})
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    bundle_manifest.update({
        "spearman_observed_mask_shape": list(mask_matrix.shape),
        "spearman_unique_gene_masks": unique_masks,
        "spearman_mask_sha256": sha256_file(mask_path),
        "spearman_note": (
            "Spearman sensitivity uses the exact source-present gene intersection for "
            "each disease signature and fills source-present NaN values with zero, "
            "matching the original pandas corrwith alignment logic."
        ),
    })
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2),
        encoding="utf-8",
    )

    zip_path = args.zip_output or args.output_dir.with_suffix(".zip")
    write_zip(args.output_dir, zip_path)

    print(mask_manifest.to_string(index=False))
    print(json.dumps({
        "mask_shape": list(mask_matrix.shape),
        "unique_gene_masks": unique_masks,
        "mask_sha256": sha256_file(mask_path),
        "zip_output": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path),
    }, indent=2))


if __name__ == "__main__":
    main()
