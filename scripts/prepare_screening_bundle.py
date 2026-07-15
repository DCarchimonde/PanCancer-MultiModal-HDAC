from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compact, gene-aligned screening bundle for revision ranking analyses."
    )
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--disease-dir", type=Path, required=True)
    parser.add_argument("--gene-ids", type=Path, required=True)
    parser.add_argument("--compoundinfo", type=Path, default=None)
    parser.add_argument("--name-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/screening_input"))
    parser.add_argument("--zip-output", type=Path, default=None)
    return parser.parse_args()


def canonicalize_smiles(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "restricted"}:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


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


def load_name_cache(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for smiles, name in raw.items():
        canonical = canonicalize_smiles(smiles)
        if canonical and str(name).strip():
            result[canonical] = str(name).strip()
    return result


def load_compoundinfo(path: Path | None) -> pd.DataFrame:
    columns = [
        "canonical_smiles", "compoundinfo_name", "compoundinfo_pert_id",
        "target", "moa", "compound_aliases",
    ]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"canonical_smiles", "cmap_name", "pert_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Compound-info file lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["canonical_smiles"] = frame["canonical_smiles"].map(canonicalize_smiles)
    frame = frame.dropna(subset=["canonical_smiles"])
    keep = [
        "canonical_smiles", "cmap_name", "pert_id", "target", "moa", "compound_aliases"
    ]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame = frame.rename(columns={
        "cmap_name": "compoundinfo_name",
        "pert_id": "compoundinfo_pert_id",
    })
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""

    def join_unique(series: pd.Series) -> str:
        values = sorted({
            str(value).strip() for value in series
            if pd.notna(value) and str(value).strip()
        })
        return " | ".join(values)

    return frame.groupby("canonical_smiles", as_index=False).agg({
        "compoundinfo_name": join_unique,
        "compoundinfo_pert_id": join_unique,
        "target": join_unique,
        "moa": join_unique,
        "compound_aliases": join_unique,
    })


def build_screening_library(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(args.clean_csv, low_memory=False)
    if "smiles" not in source.columns:
        raise ValueError("clean_dataset.csv must contain a smiles column")
    if "drug_id" not in source.columns:
        source["drug_id"] = ""

    source = source[["smiles", "drug_id"]].copy()
    source["canonical_smiles"] = source["smiles"].map(canonicalize_smiles)
    invalid_rows = int(source["canonical_smiles"].isna().sum())
    source = source.dropna(subset=["canonical_smiles"])

    structure_to_originals: defaultdict[str, set[str]] = defaultdict(set)
    structure_to_ids: defaultdict[str, set[str]] = defaultdict(set)
    for row in source.itertuples(index=False):
        structure_to_originals[row.canonical_smiles].add(str(row.smiles))
        if pd.notna(row.drug_id) and str(row.drug_id).strip():
            structure_to_ids[row.canonical_smiles].add(str(row.drug_id).strip())

    library = pd.DataFrame({"canonical_smiles": sorted(structure_to_originals)})
    library["source_smiles"] = library["canonical_smiles"].map(
        lambda key: " | ".join(sorted(structure_to_originals[key]))
    )
    library["source_drug_ids"] = library["canonical_smiles"].map(
        lambda key: " | ".join(sorted(structure_to_ids[key]))
    )

    compoundinfo = load_compoundinfo(args.compoundinfo)
    library = library.merge(compoundinfo, on="canonical_smiles", how="left")
    for column in [
        "compoundinfo_name", "compoundinfo_pert_id", "target", "moa", "compound_aliases"
    ]:
        library[column] = library[column].fillna("").astype(str)

    name_cache = load_name_cache(args.name_cache)
    library["cache_name"] = library["canonical_smiles"].map(name_cache).fillna("")
    library["display_name"] = library["cache_name"]
    empty = library["display_name"].eq("")
    library.loc[empty, "display_name"] = library.loc[empty, "compoundinfo_name"]
    empty = library["display_name"].eq("")
    library.loc[empty, "display_name"] = library.loc[empty, "source_drug_ids"]
    empty = library["display_name"].eq("")
    library.loc[empty, "display_name"] = library.loc[empty, "canonical_smiles"]

    cache_name = library["cache_name"].str.lower().str.strip()
    info_names = library["compoundinfo_name"].str.lower().str.strip()
    library["name_conflict"] = (
        cache_name.ne("")
        & info_names.ne("")
        & ~library.apply(
            lambda row: row["cache_name"].lower().strip()
            in {part.lower().strip() for part in row["compoundinfo_name"].split("|")},
            axis=1,
        )
    )
    annotation = (
        library["display_name"] + " " + library["compoundinfo_name"] + " "
        + library["target"] + " " + library["moa"] + " " + library["compound_aliases"]
    ).str.lower()
    library["is_hdac_annotated"] = annotation.str.contains(
        "hdac|histone deacetylase", regex=True
    )
    library.insert(0, "compound_index", np.arange(len(library), dtype=np.int64))

    stats = {
        "source_rows": int(len(source) + invalid_rows),
        "invalid_or_restricted_rows": invalid_rows,
        "unique_canonical_compounds": int(len(library)),
        "named_by_cache": int(library["cache_name"].ne("").sum()),
        "named_by_compoundinfo": int(library["compoundinfo_name"].ne("").sum()),
        "name_conflicts": int(library["name_conflict"].sum()),
        "hdac_annotated_compounds": int(library["is_hdac_annotated"].sum()),
    }
    return library, stats


def build_disease_matrix(
    disease_dir: Path,
    gene_ids: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    files = sorted(disease_dir.glob("signature_*.csv"))
    if not files:
        raise FileNotFoundError(f"No signature_*.csv files found under {disease_dir}")

    rows: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    gene_index = pd.Index(gene_ids, dtype=str)
    for path in files:
        cancer = path.stem.removeprefix("signature_")
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
        overlap = int(aligned.notna().sum())
        vector = aligned.fillna(0.0).to_numpy(dtype=np.float32)
        rows.append(vector)
        manifests.append({
            "cancer": cancer,
            "source_file": str(path),
            "value_column": value_column,
            "source_gene_rows": int(len(series)),
            "model_gene_overlap": overlap,
            "nonzero_aligned_genes": int(np.count_nonzero(vector)),
        })
    return np.stack(rows), pd.DataFrame(manifests)


def write_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_ids = load_gene_ids(args.gene_ids)
    library, library_stats = build_screening_library(args)
    disease_matrix, disease_manifest = build_disease_matrix(args.disease_dir, gene_ids)

    library.to_csv(args.output_dir / "screening_library.csv", index=False)
    pd.DataFrame({"gene_id": gene_ids}).to_csv(
        args.output_dir / "gene_ids.csv", index=False
    )
    np.save(args.output_dir / "disease_matrix.npy", disease_matrix)
    disease_manifest.to_csv(args.output_dir / "disease_manifest.csv", index=False)

    manifest = {
        **library_stats,
        "gene_count": len(gene_ids),
        "cancer_count": int(disease_matrix.shape[0]),
        "disease_matrix_shape": list(disease_matrix.shape),
        "wtrs_note": (
            "The screening script reports both the legacy reversal-only weighted score "
            "used by the original analysis and a signed weighted reversal sensitivity score."
        ),
    }
    (args.output_dir / "screening_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    zip_path = args.zip_output or args.output_dir.with_suffix(".zip")
    write_zip(args.output_dir, zip_path)
    print(json.dumps(manifest, indent=2))
    print(f"Screening bundle: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
