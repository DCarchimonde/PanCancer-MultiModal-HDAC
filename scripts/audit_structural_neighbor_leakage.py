from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import (  # noqa: E402
    assign_candidates,
    candidate_maps,
    resolve_candidates,
)


DEFAULT_SPLITS = [
    "pair_split",
    "leave_drug_out",
    "leave_cell_line_out",
    "scaffold_split",
    "hdac_class_holdout",
]

CANDIDATE_ORDER = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
    "RG2833",
    "Tianeptinaline_or_BG-1010",
]

FOCUS_CANDIDATES = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "RG2833",
    "Tianeptinaline_or_BG-1010",
]

STRICT_HDAC_CANDIDATES = {
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "PCI-24781",
    "Panobinostat",
    "Entinostat",
    "Vorinostat",
}

STRICT_LEAVE_DRUG_CANDIDATES = {"Mocetinostat", "PCI-24781"}

CANDIDATE_ROLES = {
    "Mocetinostat": "core_candidate",
    "NCH-51": "secondary_candidate",
    "TC-H-106": "original_method_sensitive",
    "Belinostat": "class_support",
    "PCI-24781": "class_support",
    "Panobinostat": "class_support",
    "Entinostat": "reference_control",
    "Vorinostat": "reference_control",
    "RG2833": "prediction_only",
    "Tianeptinaline_or_BG-1010": "identity_conflict_excluded",
}

SPLIT_LABELS = {
    "pair_split": "Pair",
    "leave_drug_out": "Drug",
    "leave_cell_line_out": "Cell",
    "scaffold_split": "Scaffold",
    "hdac_class_holdout": "HDAC",
}

THRESHOLDS = [0.7, 0.8, 0.9]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact-compound, Morgan/Tanimoto-neighbor, and Murcko-scaffold "
            "exposure of revision candidates in each training split."
        )
    )
    parser.add_argument(
        "--screening-library",
        type=Path,
        default=Path("data/screening_input/screening_library.csv"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/splits/generalization"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/structural_neighbor_audit"),
    )
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2_048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "restricted"}:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def murcko_scaffold(canonical_smiles: str) -> str:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise RuntimeError(f"Could not parse canonical SMILES: {canonical_smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def join_unique(values: pd.Series) -> str:
    return " | ".join(
        sorted(
            {
                str(value).strip()
                for value in values
                if pd.notna(value)
                and str(value).strip()
                and str(value).strip().lower() != "nan"
            }
        )
    )


def assign_candidates_with_canonical_fallback(
    frame: pd.DataFrame,
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
) -> pd.Series:
    assigned = assign_candidates(frame, drug_map, smiles_map)
    missing = assigned.isna()
    if missing.any():
        canonical = frame.loc[missing, "canonical_smiles"].map(
            canonicalize_smiles
        )
        assigned.loc[missing] = canonical.map(smiles_map)
    return assigned


def library_structure_lookup(library: pd.DataFrame) -> pd.DataFrame:
    frame = library.copy()
    frame["_canonical"] = frame["canonical_smiles"].map(canonicalize_smiles)
    frame = frame.dropna(subset=["_canonical"]).copy()
    name_columns = [
        column
        for column in ["display_name", "cache_name", "compoundinfo_name"]
        if column in frame.columns
    ]
    id_columns = [
        column
        for column in ["source_drug_ids", "compoundinfo_pert_id"]
        if column in frame.columns
    ]
    rows = []
    for canonical, subset in frame.groupby("_canonical", sort=True):
        names = set()
        identifiers = set()
        for column in name_columns:
            names.update(
                value
                for value in subset[column].fillna("").astype(str).str.strip()
                if value and value.lower() != "nan"
            )
        for column in id_columns:
            for value in subset[column].fillna("").astype(str):
                identifiers.update(
                    token.strip()
                    for token in value.split("|")
                    if token.strip() and token.strip().lower() != "nan"
                )
        rows.append(
            {
                "canonical_smiles": canonical,
                "library_names": " | ".join(sorted(names)),
                "library_drug_ids": " | ".join(sorted(identifiers)),
                "library_rows": len(subset),
            }
        )
    return pd.DataFrame(rows)


def prepare_training_structures(
    train: pd.DataFrame,
    library_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    frame = train.copy()
    frame["_canonical"] = frame["canonical_smiles"].map(canonicalize_smiles)
    invalid = int(frame["_canonical"].isna().sum())
    frame = frame.dropna(subset=["_canonical"]).copy()
    frame["_scaffold"] = frame["_canonical"].map(murcko_scaffold)
    grouped = (
        frame.groupby(["_canonical", "_scaffold"], as_index=False)
        .agg(
            training_signatures=("sig_id", "nunique"),
            training_drug_ids=("drug_id", join_unique),
            training_cell_lines=("cell_id", "nunique"),
        )
        .rename(
            columns={
                "_canonical": "canonical_smiles",
                "_scaffold": "murcko_scaffold",
            }
        )
    )
    grouped = grouped.merge(
        library_lookup,
        on="canonical_smiles",
        how="left",
        validate="one_to_one",
    )
    for column in ["library_names", "library_drug_ids"]:
        grouped[column] = grouped[column].fillna("")
    grouped["library_rows"] = grouped["library_rows"].fillna(0).astype(int)
    return grouped, invalid


def candidate_identity_table(library: pd.DataFrame) -> pd.DataFrame:
    resolved = resolve_candidates(library)
    require_columns(
        resolved,
        [
            "candidate",
            "resolved",
            "compound_index",
            "canonical_smiles",
            "name_conflict",
        ],
        "candidate identity map",
    )
    resolved = resolved.loc[resolved["resolved"].astype(bool)].copy()
    if set(resolved["candidate"]) != set(CANDIDATE_ORDER):
        raise RuntimeError(
            f"Resolved candidate set changed: {sorted(resolved['candidate'])}"
        )
    resolved["canonical_smiles"] = resolved["canonical_smiles"].map(
        canonicalize_smiles
    )
    if resolved["canonical_smiles"].isna().any():
        failed = resolved.loc[
            resolved["canonical_smiles"].isna(), "candidate"
        ].tolist()
        raise RuntimeError(f"Candidate structures could not be parsed: {failed}")
    resolved["murcko_scaffold"] = resolved["canonical_smiles"].map(
        murcko_scaffold
    )
    resolved["candidate_role"] = resolved["candidate"].map(CANDIDATE_ROLES)
    return resolved


def audit_split(
    split: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: pd.DataFrame,
    drug_map: dict[str, str],
    smiles_map: dict[str, str],
    library_lookup: pd.DataFrame,
    fingerprint_generator,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    key_columns = ["sig_id", "canonical_smiles", "drug_id", "cell_id"]
    require_columns(train, key_columns, f"{split} train")
    require_columns(test, key_columns, f"{split} test")
    signature_overlap = len(set(train["sig_id"]) & set(test["sig_id"]))
    if signature_overlap:
        raise RuntimeError(f"{split} has {signature_overlap} overlapping sig_id rows")

    train = train.copy()
    test = test.copy()
    train["candidate"] = assign_candidates_with_canonical_fallback(
        train, drug_map, smiles_map
    )
    test["candidate"] = assign_candidates_with_canonical_fallback(
        test, drug_map, smiles_map
    )
    training_structures, invalid_train_smiles = prepare_training_structures(
        train, library_lookup
    )
    if training_structures.empty:
        raise RuntimeError(f"{split} has no valid training structures")

    training_molecules = [
        Chem.MolFromSmiles(value)
        for value in training_structures["canonical_smiles"]
    ]
    training_fingerprints = [
        fingerprint_generator.GetFingerprint(molecule)
        for molecule in training_molecules
    ]
    training_scaffolds = set(
        value
        for value in training_structures["murcko_scaffold"]
        if value
    )
    training_canonical = set(training_structures["canonical_smiles"])

    test_canonical = test["canonical_smiles"].map(canonicalize_smiles)
    test_scaffolds = {
        scaffold
        for value in test_canonical.dropna().unique()
        if (scaffold := murcko_scaffold(value))
    }
    scaffold_overlap = len(training_scaffolds & test_scaffolds)
    if split == "scaffold_split" and scaffold_overlap:
        raise RuntimeError(
            f"scaffold_split has {scaffold_overlap} train/test Murcko overlaps"
        )

    summary_rows = []
    neighbor_rows = []
    for candidate_row in candidates.itertuples(index=False):
        molecule = Chem.MolFromSmiles(candidate_row.canonical_smiles)
        fingerprint = fingerprint_generator.GetFingerprint(molecule)
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint, training_fingerprints
            ),
            dtype=float,
        )
        order = np.argsort(-similarities, kind="stable")
        maximum = float(similarities[order[0]])
        exact_in_train = candidate_row.canonical_smiles in training_canonical
        scaffold_matches = int(
            (
                training_structures["murcko_scaffold"]
                == candidate_row.murcko_scaffold
            ).sum()
            if candidate_row.murcko_scaffold
            else 0
        )
        train_subset = train.loc[train["candidate"] == candidate_row.candidate]
        test_subset = test.loc[test["candidate"] == candidate_row.candidate]
        summary_rows.append(
            {
                "candidate": candidate_row.candidate,
                "candidate_role": candidate_row.candidate_role,
                "split": split,
                "candidate_canonical_smiles": candidate_row.canonical_smiles,
                "candidate_murcko_scaffold": candidate_row.murcko_scaffold,
                "train_signatures": int(train_subset["sig_id"].nunique()),
                "test_signatures": int(test_subset["sig_id"].nunique()),
                "training_unique_structures": len(training_structures),
                "max_tanimoto": maximum,
                "mean_top10_tanimoto": float(
                    similarities[order[: min(10, len(order))]].mean()
                ),
                "exact_structure_in_train": exact_in_train,
                "candidate_scaffold_in_train": bool(scaffold_matches),
                "same_scaffold_training_structures": scaffold_matches,
                "neighbors_ge_0_7": int((similarities >= 0.7).sum()),
                "neighbors_ge_0_8": int((similarities >= 0.8).sum()),
                "neighbors_ge_0_9": int((similarities >= 0.9).sum()),
                "name_conflict": bool(candidate_row.name_conflict),
            }
        )
        for rank, training_index in enumerate(order[:top_k], start=1):
            neighbor = training_structures.iloc[int(training_index)]
            neighbor_rows.append(
                {
                    "candidate": candidate_row.candidate,
                    "candidate_role": candidate_row.candidate_role,
                    "split": split,
                    "neighbor_rank": rank,
                    "tanimoto": float(similarities[training_index]),
                    "exact_structure": bool(
                        neighbor["canonical_smiles"]
                        == candidate_row.canonical_smiles
                    ),
                    "same_murcko_scaffold": bool(
                        candidate_row.murcko_scaffold
                        and neighbor["murcko_scaffold"]
                        == candidate_row.murcko_scaffold
                    ),
                    "neighbor_canonical_smiles": neighbor[
                        "canonical_smiles"
                    ],
                    "neighbor_murcko_scaffold": neighbor["murcko_scaffold"],
                    "training_signatures": int(
                        neighbor["training_signatures"]
                    ),
                    "training_cell_lines": int(
                        neighbor["training_cell_lines"]
                    ),
                    "training_drug_ids": neighbor["training_drug_ids"],
                    "library_names": neighbor["library_names"],
                    "library_drug_ids": neighbor["library_drug_ids"],
                }
            )

    inventory = {
        "split": split,
        "train_signatures": int(train["sig_id"].nunique()),
        "test_signatures": int(test["sig_id"].nunique()),
        "train_unique_structures": len(training_structures),
        "test_unique_valid_structures": int(test_canonical.nunique(dropna=True)),
        "invalid_train_smiles": invalid_train_smiles,
        "invalid_test_smiles": int(test_canonical.isna().sum()),
        "train_test_sig_id_overlap": signature_overlap,
        "train_test_murcko_scaffold_overlap": scaffold_overlap,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(neighbor_rows), inventory


def validate_strict_exclusions(summary: pd.DataFrame) -> None:
    hdac = summary.loc[
        (summary["split"] == "hdac_class_holdout")
        & summary["candidate"].isin(STRICT_HDAC_CANDIDATES)
    ]
    if len(hdac) != len(STRICT_HDAC_CANDIDATES):
        raise RuntimeError("Incomplete corrected-HDAC candidate similarity rows")
    leaked_hdac = hdac.loc[hdac["exact_structure_in_train"], "candidate"].tolist()
    if leaked_hdac:
        raise RuntimeError(
            f"Exact candidate structures remain in corrected-HDAC train: {leaked_hdac}"
        )

    leave_drug = summary.loc[
        (summary["split"] == "leave_drug_out")
        & summary["candidate"].isin(STRICT_LEAVE_DRUG_CANDIDATES)
    ]
    if len(leave_drug) != len(STRICT_LEAVE_DRUG_CANDIDATES):
        raise RuntimeError("Incomplete strict leave-drug candidate similarity rows")
    leaked_drugs = leave_drug.loc[
        leave_drug["exact_structure_in_train"], "candidate"
    ].tolist()
    if leaked_drugs:
        raise RuntimeError(
            f"Exact candidate structures remain in leave-drug train: {leaked_drugs}"
        )


def make_figure(
    summary: pd.DataFrame,
    splits: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    max_similarity = summary.pivot(
        index="candidate", columns="split", values="max_tanimoto"
    ).reindex(index=CANDIDATE_ORDER, columns=splits)
    exact = summary.pivot(
        index="candidate",
        columns="split",
        values="exact_structure_in_train",
    ).reindex(index=CANDIDATE_ORDER, columns=splits)
    scaffold = summary.pivot(
        index="candidate",
        columns="split",
        values="candidate_scaffold_in_train",
    ).reindex(index=CANDIDATE_ORDER, columns=splits)
    if max_similarity.isna().any().any():
        raise RuntimeError("Structural-neighbor figure matrix is incomplete")

    figure = plt.figure(figsize=(20, 13), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    heat_axis = figure.add_subplot(grid[0, :])
    image = heat_axis.imshow(
        max_similarity.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    heat_axis.set_xticks(np.arange(len(splits)))
    heat_axis.set_xticklabels([SPLIT_LABELS.get(split, split) for split in splits])
    heat_axis.set_yticks(np.arange(len(CANDIDATE_ORDER)))
    heat_axis.set_yticklabels(CANDIDATE_ORDER)
    heat_axis.set_title(
        "A  Maximum Morgan/Tanimoto similarity to each split's training structures",
        loc="left",
        fontweight="bold",
    )
    for row in range(len(CANDIDATE_ORDER)):
        for column in range(len(splits)):
            value = float(max_similarity.iloc[row, column])
            marker = "*" if bool(exact.iloc[row, column]) else ""
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance >= 0.53 else "white"
            heat_axis.text(
                column,
                row,
                f"{value:.2f}{marker}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.025, pad=0.01)
    colorbar.set_label("Maximum Tanimoto similarity (* = exact structure in train)")

    bar_axis = figure.add_subplot(grid[1, 0])
    hdac_focus = (
        summary.loc[
            (summary["split"] == "hdac_class_holdout")
            & summary["candidate"].isin(FOCUS_CANDIDATES)
        ]
        .set_index("candidate")
        .reindex(FOCUS_CANDIDATES)
    )
    positions = np.arange(len(FOCUS_CANDIDATES))
    width = 0.24
    threshold_columns = [
        ("neighbors_ge_0_7", ">= 0.7", "#67a9cf"),
        ("neighbors_ge_0_8", ">= 0.8", "#ef8a62"),
        ("neighbors_ge_0_9", ">= 0.9", "#b2182b"),
    ]
    for offset, (column, label, color) in enumerate(threshold_columns):
        bars = bar_axis.bar(
            positions + (offset - 1) * width,
            hdac_focus[column].to_numpy(dtype=float),
            width=width,
            label=label,
            color=color,
        )
        bar_axis.bar_label(bars, fontsize=8, padding=2)
    bar_axis.set_xticks(positions)
    bar_axis.set_xticklabels(FOCUS_CANDIDATES, rotation=30, ha="right")
    bar_axis.set_ylabel("Unique corrected-HDAC training structures")
    bar_axis.set_title(
        "B  High-similarity analog counts in corrected-HDAC training",
        loc="left",
        fontweight="bold",
    )
    bar_axis.grid(axis="y", alpha=0.25)
    bar_axis.legend(frameon=False)

    scaffold_axis = figure.add_subplot(grid[1, 1])
    scaffold_values = scaffold.to_numpy(dtype=float)
    scaffold_image = scaffold_axis.imshow(
        scaffold_values,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    scaffold_axis.set_xticks(np.arange(len(splits)))
    scaffold_axis.set_xticklabels(
        [SPLIT_LABELS.get(split, split) for split in splits]
    )
    scaffold_axis.set_yticks(np.arange(len(CANDIDATE_ORDER)))
    scaffold_axis.set_yticklabels(CANDIDATE_ORDER)
    scaffold_axis.set_title(
        "C  Candidate Murcko scaffold present in training",
        loc="left",
        fontweight="bold",
    )
    for row in range(len(CANDIDATE_ORDER)):
        for column in range(len(splits)):
            present = bool(scaffold.iloc[row, column])
            scaffold_axis.text(
                column,
                row,
                "Yes" if present else "No",
                ha="center",
                va="center",
                color="white" if present else "black",
                fontsize=8,
            )
    figure.colorbar(
        scaffold_image,
        ax=scaffold_axis,
        fraction=0.035,
        pad=0.02,
        ticks=[0, 1],
        label="Scaffold overlap",
    )

    figure.suptitle(
        "Candidate structural-neighbor and analog-leakage audit",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.012,
        (
            "Morgan radius 2, 2048-bit fingerprints with chirality. Similarity and "
            "scaffold exposure quantify chemical novelty; they are not efficacy or "
            "target-validation evidence. RG2833 is prediction-only and "
            "Tianeptinaline/BG-1010 remains identity-conflicted."
        ),
        ha="center",
        fontsize=9,
    )
    figure.savefig(
        output_dir / "structural_neighbor_audit_figure.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "structural_neighbor_audit_figure.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    splits = parse_csv_list(args.splits)
    if len(splits) != len(set(splits)):
        raise RuntimeError("Duplicate split names were requested")
    if args.radius < 1 or args.fingerprint_bits < 64 or args.top_k < 1:
        raise RuntimeError(
            "radius and top-k must be positive; fingerprint-bits must be >= 64"
        )
    if not {"leave_drug_out", "hdac_class_holdout"}.issubset(splits):
        raise RuntimeError(
            "Strict structural audit requires leave_drug_out and hdac_class_holdout"
        )

    library = pd.read_csv(args.screening_library, low_memory=False)
    require_columns(
        library,
        ["compound_index", "canonical_smiles"],
        "screening library",
    )
    candidates = candidate_identity_table(library)
    drug_map, smiles_map = candidate_maps(candidates)
    library_lookup = library_structure_lookup(library)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.fingerprint_bits,
        includeChirality=True,
    )

    summary_frames = []
    neighbor_frames = []
    inventory_rows = []
    input_hashes = {}
    for split in splits:
        train_path = args.split_root / split / "train.csv"
        test_path = args.split_root / split / "test.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing split input: {train_path} or {test_path}"
            )
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        summary, neighbors, inventory = audit_split(
            split,
            train,
            test,
            candidates,
            drug_map,
            smiles_map,
            library_lookup,
            fingerprint_generator,
            args.top_k,
        )
        summary_frames.append(summary)
        neighbor_frames.append(neighbors)
        inventory_rows.append(inventory)
        input_hashes[f"{split}_train_sha256"] = sha256_file(train_path)
        input_hashes[f"{split}_test_sha256"] = sha256_file(test_path)

    similarity_summary = pd.concat(summary_frames, ignore_index=True)
    top_neighbors = pd.concat(neighbor_frames, ignore_index=True)
    split_inventory = pd.DataFrame(inventory_rows)
    validate_strict_exclusions(similarity_summary)

    similarity_summary.to_csv(
        args.output_dir / "candidate_train_similarity_summary.csv", index=False
    )
    top_neighbors.to_csv(
        args.output_dir / "candidate_top10_training_neighbors.csv", index=False
    )
    split_inventory.to_csv(
        args.output_dir / "structural_split_inventory.csv", index=False
    )
    candidates.to_csv(
        args.output_dir / "candidate_structure_identity_snapshot.csv", index=False
    )
    make_figure(similarity_summary, splits, args.output_dir, args.figure_dpi)

    runtime = time.perf_counter() - started
    manifest = {
        "splits": splits,
        "candidates": CANDIDATE_ORDER,
        "focus_candidates": FOCUS_CANDIDATES,
        "radius": args.radius,
        "fingerprint_bits": args.fingerprint_bits,
        "include_chirality": True,
        "similarity": "Tanimoto",
        "neighbor_thresholds": THRESHOLDS,
        "top_k": args.top_k,
        "strict_hdac_exact_structure_leaks": int(
            similarity_summary.loc[
                (similarity_summary["split"] == "hdac_class_holdout")
                & similarity_summary["candidate"].isin(STRICT_HDAC_CANDIDATES),
                "exact_structure_in_train",
            ].sum()
        ),
        "strict_leave_drug_exact_structure_leaks": int(
            similarity_summary.loc[
                (similarity_summary["split"] == "leave_drug_out")
                & similarity_summary["candidate"].isin(
                    STRICT_LEAVE_DRUG_CANDIDATES
                ),
                "exact_structure_in_train",
            ].sum()
        ),
        "runtime_seconds": runtime,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "screening_library_sha256": sha256_file(args.screening_library),
        **input_hashes,
        "identity_policy": (
            "RG2833 is retained as prediction-only. Tianeptinaline/BG-1010 is "
            "reported only as an identity-conflict sensitivity row."
        ),
        "evidence_note": (
            "Structural-neighbor exposure assesses leakage and chemical novelty; "
            "it is not direct efficacy, target, or mechanism validation."
        ),
        "outputs": [
            "candidate_train_similarity_summary.csv",
            "candidate_top10_training_neighbors.csv",
            "structural_split_inventory.csv",
            "candidate_structure_identity_snapshot.csv",
            "structural_neighbor_audit_figure.png",
            "structural_neighbor_audit_figure.pdf",
            "structural_neighbor_audit_manifest.json",
            "audit.log",
        ],
    }
    (args.output_dir / "structural_neighbor_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    focus_display = similarity_summary.loc[
        similarity_summary["candidate"].isin(FOCUS_CANDIDATES),
        [
            "candidate",
            "candidate_role",
            "split",
            "train_signatures",
            "test_signatures",
            "max_tanimoto",
            "exact_structure_in_train",
            "candidate_scaffold_in_train",
            "neighbors_ge_0_7",
            "neighbors_ge_0_8",
            "neighbors_ge_0_9",
        ],
    ]
    hdac_top_neighbors = top_neighbors.loc[
        (top_neighbors["split"] == "hdac_class_holdout")
        & top_neighbors["candidate"].isin(FOCUS_CANDIDATES)
        & (top_neighbors["neighbor_rank"] <= 3),
        [
            "candidate",
            "neighbor_rank",
            "tanimoto",
            "same_murcko_scaffold",
            "library_names",
            "training_drug_ids",
        ],
    ]
    report = "\n".join(
        [
            "===== FOCUS-CANDIDATE STRUCTURAL EXPOSURE =====",
            focus_display.round(6).to_string(index=False),
            "",
            "===== CORRECTED-HDAC TOP 3 TRAINING NEIGHBORS =====",
            hdac_top_neighbors.round(6).to_string(index=False),
            "",
            "===== SPLIT INVENTORY =====",
            split_inventory.to_string(index=False),
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "STRUCTURAL NEIGHBOR / ANALOG LEAKAGE AUDIT: PASSED",
        ]
    )
    (args.output_dir / "audit.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
