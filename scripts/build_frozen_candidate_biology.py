from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import resolve_candidates  # noqa: E402


MODELS = ["dual_stream", "fingerprint_mlp"]
SEEDS = [1, 2, 3]

# This table is intentionally explicit and version-controlled.  It freezes the
# candidate interpretation after the corrected generalization, measured-LINCS,
# condition/QC, stability, and structural-neighbor audits.  Later scripts must
# consume this table instead of silently changing the candidate set.
FROZEN_CANDIDATES = [
    {
        "candidate": "Mocetinostat",
        "candidate_role": "core_candidate",
        "biology_scope": "primary_network_and_pathway",
        "include_gene_rerun": True,
        "evidence_status": "measured_and_prediction_supported",
        "freeze_reason": (
            "Clean strict holdout/structural-neighbor evidence and official HiQ "
            "measured LINCS reversal support."
        ),
    },
    {
        "candidate": "NCH-51",
        "candidate_role": "secondary_candidate",
        "biology_scope": "primary_network_and_pathway",
        "include_gene_rerun": True,
        "evidence_status": "measured_support_with_scaffold_exposure_caveat",
        "freeze_reason": (
            "Official HiQ measured reversal support; retained as secondary because "
            "its candidate scaffold is represented in HDAC-holdout training data."
        ),
    },
    {
        "candidate": "TC-H-106",
        "candidate_role": "exploratory_candidate",
        "biology_scope": "primary_network_and_pathway_with_caveat",
        "include_gene_rerun": True,
        "evidence_status": "limited_measured_and_analog_exposed",
        "freeze_reason": (
            "Retained to test the original method-sensitive signal, but interpreted "
            "as exploratory because HiQ coverage is limited and a close training "
            "neighbor is present."
        ),
    },
    {
        "candidate": "Belinostat",
        "candidate_role": "class_support",
        "biology_scope": "sensitivity_only",
        "include_gene_rerun": True,
        "evidence_status": "measured_class_support",
        "freeze_reason": (
            "Included only as an additional measured HDAC-class sensitivity comparator."
        ),
    },
    {
        "candidate": "PCI-24781",
        "candidate_role": "class_support",
        "biology_scope": "measured_validation_only",
        "include_gene_rerun": False,
        "evidence_status": "measured_class_support_not_primary_candidate",
        "freeze_reason": (
            "Retained in measured validation figures but excluded from the primary "
            "candidate network/pathway rerun."
        ),
    },
    {
        "candidate": "Panobinostat",
        "candidate_role": "class_support",
        "biology_scope": "measured_validation_only",
        "include_gene_rerun": False,
        "evidence_status": "measured_class_support_not_primary_candidate",
        "freeze_reason": (
            "Retained in measured validation figures but excluded from the primary "
            "candidate network/pathway rerun."
        ),
    },
    {
        "candidate": "Entinostat",
        "candidate_role": "reference_control",
        "biology_scope": "reference_control",
        "include_gene_rerun": True,
        "evidence_status": "measured_reference_control",
        "freeze_reason": "Established HDAC reference comparator for biological interpretation.",
    },
    {
        "candidate": "Vorinostat",
        "candidate_role": "reference_control",
        "biology_scope": "reference_control",
        "include_gene_rerun": True,
        "evidence_status": "measured_reference_control",
        "freeze_reason": "High-coverage HDAC reference comparator for biological interpretation.",
    },
    {
        "candidate": "RG2833",
        "candidate_role": "prediction_only",
        "biology_scope": "excluded_from_measured_gene_consensus",
        "include_gene_rerun": False,
        "evidence_status": "no_measured_lincs_signature",
        "freeze_reason": (
            "No measured LINCS signature; may be reported only as prediction-only "
            "and cannot support the measured biology rerun."
        ),
    },
    {
        "candidate": "Tianeptinaline_or_BG-1010",
        "candidate_role": "identity_conflict_excluded",
        "biology_scope": "excluded",
        "include_gene_rerun": False,
        "evidence_status": "identity_conflict_and_training_exposure",
        "freeze_reason": (
            "Identity conflict and exact HDAC-training exposure preclude primary inference."
        ),
    },
]

PRIMARY_CANDIDATES = ["Mocetinostat", "NCH-51", "TC-H-106"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the post-audit candidate set and build predicted/measured "
            "reversal-associated gene consensus tables for network and pathway reruns."
        )
    )
    parser.add_argument(
        "--screening-input", type=Path, default=Path("data/screening_input")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/revision"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results/revision/benchmarks"),
    )
    parser.add_argument(
        "--condition-root",
        type=Path,
        default=Path("results/revision/lincs_condition_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/frozen_candidate_biology"),
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.70,
        help=(
            "Predeclared minimum reversal/support fraction for the primary gene set. "
            "Sensitivity counts at 0.60, 0.70, and 0.80 are always exported."
        ),
    )
    parser.add_argument("--network-top-k", type=int, default=200)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow checkpoint inference on CPU when CUDA is unavailable.",
    )
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


def boolean_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "yes", "y"})
    )


def normalized_unit(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
    )


def normalize_values(
    values: pd.Series,
    units: pd.Series,
    factors: dict[str, float],
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unit_values = units.map(normalized_unit)
    unknown = sorted(set(unit_values.loc[numeric.notna()]) - set(factors))
    if unknown:
        raise RuntimeError(f"Unsupported official {label} units: {unknown}")
    converted = numeric * unit_values.map(factors)
    if converted.isna().any():
        raise RuntimeError(
            f"Could not normalize {int(converted.isna().sum())} official {label} rows"
        )
    return converted.astype(float)


def load_gene_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    if frame.empty:
        raise RuntimeError(f"Empty gene table: {path}")
    gene_column = "gene_id" if "gene_id" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={gene_column: "gene_id"}).copy()
    frame["gene_id"] = frame["gene_id"].astype(str).str.strip()
    if frame["gene_id"].eq("").any() or frame["gene_id"].duplicated().any():
        raise RuntimeError(f"Gene IDs must be non-empty and unique: {path}")
    return frame


def build_frozen_table(library: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.DataFrame(FROZEN_CANDIDATES)
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
        "resolved candidate table",
    )
    result = frozen.merge(
        resolved,
        on="candidate",
        how="left",
        validate="one_to_one",
    )
    if result["resolved"].isna().any():
        missing = result.loc[result["resolved"].isna(), "candidate"].tolist()
        raise RuntimeError(f"Frozen candidates absent from candidate resolver: {missing}")
    included = result.loc[result["include_gene_rerun"]]
    unresolved = included.loc[~included["resolved"].astype(bool), "candidate"].tolist()
    if unresolved:
        raise RuntimeError(f"Gene-rerun candidates could not be resolved: {unresolved}")
    if included["compound_index"].duplicated().any():
        duplicates = included.loc[
            included["compound_index"].duplicated(keep=False),
            ["candidate", "compound_index"],
        ]
        raise RuntimeError(
            "Included candidates resolve to duplicate compounds:\n"
            f"{duplicates.to_string(index=False)}"
        )
    return result


def predict_candidate_profiles(
    candidates: pd.DataFrame,
    checkpoint_root: Path,
    models: list[str],
    seeds: list[int],
    gene_count: int,
    disable_amp: bool,
    allow_cpu: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    # Heavy dependencies stay inside this function so table/aggregation helpers
    # can be tested without a GPU-enabled PyTorch installation.
    import torch

    from src.revision_dataset import featurize_smiles
    from src.revision_models import build_model

    unknown_models = sorted(set(models) - set(MODELS))
    if unknown_models:
        raise RuntimeError(f"Unsupported models: {unknown_models}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Use --allow-cpu only if CPU inference is intended."
        )
    amp_enabled = device.type == "cuda" and not disable_amp

    atom_arrays = []
    fingerprints = []
    for smiles in candidates["canonical_smiles"].astype(str):
        features = featurize_smiles(smiles)
        atom_arrays.append(features.atom_features)
        fingerprints.append(features.fingerprint)
    max_atoms = max(array.shape[0] for array in atom_arrays)
    feature_dim = atom_arrays[0].shape[1]
    atoms = torch.zeros(
        (len(candidates), max_atoms, feature_dim), dtype=torch.float32
    )
    atom_mask = torch.zeros((len(candidates), max_atoms), dtype=torch.bool)
    for row, atom_features in enumerate(atom_arrays):
        count = atom_features.shape[0]
        atoms[row, :count] = atom_features
        atom_mask[row, :count] = True
    fingerprint_batch = torch.stack(fingerprints)
    atoms = atoms.to(device)
    atom_mask = atom_mask.to(device)
    fingerprint_batch = fingerprint_batch.to(device)

    profiles: list[np.ndarray] = []
    inventory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for model_name in models:
        for seed in seeds:
            checkpoint_path = (
                checkpoint_root
                / "hdac_class_holdout"
                / model_name
                / f"seed_{seed}"
                / "best_model.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing corrected HDAC checkpoint: {checkpoint_path}"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            arguments = checkpoint.get("arguments", {})
            hidden_dim = int(arguments.get("hidden_dim", 512))
            recorded_model = str(arguments.get("model", model_name))
            if recorded_model != model_name:
                raise RuntimeError(
                    f"Checkpoint model mismatch at {checkpoint_path}: "
                    f"{recorded_model!r} != {model_name!r}"
                )
            model = build_model(model_name, hidden_dim, gene_count).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            run_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(atoms, atom_mask, fingerprint_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values = prediction.float().cpu().numpy()
            if values.shape != (len(candidates), gene_count):
                raise RuntimeError(
                    f"Unexpected prediction shape for {model_name}/seed {seed}: "
                    f"{values.shape}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Non-finite predictions for {model_name}/seed {seed}"
                )
            profiles.append(values.astype(np.float32, copy=False))
            inventory.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_validation_mse": checkpoint.get("validation_mse"),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "runtime_seconds": time.perf_counter() - run_started,
                }
            )
            del prediction, model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # candidate x model-seed run x gene
    stacked = np.stack(profiles, axis=1)
    device_metadata = {
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "amp_enabled": amp_enabled,
        "runtime_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    return stacked, pd.DataFrame(inventory), device_metadata


def aggregate_measured_hiq_profiles(
    conditions: pd.DataFrame,
    expression: np.ndarray,
    included_candidates: list[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    required = [
        "cache_row",
        "sig_id",
        "candidate",
        "official_cell_iname",
        "official_pert_dose",
        "official_pert_dose_unit",
        "official_pert_time",
        "official_pert_time_unit",
        "official_is_hiq",
    ]
    require_columns(conditions, required, "official candidate conditions")
    if conditions["sig_id"].duplicated().any():
        raise RuntimeError("Official candidate conditions contain duplicate sig_id")

    frame = conditions.loc[
        conditions["candidate"].isin(included_candidates)
        & boolean_series(conditions["official_is_hiq"])
    ].copy()
    if frame.empty:
        raise RuntimeError("No included candidate has an official HiQ profile")
    frame["official_cell_iname"] = (
        frame["official_cell_iname"].fillna("").astype(str).str.strip()
    )
    if frame["official_cell_iname"].eq("").any():
        raise RuntimeError("At least one HiQ row lacks official_cell_iname")
    frame["dose_um"] = normalize_values(
        frame["official_pert_dose"],
        frame["official_pert_dose_unit"],
        {
            "um": 1.0,
            "micromolar": 1.0,
            "nm": 0.001,
            "nanomolar": 0.001,
            "mm": 1_000.0,
            "millimolar": 1_000.0,
            "pm": 0.000001,
            "picomolar": 0.000001,
        },
        "dose",
    )
    frame["time_h"] = normalize_values(
        frame["official_pert_time"],
        frame["official_pert_time_unit"],
        {
            "h": 1.0,
            "hr": 1.0,
            "hrs": 1.0,
            "hour": 1.0,
            "hours": 1.0,
            "min": 1.0 / 60.0,
            "mins": 1.0 / 60.0,
            "minute": 1.0 / 60.0,
            "minutes": 1.0 / 60.0,
            "d": 24.0,
            "day": 24.0,
            "days": 24.0,
        },
        "time",
    )
    frame["cache_row"] = pd.to_numeric(
        frame["cache_row"], errors="raise"
    ).astype(np.int64)
    if (frame["cache_row"] < 0).any() or (
        frame["cache_row"] >= expression.shape[0]
    ).any():
        raise RuntimeError("Official-condition cache_row is outside expression array")

    missing = sorted(set(included_candidates) - set(frame["candidate"]))
    if missing:
        raise RuntimeError(f"Included candidates without official HiQ profiles: {missing}")

    condition_keys = [
        "candidate",
        "official_cell_iname",
        "dose_um",
        "time_h",
    ]
    condition_profiles: list[np.ndarray] = []
    condition_metadata: list[dict[str, Any]] = []
    for key, subset in frame.groupby(condition_keys, sort=True, dropna=False):
        rows = subset["cache_row"].to_numpy(dtype=np.int64)
        values = np.asarray(expression[rows], dtype=np.float32)
        condition_profiles.append(np.nanmedian(values, axis=0).astype(np.float32))
        condition_metadata.append(
            {
                "candidate": key[0],
                "official_cell_iname": key[1],
                "dose_um": float(key[2]),
                "time_h": float(key[3]),
                "signatures": int(subset["sig_id"].nunique()),
            }
        )
    condition_frame = pd.DataFrame(condition_metadata)
    condition_matrix = np.stack(condition_profiles)

    cell_profiles: dict[str, np.ndarray] = {}
    coverage_rows: list[dict[str, Any]] = []
    for candidate in included_candidates:
        candidate_condition_indices = condition_frame.index[
            condition_frame["candidate"] == candidate
        ].to_numpy(dtype=np.int64)
        candidate_conditions = condition_frame.loc[candidate_condition_indices]
        cell_values = []
        for cell in sorted(candidate_conditions["official_cell_iname"].unique()):
            local_indices = candidate_conditions.index[
                candidate_conditions["official_cell_iname"] == cell
            ].to_numpy(dtype=np.int64)
            cell_values.append(
                np.nanmedian(condition_matrix[local_indices], axis=0).astype(np.float32)
            )
        cell_profiles[candidate] = np.stack(cell_values)
        source_rows = frame.loc[frame["candidate"] == candidate]
        coverage_rows.append(
            {
                "candidate": candidate,
                "hiq_signatures": int(source_rows["sig_id"].nunique()),
                "official_cell_lines": len(cell_values),
                "normalized_conditions": len(candidate_condition_indices),
                "dose_levels": int(source_rows["dose_um"].nunique()),
                "time_levels": int(source_rows["time_h"].nunique()),
                "dose_um_min": float(source_rows["dose_um"].min()),
                "dose_um_max": float(source_rows["dose_um"].max()),
                "times_h": " | ".join(
                    f"{value:g}" for value in sorted(source_rows["time_h"].unique())
                ),
            }
        )
    return cell_profiles, pd.DataFrame(coverage_rows)


def safe_nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmedian(values, axis=axis)


def summarize_gene_evidence(
    profiles: np.ndarray,
    disease: np.ndarray,
    observed_mask: np.ndarray,
    gene_annotation: pd.DataFrame,
    evidence_prefix: str,
) -> pd.DataFrame:
    if profiles.ndim != 2:
        raise RuntimeError(f"Profiles must be source x gene, got {profiles.shape}")
    if profiles.shape[0] == 0 or not np.isfinite(profiles).all():
        raise RuntimeError(
            "Gene evidence requires at least one finite source profile"
        )
    if disease.shape != observed_mask.shape:
        raise RuntimeError("Disease matrix and observed mask shapes differ")
    if profiles.shape[1] != disease.shape[1] or len(gene_annotation) != disease.shape[1]:
        raise RuntimeError("Profile/disease/gene dimensions do not align")

    valid = observed_mask.astype(bool) & np.isfinite(disease)
    denominator = np.where(valid, np.abs(disease), 0.0).sum(axis=1)
    if (denominator <= 0).any():
        raise RuntimeError("At least one cancer has no disease-signature denominator")

    # source x cancer x gene.  Positive values mean transcriptomic reversal;
    # this is an association score and is not evidence of direct binding.
    contribution = (
        -profiles[:, None, :]
        * disease[None, :, :]
        / denominator[None, :, None]
    ).astype(np.float32, copy=False)
    expanded_valid = np.broadcast_to(valid[None, :, :], contribution.shape)
    contribution = np.where(expanded_valid, contribution, np.nan)

    unit_counts = expanded_valid.sum(axis=(0, 1))
    reversal_counts = ((contribution > 0) & expanded_valid).sum(axis=(0, 1))
    reversal_fraction = np.divide(
        reversal_counts,
        unit_counts,
        out=np.full(contribution.shape[2], np.nan, dtype=float),
        where=unit_counts > 0,
    )
    source_medians = safe_nanmedian(contribution, axis=1)
    cancer_medians = safe_nanmedian(contribution, axis=0)
    source_consensus = np.mean(source_medians > 0, axis=0)
    cancer_valid = valid.sum(axis=0)
    cancer_consensus = np.divide(
        ((cancer_medians > 0) & valid).sum(axis=0),
        cancer_valid,
        out=np.full(contribution.shape[2], np.nan, dtype=float),
        where=cancer_valid > 0,
    )
    disease_up_fraction = np.divide(
        ((disease > 0) & valid).sum(axis=0),
        cancer_valid,
        out=np.full(contribution.shape[2], np.nan, dtype=float),
        where=cancer_valid > 0,
    )

    output = gene_annotation.copy()
    output[f"{evidence_prefix}_source_units"] = profiles.shape[0]
    output[f"{evidence_prefix}_source_cancer_units"] = unit_counts
    output[f"{evidence_prefix}_reversal_fraction"] = reversal_fraction
    output[f"{evidence_prefix}_source_consensus_fraction"] = source_consensus
    output[f"{evidence_prefix}_cancer_consensus_fraction"] = cancer_consensus
    output[f"{evidence_prefix}_median_signed_contribution"] = safe_nanmedian(
        contribution.reshape(-1, contribution.shape[2]), axis=0
    )
    output[f"{evidence_prefix}_mean_signed_contribution"] = np.nanmean(
        contribution, axis=(0, 1)
    )
    output[f"{evidence_prefix}_drug_up_fraction"] = np.mean(profiles > 0, axis=0)
    output["disease_up_fraction"] = disease_up_fraction
    output["disease_observed_cancers"] = cancer_valid
    return output


def classify_direction(up_fraction: pd.Series, threshold: float = 0.70) -> pd.Series:
    result = pd.Series("mixed", index=up_fraction.index, dtype="object")
    result.loc[up_fraction >= threshold] = "up"
    result.loc[up_fraction <= 1.0 - threshold] = "down"
    return result


def consensus_mask(frame: pd.DataFrame, threshold: float) -> pd.Series:
    columns = [
        "predicted_reversal_fraction",
        "predicted_source_consensus_fraction",
        "predicted_cancer_consensus_fraction",
        "measured_reversal_fraction",
        "measured_source_consensus_fraction",
        "measured_cancer_consensus_fraction",
    ]
    return (
        frame[columns].ge(threshold).all(axis=1)
        & frame["predicted_mean_signed_contribution"].gt(0)
        & frame["measured_mean_signed_contribution"].gt(0)
        & frame["measured_source_units"].ge(2)
    )


def build_candidate_consensus(
    predicted: pd.DataFrame,
    measured: pd.DataFrame,
    candidate_role: str,
    biology_scope: str,
    threshold: float,
    top_k: int,
) -> pd.DataFrame:
    annotation_columns = [
        column
        for column in predicted.columns
        if not column.startswith("predicted_")
        and column not in {"disease_up_fraction", "disease_observed_cancers"}
    ]
    predicted_columns = annotation_columns + [
        column for column in predicted.columns if column.startswith("predicted_")
    ] + ["disease_up_fraction", "disease_observed_cancers"]
    measured_columns = ["gene_id"] + [
        column for column in measured.columns if column.startswith("measured_")
    ]
    consensus = predicted[predicted_columns].merge(
        measured[measured_columns],
        on="gene_id",
        how="inner",
        validate="one_to_one",
    )
    consensus.insert(0, "candidate", predicted.attrs["candidate"])
    consensus.insert(1, "candidate_role", candidate_role)
    consensus.insert(2, "biology_scope", biology_scope)

    rank_columns = [
        "predicted_reversal_fraction",
        "predicted_source_consensus_fraction",
        "predicted_cancer_consensus_fraction",
        "predicted_mean_signed_contribution",
        "measured_reversal_fraction",
        "measured_source_consensus_fraction",
        "measured_cancer_consensus_fraction",
        "measured_mean_signed_contribution",
    ]
    percentile_columns = []
    for column in rank_columns:
        percentile_column = f"{column}_percentile"
        consensus[percentile_column] = consensus[column].rank(
            pct=True, method="average"
        )
        percentile_columns.append(percentile_column)
    consensus["consensus_rank_score"] = consensus[percentile_columns].mean(axis=1)
    consensus["passes_predeclared_consensus"] = consensus_mask(
        consensus, threshold
    )
    eligible = consensus.loc[consensus["passes_predeclared_consensus"]].sort_values(
        ["consensus_rank_score", "gene_id"], ascending=[False, True]
    )
    consensus["consensus_rank"] = np.nan
    consensus.loc[eligible.index, "consensus_rank"] = np.arange(1, len(eligible) + 1)
    consensus["selected_network_gene"] = (
        consensus["passes_predeclared_consensus"]
        & consensus["consensus_rank"].le(top_k)
    )
    consensus["disease_direction"] = classify_direction(
        consensus["disease_up_fraction"]
    )
    consensus["predicted_drug_direction"] = classify_direction(
        consensus["predicted_drug_up_fraction"]
    )
    consensus["measured_drug_direction"] = classify_direction(
        consensus["measured_drug_up_fraction"]
    )
    consensus["reversal_pattern"] = "mixed_or_context_dependent"
    up_down = (
        consensus["disease_direction"].eq("up")
        & consensus["predicted_drug_direction"].eq("down")
        & consensus["measured_drug_direction"].eq("down")
    )
    down_up = (
        consensus["disease_direction"].eq("down")
        & consensus["predicted_drug_direction"].eq("up")
        & consensus["measured_drug_direction"].eq("up")
    )
    consensus.loc[up_down, "reversal_pattern"] = "disease_up_drug_down"
    consensus.loc[down_up, "reversal_pattern"] = "disease_down_drug_up"
    return consensus


def sensitivity_counts(
    consensus: pd.DataFrame, thresholds: list[float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, subset in consensus.groupby("candidate", sort=False):
        for threshold in thresholds:
            mask = consensus_mask(subset, threshold)
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_role": subset["candidate_role"].iloc[0],
                    "consensus_threshold": threshold,
                    "eligible_genes": int(mask.sum()),
                    "stable_direction_genes": int(
                        (mask & subset["reversal_pattern"].ne("mixed_or_context_dependent")).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def upstream_provenance() -> list[dict[str, Any]]:
    paths = [
        Path("results/revision/candidate_stability/candidate_stability_manifest.json"),
        Path(
            "results/revision/structural_neighbor_audit/"
            "structural_neighbor_audit_manifest.json"
        ),
        Path(
            "results/revision/measured_lincs_figures/"
            "measured_lincs_figure_manifest.json"
        ),
        Path(
            "results/revision/leave_drug_candidate_validation/"
            "leave_drug_candidate_validation_manifest.json"
        ),
    ]
    return [
        {
            "path": str(path),
            "present": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    ]


def main() -> None:
    args = parse_args()
    if not 0.5 < args.consensus_threshold < 1.0:
        raise ValueError("--consensus-threshold must be between 0.5 and 1.0")
    if args.network_top_k <= 0:
        raise ValueError("--network-top-k must be positive")
    models = parse_csv_list(args.models)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    if not models or not seeds:
        raise ValueError("At least one model and seed are required")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    library_path = args.screening_input / "screening_library.csv"
    disease_path = args.screening_input / "disease_matrix.npy"
    observed_mask_path = args.screening_input / "disease_observed_mask.npy"
    disease_manifest_path = args.screening_input / "disease_manifest.csv"
    input_gene_path = args.screening_input / "gene_ids.csv"
    cache_gene_path = args.cache_dir / "gene_ids.csv"
    expression_path = args.cache_dir / "expression_float32.npy"
    condition_path = args.condition_root / "official_candidate_conditions.csv"

    library = pd.read_csv(library_path, low_memory=False)
    require_columns(
        library,
        ["compound_index", "canonical_smiles"],
        "screening library",
    )
    expected_indices = np.arange(len(library), dtype=np.int64)
    if not np.array_equal(
        library["compound_index"].to_numpy(dtype=np.int64), expected_indices
    ):
        raise RuntimeError("screening-library compound_index must be sequential")
    frozen = build_frozen_table(library)
    included = frozen.loc[frozen["include_gene_rerun"]].reset_index(drop=True)
    included_candidates = included["candidate"].tolist()

    gene_annotation = load_gene_table(input_gene_path)
    cache_gene_annotation = load_gene_table(cache_gene_path)
    if gene_annotation["gene_id"].tolist() != cache_gene_annotation["gene_id"].tolist():
        raise RuntimeError("Cache and disease-matrix gene IDs/order do not match")
    disease = np.load(disease_path).astype(np.float32, copy=False)
    observed_mask = np.load(observed_mask_path).astype(bool, copy=False)
    disease_manifest = pd.read_csv(disease_manifest_path)
    require_columns(disease_manifest, ["cancer"], "disease manifest")
    cancers = disease_manifest["cancer"].astype(str).tolist()
    expected_shape = (len(cancers), len(gene_annotation))
    if disease.shape != expected_shape or observed_mask.shape != expected_shape:
        raise RuntimeError(
            "Disease matrix/mask dimensions do not match cancers and gene table"
        )

    predicted_profiles, checkpoint_inventory, device_metadata = (
        predict_candidate_profiles(
            included,
            args.checkpoint_root,
            models,
            seeds,
            len(gene_annotation),
            args.disable_amp,
            args.allow_cpu,
        )
    )
    np.save(args.output_dir / "predicted_candidate_profiles_by_run.npy", predicted_profiles)

    expression = np.load(expression_path, mmap_mode="r")
    if expression.ndim != 2 or expression.shape[1] != len(gene_annotation):
        raise RuntimeError(
            f"Expression shape {expression.shape} is incompatible with gene table"
        )
    conditions = pd.read_csv(condition_path, low_memory=False)
    measured_profiles, measured_coverage = aggregate_measured_hiq_profiles(
        conditions,
        expression,
        included_candidates,
    )

    predicted_tables: list[pd.DataFrame] = []
    measured_tables: list[pd.DataFrame] = []
    consensus_tables: list[pd.DataFrame] = []
    role_lookup = frozen.set_index("candidate")
    for candidate_index, candidate in enumerate(included_candidates):
        predicted_summary = summarize_gene_evidence(
            predicted_profiles[candidate_index],
            disease,
            observed_mask,
            gene_annotation,
            "predicted",
        )
        predicted_summary.insert(0, "candidate", candidate)
        predicted_summary.attrs["candidate"] = candidate
        measured_summary = summarize_gene_evidence(
            measured_profiles[candidate],
            disease,
            observed_mask,
            gene_annotation,
            "measured",
        )
        measured_summary.insert(0, "candidate", candidate)
        measured_summary.attrs["candidate"] = candidate
        # build_candidate_consensus expects the candidate attribute after the
        # display column is removed from its one-candidate inputs.
        predicted_for_merge = predicted_summary.drop(columns="candidate")
        predicted_for_merge.attrs["candidate"] = candidate
        measured_for_merge = measured_summary.drop(columns="candidate")
        measured_for_merge.attrs["candidate"] = candidate
        consensus = build_candidate_consensus(
            predicted_for_merge,
            measured_for_merge,
            str(role_lookup.loc[candidate, "candidate_role"]),
            str(role_lookup.loc[candidate, "biology_scope"]),
            args.consensus_threshold,
            args.network_top_k,
        )
        predicted_tables.append(predicted_summary)
        measured_tables.append(measured_summary)
        consensus_tables.append(consensus)

    predicted_gene_summary = pd.concat(predicted_tables, ignore_index=True)
    measured_gene_summary = pd.concat(measured_tables, ignore_index=True)
    consensus = pd.concat(consensus_tables, ignore_index=True)
    network_gene_sets = consensus.loc[consensus["selected_network_gene"]].copy()
    network_gene_sets = network_gene_sets.sort_values(
        ["candidate", "consensus_rank"], kind="stable"
    )
    thresholds = sorted({0.60, 0.70, 0.80, args.consensus_threshold})
    threshold_sensitivity = sensitivity_counts(consensus, thresholds)

    frozen.to_csv(args.output_dir / "frozen_candidate_manifest.csv", index=False)
    checkpoint_inventory.to_csv(
        args.output_dir / "checkpoint_inventory.csv", index=False
    )
    measured_coverage.to_csv(
        args.output_dir / "measured_hiq_gene_coverage.csv", index=False
    )
    predicted_gene_summary.to_csv(
        args.output_dir / "predicted_reversal_gene_summary.csv", index=False
    )
    measured_gene_summary.to_csv(
        args.output_dir / "measured_hiq_reversal_gene_summary.csv", index=False
    )
    consensus.to_csv(
        args.output_dir / "candidate_reversal_gene_consensus.csv", index=False
    )
    network_gene_sets.to_csv(
        args.output_dir / "candidate_network_gene_sets.csv", index=False
    )
    threshold_sensitivity.to_csv(
        args.output_dir / "consensus_threshold_sensitivity.csv", index=False
    )
    gene_set_dir = args.output_dir / "gene_sets"
    gene_set_dir.mkdir(parents=True, exist_ok=True)
    for candidate in included_candidates:
        genes = network_gene_sets.loc[
            network_gene_sets["candidate"] == candidate, "gene_id"
        ].astype(str)
        (gene_set_dir / f"{slug(candidate)}_reversal_associated_genes.txt").write_text(
            "\n".join(genes) + ("\n" if len(genes) else ""),
            encoding="utf-8",
        )

    output_names = [
        "frozen_candidate_manifest.csv",
        "checkpoint_inventory.csv",
        "predicted_candidate_profiles_by_run.npy",
        "measured_hiq_gene_coverage.csv",
        "predicted_reversal_gene_summary.csv",
        "measured_hiq_reversal_gene_summary.csv",
        "candidate_reversal_gene_consensus.csv",
        "candidate_network_gene_sets.csv",
        "consensus_threshold_sensitivity.csv",
        "gene_sets/*_reversal_associated_genes.txt",
        "frozen_candidate_biology_manifest.json",
        "audit.log",
    ]
    manifest = {
        "analysis_stage": "frozen_candidate_biology_gene_rerun",
        "candidate_policy_version": "post_experiment_package_1_2026-07-18",
        "included_candidates": included_candidates,
        "primary_candidates": PRIMARY_CANDIDATES,
        "models": models,
        "seeds": seeds,
        "split": "hdac_class_holdout",
        "cancers": cancers,
        "gene_count": len(gene_annotation),
        "consensus_threshold": args.consensus_threshold,
        "threshold_sensitivity": thresholds,
        "network_top_k_per_candidate": args.network_top_k,
        "measured_quality_set": "official LINCS is_hiq rows",
        "measured_aggregation": (
            "Median across signatures within candidate/dose/time/official-cell "
            "condition, then median across conditions within official cell."
        ),
        "gene_reversal_definition": (
            "For disease signature d and drug profile p, gene contribution is "
            "-(p_g*d_g)/sum_observed_g(|d_g|); positive values indicate "
            "transcriptomic reversal."
        ),
        "selection_definition": (
            "A gene must meet the predeclared threshold for predicted reversal, "
            "predicted model-seed support, predicted cancer support, measured "
            "reversal, measured official-cell support, and measured cancer support; "
            "both mean signed contributions must be positive. Eligible genes are "
            "ranked by the mean percentile of eight predicted/measured evidence terms."
        ),
        "evidence_note": (
            "Outputs are reversal-associated genes for hypothesis generation. They "
            "are not direct drug targets, binding evidence, efficacy validation, or "
            "causal mechanism claims. Applying each official-cell drug profile to "
            "all cancer disease signatures is a computational cross-context analysis."
        ),
        "input_sha256": {
            "screening_library": sha256_file(library_path),
            "disease_matrix": sha256_file(disease_path),
            "disease_observed_mask": sha256_file(observed_mask_path),
            "disease_manifest": sha256_file(disease_manifest_path),
            "gene_ids": sha256_file(input_gene_path),
            "official_candidate_conditions": sha256_file(condition_path),
        },
        "expression_file": {
            "path": str(expression_path),
            "size_bytes": expression_path.stat().st_size,
            "upstream_hash_source": (
                "results/revision/measured_reversal/measured_reversal_manifest.json"
            ),
        },
        "upstream_provenance": upstream_provenance(),
        "device": device_metadata,
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": None,
        "determinism_note": "Evaluation and median/rank aggregation contain no random sampling.",
        "outputs": output_names,
    }
    manifest_path = args.output_dir / "frozen_candidate_biology_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    freeze_display = frozen[
        [
            "candidate",
            "candidate_role",
            "biology_scope",
            "include_gene_rerun",
            "evidence_status",
        ]
    ]
    observed_network_counts = (
        network_gene_sets.groupby(
            ["candidate", "candidate_role"], as_index=False
        )["gene_id"]
        .nunique()
        .rename(columns={"gene_id": "selected_network_genes"})
    )
    network_counts = included[
        ["candidate", "candidate_role"]
    ].merge(
        observed_network_counts,
        on=["candidate", "candidate_role"],
        how="left",
        validate="one_to_one",
    )
    network_counts["selected_network_genes"] = (
        network_counts["selected_network_genes"].fillna(0).astype(int)
    )
    sections = [
        "===== FROZEN CANDIDATE POLICY =====",
        freeze_display.to_string(index=False),
        "",
        "===== OFFICIAL HiQ GENE-RERUN COVERAGE =====",
        measured_coverage.to_string(index=False),
        "",
        "===== PREDECLARED CONSENSUS SENSITIVITY =====",
        threshold_sensitivity.to_string(index=False),
        "",
        "===== SELECTED REVERSAL-ASSOCIATED GENE SETS =====",
        network_counts.to_string(index=False),
        "",
        "FROZEN CANDIDATE BIOLOGY GENE RERUN: PASSED",
    ]
    audit_text = "\n".join(sections) + "\n"
    (args.output_dir / "audit.log").write_text(audit_text, encoding="utf-8")
    print(audit_text, end="")


if __name__ == "__main__":
    main()
