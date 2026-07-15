from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.screen_revision_models import (  # noqa: E402
    ScreeningDataset,
    collate_screening,
    load_checkpoint_model,
)
from src.revision_dataset import warm_molecule_cache  # noqa: E402


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the original pan-cancer scoring rule: row-wise Spearman "
            "correlation between predicted perturbation and each disease signature, "
            "with more negative correlation interpreted as stronger reversal."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/screening_input"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results/revision/benchmarks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/revision/screening_spearman"),
    )
    parser.add_argument("--splits", default="pair_split")
    parser.add_argument("--models", default="dual_stream,fingerprint_mlp")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_masks(mask_matrix: np.ndarray) -> list[dict[str, object]]:
    grouped: defaultdict[bytes, list[int]] = defaultdict(list)
    mask_lookup: dict[bytes, np.ndarray] = {}
    for cancer_index, mask in enumerate(mask_matrix):
        key = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        grouped[key].append(cancer_index)
        mask_lookup[key] = mask.astype(bool, copy=False)
    return [
        {
            "mask": mask_lookup[key],
            "cancer_indices": indices,
        }
        for key, indices in grouped.items()
    ]


def prepare_disease_rank_groups(
    disease_matrix: np.ndarray,
    mask_matrix: np.ndarray,
) -> list[dict[str, object]]:
    groups = group_masks(mask_matrix)
    prepared: list[dict[str, object]] = []
    for group in groups:
        mask = group["mask"]
        cancer_indices = group["cancer_indices"]
        disease_subset = disease_matrix[np.asarray(cancer_indices)][:, mask]
        disease_ranks = rankdata(
            disease_subset,
            method="average",
            axis=1,
        ).astype(np.float64, copy=False)
        disease_centered = disease_ranks - disease_ranks.mean(axis=1, keepdims=True)
        disease_norm = np.sqrt(np.square(disease_centered).sum(axis=1))
        if np.any(disease_norm == 0):
            raise ValueError("At least one disease signature is constant after alignment")
        prepared.append({
            "mask": mask,
            "cancer_indices": cancer_indices,
            "disease_centered": disease_centered,
            "disease_norm": disease_norm,
            "gene_count": int(mask.sum()),
        })
    return prepared


def spearman_reversal_scores(
    predictions: np.ndarray,
    prepared_groups: list[dict[str, object]],
    cancer_count: int,
) -> np.ndarray:
    output = np.empty((predictions.shape[0], cancer_count), dtype=np.float32)
    for group in prepared_groups:
        mask = group["mask"]
        cancer_indices = group["cancer_indices"]
        prediction_subset = predictions[:, mask]
        prediction_ranks = rankdata(
            prediction_subset,
            method="average",
            axis=1,
        ).astype(np.float64, copy=False)
        prediction_centered = prediction_ranks - prediction_ranks.mean(axis=1, keepdims=True)
        prediction_norm = np.sqrt(np.square(prediction_centered).sum(axis=1))
        numerator = prediction_centered @ group["disease_centered"].T
        denominator = prediction_norm[:, None] * group["disease_norm"][None, :]
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        output[:, np.asarray(cancer_indices)] = (-correlations).astype(np.float32)
    return output


def top_candidates(
    library: pd.DataFrame,
    cancer_names: list[str],
    scores: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    columns = [
        "compound_index",
        "canonical_smiles",
        "display_name",
        "cache_name",
        "compoundinfo_name",
        "source_drug_ids",
        "target",
        "moa",
        "name_conflict",
        "is_hdac_annotated",
        "hdac_annotation_source",
    ]
    available = [column for column in columns if column in library.columns]
    rows: list[pd.DataFrame] = []
    for cancer_index, cancer in enumerate(cancer_names):
        order = np.argsort(-scores[:, cancer_index], kind="stable")[:top_k]
        frame = library.iloc[order][available].copy()
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "spearman_rho", -scores[order, cancer_index])
        frame.insert(0, "reversal_score", scores[order, cancer_index])
        frame.insert(0, "cancer", cancer)
        frame.insert(0, "metric", "spearman_reversal")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    splits = parse_csv_list(args.splits)
    models = parse_csv_list(args.models)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]

    library = pd.read_csv(args.input_dir / "screening_library.csv", low_memory=False)
    disease_manifest = pd.read_csv(args.input_dir / "disease_manifest.csv")
    disease_matrix = np.load(args.input_dir / "disease_matrix.npy")
    mask_path = args.input_dir / "disease_observed_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing {mask_path}. Run scripts/prepare_spearman_mask.py first."
        )
    mask_matrix = np.load(mask_path).astype(bool, copy=False)
    gene_ids = pd.read_csv(args.input_dir / "gene_ids.csv", dtype=str)
    cancer_names = disease_manifest["cancer"].astype(str).tolist()

    expected_shape = (len(cancer_names), len(gene_ids))
    if disease_matrix.shape != expected_shape:
        raise ValueError(f"Unexpected disease matrix shape: {disease_matrix.shape}")
    if mask_matrix.shape != expected_shape:
        raise ValueError(f"Unexpected observed-mask shape: {mask_matrix.shape}")

    prepared_groups = prepare_disease_rank_groups(disease_matrix, mask_matrix)
    print(
        "Spearman mask groups:",
        [
            {
                "cancers": [cancer_names[i] for i in group["cancer_indices"]],
                "genes": group["gene_count"],
            }
            for group in prepared_groups
        ],
    )

    dataset = ScreeningDataset(library)
    warm_molecule_cache(dataset.smiles)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_screening,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    args.output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    for split in splits:
        for model_name in models:
            for seed in seeds:
                checkpoint_path = (
                    args.checkpoint_root
                    / split
                    / model_name
                    / f"seed_{seed}"
                    / "best_model.pt"
                )
                if not checkpoint_path.exists():
                    print(f"[MISSING] {checkpoint_path}")
                    continue
                run_dir = args.output_root / split / model_name / f"seed_{seed}"
                manifest_path = run_dir / "screening_manifest.json"
                if manifest_path.exists():
                    print(f"[SKIP] {split} / {model_name} / seed {seed}")
                    completed += 1
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                model, checkpoint, hidden_dim = load_checkpoint_model(
                    checkpoint_path,
                    model_name,
                    len(gene_ids),
                    device,
                )
                scores = np.empty(
                    (len(dataset), len(cancer_names)),
                    dtype=np.float32,
                )
                started = time.perf_counter()

                progress = tqdm(
                    loader,
                    desc=f"spearman {split}/{model_name}/seed{seed}",
                )
                with torch.inference_mode():
                    for indices, atoms, atom_mask, fingerprints in progress:
                        atoms = atoms.to(device, non_blocking=True)
                        atom_mask = atom_mask.to(device, non_blocking=True)
                        fingerprints = fingerprints.to(device, non_blocking=True)
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=amp_enabled,
                        ):
                            predictions = model(atoms, atom_mask, fingerprints)
                        prediction_array = predictions.float().cpu().numpy()
                        scores[indices] = spearman_reversal_scores(
                            prediction_array,
                            prepared_groups,
                            len(cancer_names),
                        )

                score_path = run_dir / "spearman_reversal_scores.npy"
                np.save(score_path, scores)
                top_candidates(
                    library,
                    cancer_names,
                    scores,
                    args.top_k,
                ).to_csv(run_dir / "top_candidates.csv", index=False)

                manifest = {
                    "split": split,
                    "model": model_name,
                    "seed": seed,
                    "metric": "spearman_reversal",
                    "hidden_dim": hidden_dim,
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                    "checkpoint_validation_mse": float(
                        checkpoint.get("validation_mse", np.nan)
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "score_sha256": sha256_file(score_path),
                    "observed_mask_sha256": sha256_file(mask_path),
                    "screened_compounds": len(dataset),
                    "cancers": cancer_names,
                    "gene_count": len(gene_ids),
                    "unique_observed_gene_masks": len(prepared_groups),
                    "device": str(device),
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "amp_enabled": amp_enabled,
                    "runtime_seconds": time.perf_counter() - started,
                    "definition": (
                        "Negative row-wise Spearman correlation between predicted drug "
                        "perturbation and the observed genes of each disease signature; "
                        "larger reversal_score values indicate stronger anticorrelation."
                    ),
                    "original_method_match": (
                        "Matches the original pandas DataFrame.corrwith(..., axis=1, "
                        "method='spearman') scoring and ascending-rho ranking rule."
                    ),
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(manifest, indent=2))
                completed += 1

    print(f"Completed Spearman screening runs: {completed}")


if __name__ == "__main__":
    main()
