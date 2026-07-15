from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.revision_dataset import featurize_smiles, warm_molecule_cache  # noqa: E402
from src.revision_models import build_model  # noqa: E402


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen the compound library with revision checkpoints without storing "
            "large transcriptome prediction matrices."
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
        default=Path("results/revision/screening"),
    )
    parser.add_argument("--splits", default="pair_split")
    parser.add_argument("--models", default="dual_stream,fingerprint_mlp")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


class ScreeningDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"compound_index", "canonical_smiles"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Screening library lacks columns: {sorted(missing)}")
        expected = np.arange(len(frame), dtype=np.int64)
        actual = frame["compound_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(expected, actual):
            raise ValueError("compound_index must be sequential from zero")
        self.frame = frame.reset_index(drop=True)
        self.smiles = self.frame["canonical_smiles"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        features = featurize_smiles(self.smiles[index])
        return index, features.atom_features, features.fingerprint


def collate_screening(batch):
    indices, atom_arrays, fingerprints = zip(*batch)
    max_atoms = max(array.shape[0] for array in atom_arrays)
    feature_dim = atom_arrays[0].shape[1]
    atoms = torch.zeros((len(batch), max_atoms, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_atoms), dtype=torch.bool)
    for row, atom_features in enumerate(atom_arrays):
        count = atom_features.shape[0]
        atoms[row, :count] = atom_features
        mask[row, :count] = True
    return (
        np.asarray(indices, dtype=np.int64),
        atoms,
        mask,
        torch.stack(fingerprints),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_model(
    checkpoint_path: Path,
    model_name: str,
    gene_count: int,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arguments = checkpoint.get("arguments", {})
    hidden_dim = int(arguments.get("hidden_dim", 512))
    recorded_model = str(arguments.get("model", model_name))
    if recorded_model != model_name:
        raise ValueError(
            f"Checkpoint model mismatch: path says {model_name}, "
            f"checkpoint says {recorded_model}"
        )
    model = build_model(model_name, hidden_dim, gene_count).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, hidden_dim


def compute_scores(predictions: torch.Tensor, disease: torch.Tensor):
    disease_squared = disease.square()
    positive_weights = torch.where(
        disease > 0,
        disease_squared,
        torch.zeros_like(disease),
    )
    negative_weights = torch.where(
        disease < 0,
        disease_squared,
        torch.zeros_like(disease),
    )
    denominator_legacy = disease_squared.sum(dim=1).clamp_min(1e-12)

    legacy = (
        torch.relu(-predictions) @ positive_weights.T
        + torch.relu(predictions) @ negative_weights.T
    ) / denominator_legacy.unsqueeze(0)

    denominator_signed = disease.abs().sum(dim=1).clamp_min(1e-12)
    signed = -(predictions @ disease.T) / denominator_signed.unsqueeze(0)
    return legacy, signed


def top_candidates(
    library: pd.DataFrame,
    cancer_names: list[str],
    scores: np.ndarray,
    metric: str,
    top_k: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
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
    ]
    available = [column for column in columns if column in library.columns]
    for cancer_index, cancer in enumerate(cancer_names):
        order = np.argsort(-scores[:, cancer_index], kind="stable")[:top_k]
        frame = library.iloc[order][available].copy()
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "score", scores[order, cancer_index])
        frame.insert(0, "cancer", cancer)
        frame.insert(0, "metric", metric)
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
    gene_ids = pd.read_csv(args.input_dir / "gene_ids.csv", dtype=str)
    cancer_names = disease_manifest["cancer"].astype(str).tolist()

    if disease_matrix.shape != (len(cancer_names), len(gene_ids)):
        raise ValueError(
            f"Disease matrix shape {disease_matrix.shape} does not match "
            "the manifest and gene-ID table"
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
    disease = torch.from_numpy(
        disease_matrix.astype(np.float32, copy=False)
    ).to(device)
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
                legacy_scores = np.empty(
                    (len(dataset), len(cancer_names)),
                    dtype=np.float32,
                )
                signed_scores = np.empty_like(legacy_scores)
                started = time.perf_counter()

                progress = tqdm(
                    loader,
                    desc=f"screen {split}/{model_name}/seed{seed}",
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
                        legacy, signed = compute_scores(predictions.float(), disease)
                        legacy_scores[indices] = legacy.cpu().numpy()
                        signed_scores[indices] = signed.cpu().numpy()

                np.save(run_dir / "legacy_wtrs_scores.npy", legacy_scores)
                np.save(run_dir / "signed_wtrs_scores.npy", signed_scores)
                top = pd.concat(
                    [
                        top_candidates(
                            library,
                            cancer_names,
                            legacy_scores,
                            "legacy_wtrs",
                            args.top_k,
                        ),
                        top_candidates(
                            library,
                            cancer_names,
                            signed_scores,
                            "signed_wtrs",
                            args.top_k,
                        ),
                    ],
                    ignore_index=True,
                )
                top.to_csv(run_dir / "top_candidates.csv", index=False)

                manifest = {
                    "split": split,
                    "model": model_name,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                    "checkpoint_validation_mse": float(
                        checkpoint.get("validation_mse", np.nan)
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "screened_compounds": len(dataset),
                    "cancers": cancer_names,
                    "gene_count": len(gene_ids),
                    "device": str(device),
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "amp_enabled": amp_enabled,
                    "runtime_seconds": time.perf_counter() - started,
                    "legacy_wtrs_definition": (
                        "sum_g I(d_g*p_g<0)*|d_g|*|d_g*p_g| / sum_g d_g^2; "
                        "larger values indicate greater reversal-only weighted magnitude"
                    ),
                    "signed_wtrs_definition": (
                        "-sum_g d_g*p_g / sum_g |d_g|; larger values indicate "
                        "stronger signed reversal"
                    ),
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(manifest, indent=2))
                completed += 1

    print(f"Completed screening runs: {completed}")


if __name__ == "__main__":
    main()
