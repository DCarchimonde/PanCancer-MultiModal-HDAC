from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.revision_dataset import (  # noqa: E402
    CachedLINCSExpressionDataset,
    collate_cached_profiles,
    warm_molecule_cache,
)
from src.revision_models import build_model  # noqa: E402

GROUP_COLUMNS = {
    "pair_split": "drug_cell_pair",
    "leave_drug_out": "canonical_smiles",
    "leave_cell_line_out": "cell_id",
    "scaffold_split": "scaffold_group",
    "hdac_class_holdout": "canonical_smiles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one reproducible BMC revision benchmark run.")
    parser.add_argument("--split", choices=sorted(GROUP_COLUMNS), required=True)
    parser.add_argument("--model", choices=["dual_stream", "fingerprint_mlp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-root", type=Path, default=Path("data/splits/generalization"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/revision"))
    parser.add_argument("--output-root", type=Path, default=Path("results/revision/benchmarks"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_validation_split(train_csv: Path, grouping_column: str, seed: int, validation_size: float, output_dir: Path) -> tuple[Path, Path]:
    frame = pd.read_csv(train_csv, low_memory=False)
    if grouping_column not in frame.columns:
        raise ValueError(f"{train_csv} lacks validation grouping column {grouping_column}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_index, validation_index = next(splitter.split(frame, groups=frame[grouping_column]))
    fit = frame.iloc[train_index].copy()
    validation = frame.iloc[validation_index].copy()
    if set(fit[grouping_column].astype(str)) & set(validation[grouping_column].astype(str)):
        raise RuntimeError("Validation grouping overlap detected")
    split_dir = output_dir / "resolved_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    fit_path = split_dir / "fit.csv"
    validation_path = split_dir / "validation.csv"
    fit.to_csv(fit_path, index=False)
    validation.to_csv(validation_path, index=False)
    return fit_path, validation_path


def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def make_loader(dataset: CachedLINCSExpressionDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, drop_last: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_cached_profiles,
        generator=generator,
    )


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
    profile_output: Path | None = None,
    compute_rank_metrics: bool = False,
) -> dict[str, float]:
    model.eval()
    profile_rows: list[pd.DataFrame] = []
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0
    sum_y = 0.0
    sum_y_squared = 0.0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            sig_ids, atoms, atom_mask, fingerprints, targets = batch
            atoms = atoms.to(device, non_blocking=True)
            atom_mask = atom_mask.to(device, non_blocking=True)
            fingerprints = fingerprints.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(amp_enabled):
                predictions = model(atoms, atom_mask, fingerprints)

            residual = predictions.float() - targets.float()
            total_squared_error += float(torch.sum(residual.square()).item())
            total_absolute_error += float(torch.sum(residual.abs()).item())
            total_elements += targets.numel()
            sum_y += float(torch.sum(targets.float()).item())
            sum_y_squared += float(torch.sum(targets.float().square()).item())

            if compute_rank_metrics or profile_output is not None:
                y_true = targets.float().cpu().numpy()
                y_pred = predictions.float().cpu().numpy()
                pearson = rowwise_pearson(y_true, y_pred)
                spearman = rowwise_pearson(
                    rankdata(y_true, axis=1),
                    rankdata(y_pred, axis=1),
                )
                profile_rows.append(pd.DataFrame({
                    "sig_id": sig_ids,
                    "pearson": pearson,
                    "spearman": spearman,
                    "profile_mse": np.mean((y_true - y_pred) ** 2, axis=1),
                    "profile_mae": np.mean(np.abs(y_true - y_pred), axis=1),
                }))

    if total_elements == 0:
        raise RuntimeError("Evaluation processed zero elements")
    mse = total_squared_error / total_elements
    mae = total_absolute_error / total_elements
    total_variation = sum_y_squared - (sum_y * sum_y / total_elements)
    r2 = 1.0 - total_squared_error / total_variation if total_variation > 0 else math.nan
    metrics: dict[str, float] = {"mse": mse, "mae": mae, "r2": r2}

    if profile_rows:
        profiles = pd.concat(profile_rows, ignore_index=True)
        metrics["pearson_mean"] = float(profiles["pearson"].mean())
        metrics["pearson_sd"] = float(profiles["pearson"].std(ddof=1))
        metrics["spearman_mean"] = float(profiles["spearman"].mean())
        metrics["spearman_sd"] = float(profiles["spearman"].std(ddof=1))
        metrics["evaluated_profiles"] = int(len(profiles))
        if profile_output is not None:
            profiles.to_csv(profile_output, index=False)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    run_dir = args.output_root / args.split / args.model / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.split_root / args.split
    fit_csv, validation_csv = make_validation_split(
        split_dir / "train.csv",
        GROUP_COLUMNS[args.split],
        args.seed,
        args.validation_size,
        run_dir,
    )
    test_csv = split_dir / "test.csv"
    cache_metadata = args.cache_dir / "cache_metadata.csv"
    expression_npy = args.cache_dir / "expression_float32.npy"

    fit_dataset = CachedLINCSExpressionDataset(fit_csv, cache_metadata, expression_npy, args.max_train_samples)
    validation_dataset = CachedLINCSExpressionDataset(validation_csv, cache_metadata, expression_npy, args.max_test_samples)
    test_dataset = CachedLINCSExpressionDataset(test_csv, cache_metadata, expression_npy, args.max_test_samples)
    warm_molecule_cache(
        list(fit_dataset.frame["canonical_smiles"])
        + list(validation_dataset.frame["canonical_smiles"])
        + list(test_dataset.frame["canonical_smiles"])
    )

    fit_loader = make_loader(fit_dataset, args.batch_size, True, args.num_workers, args.seed, drop_last=True)
    validation_loader = make_loader(validation_dataset, args.batch_size, False, args.num_workers, args.seed, drop_last=False)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, args.seed, drop_last=False)

    output_dim = int(np.load(expression_npy, mmap_mode="r").shape[1])
    model = build_model(args.model, args.hidden_dim, output_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_validation_mse = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        observed = 0
        progress = tqdm(fit_loader, desc=f"{args.split}/{args.model}/seed{args.seed} epoch {epoch}")
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            _, atoms, atom_mask, fingerprints, targets = batch
            atoms = atoms.to(device, non_blocking=True)
            atom_mask = atom_mask.to(device, non_blocking=True)
            fingerprints = fingerprints.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                predictions = model(atoms, atom_mask, fingerprints)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            observed += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_mse = epoch_loss / max(observed, 1)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            amp_enabled,
            args.max_eval_batches,
        )
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_metrics["mse"],
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if validation_metrics["mse"] < best_validation_mse - 1e-6:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_mse": best_validation_mse,
                "arguments": vars(args),
            }, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        amp_enabled,
        args.max_eval_batches,
        profile_output=run_dir / "test_profile_metrics.csv",
        compute_rank_metrics=True,
    )

    runtime_seconds = time.perf_counter() - started
    summary = {
        "split": args.split,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "fit_profiles": len(fit_dataset),
        "validation_profiles": len(validation_dataset),
        "test_profiles": len(test_dataset),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "runtime_seconds": runtime_seconds,
        "amp_enabled": amp_enabled,
        **test_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
