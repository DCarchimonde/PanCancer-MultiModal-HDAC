from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import (  # noqa: E402
    MODELS,
    assign_candidates,
    candidate_maps,
    mean_ci,
    resolve_candidates,
)


SCREENING_LIBRARY = Path("data/screening_input/screening_library.csv")
SPLIT_ROOT = Path("data/splits/generalization/leave_drug_out")
BENCHMARK_ROOT = Path("results/revision/benchmarks/leave_drug_out")
OUTPUT_DIR = Path("results/revision/leave_drug_candidate_validation")

SEEDS = [1, 2, 3]
METRICS = ["pearson", "spearman", "profile_mse", "profile_mae"]
EXPECTED_STRICT_CANDIDATES = {"Mocetinostat", "PCI-24781"}


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def candidate_membership(
    resolved: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in resolved["candidate"]:
        train_subset = train.loc[train["candidate"] == candidate]
        test_subset = test.loc[test["candidate"] == candidate]
        train_count = int(train_subset["sig_id"].nunique())
        test_count = int(test_subset["sig_id"].nunique())
        rows.append(
            {
                "candidate": candidate,
                "train_signatures": train_count,
                "test_signatures": test_count,
                "test_cell_lines": (
                    int(test_subset["cell_id"].astype(str).nunique())
                    if test_count
                    else 0
                ),
                "strict_leave_drug_out": train_count == 0 and test_count > 0,
            }
        )
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def summarize_seed_metrics(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, model), subset in by_seed.groupby(
        ["candidate", "model"], sort=True
    ):
        row: dict[str, object] = {
            "candidate": candidate,
            "model": model,
            "seeds": int(subset["seed"].nunique()),
            "signatures": int(subset["signatures"].max()),
            "cell_lines": int(subset["cell_lines"].max()),
        }
        for metric in METRICS:
            mean, seed_sd, ci_low, ci_high = mean_ci(subset[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sd"] = seed_sd
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_model_comparison(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, candidate_rows in by_seed.groupby("candidate", sort=True):
        for metric in METRICS:
            pivot = candidate_rows.pivot(
                index="seed", columns="model", values=metric
            ).sort_index()
            if list(pivot.index) != SEEDS or set(pivot.columns) != set(MODELS):
                raise RuntimeError(
                    f"Incomplete model/seed pairing for {candidate}, {metric}"
                )

            if metric in {"pearson", "spearman"}:
                advantage = pivot["dual_stream"] - pivot["fingerprint_mlp"]
            else:
                # Lower error is better; this orientation keeps positive values
                # consistently in favor of the dual-stream model.
                advantage = pivot["fingerprint_mlp"] - pivot["dual_stream"]

            mean, seed_sd, ci_low, ci_high = mean_ci(advantage)
            rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "seeds": int(len(advantage)),
                    "advantage_dual_mean": mean,
                    "advantage_dual_seed_sd": seed_sd,
                    "advantage_dual_ci95_low": ci_low,
                    "advantage_dual_ci95_high": ci_high,
                    "interpretation": "positive values favor dual_stream",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    library = pd.read_csv(SCREENING_LIBRARY, low_memory=False)
    resolved = resolve_candidates(library)
    drug_map, smiles_map = candidate_maps(resolved)

    train = pd.read_csv(SPLIT_ROOT / "train.csv", low_memory=False)
    test = pd.read_csv(SPLIT_ROOT / "test.csv", low_memory=False)
    required_split_columns = [
        "sig_id",
        "drug_id",
        "canonical_smiles",
        "cell_id",
    ]
    require_columns(train, required_split_columns, "leave-drug train split")
    require_columns(test, required_split_columns, "leave-drug test split")

    if train["sig_id"].duplicated().any() or test["sig_id"].duplicated().any():
        raise RuntimeError("The leave-drug split contains duplicate sig_id values")

    # assign_candidates returns a Series; retain the split DataFrames and attach it.
    train["candidate"] = assign_candidates(train, drug_map, smiles_map)
    test["candidate"] = assign_candidates(test, drug_map, smiles_map)

    membership = candidate_membership(resolved, train, test)
    observed_strict = set(
        membership.loc[membership["strict_leave_drug_out"], "candidate"]
    )
    if observed_strict != EXPECTED_STRICT_CANDIDATES:
        raise RuntimeError(
            "Unexpected strict leave-drug candidate set: "
            f"expected {sorted(EXPECTED_STRICT_CANDIDATES)}, "
            f"observed {sorted(observed_strict)}"
        )

    strict_test = test.loc[
        test["candidate"].isin(observed_strict),
        ["sig_id", "drug_id", "cell_id", "candidate"],
    ].copy()
    strict_test["sig_id"] = strict_test["sig_id"].astype(str)
    all_test_signatures = set(test["sig_id"].astype(str))

    seed_tables: list[pd.DataFrame] = []
    for model in MODELS:
        for seed in SEEDS:
            metrics_path = (
                BENCHMARK_ROOT
                / model
                / f"seed_{seed}"
                / "test_profile_metrics.csv"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)

            metrics = pd.read_csv(metrics_path, low_memory=False)
            require_columns(metrics, ["sig_id", *METRICS], str(metrics_path))
            metrics["sig_id"] = metrics["sig_id"].astype(str)
            if metrics["sig_id"].duplicated().any():
                raise RuntimeError(f"Duplicate sig_id values in {metrics_path}")

            metric_signatures = set(metrics["sig_id"])
            if metric_signatures != all_test_signatures:
                raise RuntimeError(
                    f"Incomplete or mismatched profiles in {metrics_path}: "
                    f"missing={len(all_test_signatures - metric_signatures)}, "
                    f"extra={len(metric_signatures - all_test_signatures)}"
                )

            merged = strict_test.merge(
                metrics[["sig_id", *METRICS]],
                on="sig_id",
                how="left",
                validate="one_to_one",
            )
            if merged[METRICS].isna().any().any():
                raise RuntimeError(f"Missing candidate metrics in {metrics_path}")

            grouped = merged.groupby("candidate", as_index=False).agg(
                signatures=("sig_id", "nunique"),
                cell_lines=("cell_id", "nunique"),
                pearson=("pearson", "mean"),
                spearman=("spearman", "mean"),
                profile_mse=("profile_mse", "mean"),
                profile_mae=("profile_mae", "mean"),
            )
            grouped.insert(1, "model", model)
            grouped.insert(2, "seed", seed)
            seed_tables.append(grouped)

    by_seed = pd.concat(seed_tables, ignore_index=True)
    expected_rows = len(observed_strict) * len(MODELS) * len(SEEDS)
    if len(by_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} candidate/model/seed rows, found {len(by_seed)}"
        )

    summary = summarize_seed_metrics(by_seed)
    paired = paired_model_comparison(by_seed)

    membership.to_csv(OUTPUT_DIR / "candidate_leave_drug_membership.csv", index=False)
    by_seed.to_csv(
        OUTPUT_DIR / "leave_drug_candidate_metrics_by_seed.csv", index=False
    )
    summary.to_csv(
        OUTPUT_DIR / "leave_drug_candidate_metrics_summary.csv", index=False
    )
    paired.to_csv(
        OUTPUT_DIR / "leave_drug_candidate_paired_comparison.csv", index=False
    )

    manifest = {
        "split": "leave_drug_out",
        "models": MODELS,
        "seeds": SEEDS,
        "complete_test_signatures_per_run": len(all_test_signatures),
        "strict_candidates": sorted(observed_strict),
        "strict_candidate_count": len(observed_strict),
        "strict_test_signatures": int(strict_test["sig_id"].nunique()),
        "strict_test_cell_lines": int(strict_test["cell_id"].nunique()),
        "comparison_orientation": (
            "positive advantage_dual values favor dual_stream; correlations "
            "use dual-fingerprint and errors use fingerprint-dual"
        ),
        "outputs": [
            "candidate_leave_drug_membership.csv",
            "leave_drug_candidate_metrics_by_seed.csv",
            "leave_drug_candidate_metrics_summary.csv",
            "leave_drug_candidate_paired_comparison.csv",
            "leave_drug_candidate_validation_manifest.json",
            "audit.log",
        ],
    }
    (OUTPUT_DIR / "leave_drug_candidate_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    membership_view = membership[
        [
            "candidate",
            "train_signatures",
            "test_signatures",
            "test_cell_lines",
            "strict_leave_drug_out",
        ]
    ].to_string(index=False)
    summary_view = summary[
        [
            "candidate",
            "model",
            "seeds",
            "signatures",
            "cell_lines",
            "pearson_mean",
            "pearson_seed_sd",
            "spearman_mean",
            "spearman_seed_sd",
            "profile_mse_mean",
            "profile_mae_mean",
        ]
    ].round(6).to_string(index=False)
    paired_view = paired.round(6).to_string(index=False)

    report = "\n".join(
        [
            "===== STRICT LEAVE-DRUG MEMBERSHIP =====",
            membership_view,
            "",
            "===== STRICT CANDIDATE SUMMARY =====",
            summary_view,
            "",
            "===== PAIRED DUAL-STREAM ADVANTAGE =====",
            paired_view,
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "STRICT LEAVE-DRUG CANDIDATE VALIDATION: PASSED",
        ]
    )
    (OUTPUT_DIR / "audit.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
