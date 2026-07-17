from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lincs_candidate_validation import (  # noqa: E402
    assign_candidates,
    candidate_maps,
    resolve_candidates,
)


DEFAULT_CACHE_METADATA = Path("cache/revision/cache_metadata.csv")
DEFAULT_SCREENING_LIBRARY = Path("data/screening_input/screening_library.csv")
DEFAULT_OUTPUT_DIR = Path("results/revision/lincs_condition_audit")

EXPECTED_CANDIDATE_COUNTS = {
    "Mocetinostat": 14,
    "TC-H-106": 6,
    "NCH-51": 17,
    "Belinostat": 24,
    "PCI-24781": 19,
    "Panobinostat": 126,
    "Entinostat": 101,
    "Vorinostat": 899,
    "RG2833": 0,
    "Tianeptinaline_or_BG-1010": 6,
}

PREFERRED_SIGINFO_COLUMNS = [
    "sig_id",
    "pert_id",
    "pert_iname",
    "cmap_name",
    "pert_type",
    "cell_iname",
    "cell_mfc_name",
    "pert_dose",
    "pert_dose_unit",
    "pert_time",
    "pert_time_unit",
    "nsample",
    "distil_ids",
    "tas",
    "is_hiq",
    "qc_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match the revision expression cache and candidate signatures to "
            "official LINCS siginfo condition metadata."
        )
    )
    parser.add_argument(
        "--siginfo",
        type=Path,
        default=None,
        help=(
            "Path to official siginfo_beta.txt. If omitted, common AutoDL "
            "locations and LINCS_SIGINFO are checked."
        ),
    )
    parser.add_argument(
        "--cache-metadata",
        type=Path,
        default=DEFAULT_CACHE_METADATA,
    )
    parser.add_argument(
        "--screening-library",
        type=Path,
        default=DEFAULT_SCREENING_LIBRARY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def locate_siginfo(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get("LINCS_SIGINFO", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/root/autodl-tmp/pancancer_data/siginfo_beta.txt"),
            REPO_ROOT / "data" / "raw" / "siginfo_beta.txt",
            REPO_ROOT / "data" / "siginfo_beta.txt",
            REPO_ROOT / "siginfo_beta.txt",
            REPO_ROOT.parent / "pancancer_data" / "siginfo_beta.txt",
        ]
    )
    for path in candidates:
        expanded = path.expanduser().resolve()
        if expanded.is_file():
            return expanded
    checked = "\n".join(f"  - {path.expanduser()}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate official siginfo_beta.txt. Checked:\n"
        f"{checked}\n"
        "Rerun with --siginfo /absolute/path/to/siginfo_beta.txt"
    )


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def scan_siginfo(
    path: Path,
    wanted_ids: set[str],
    chunksize: int,
) -> tuple[pd.DataFrame, list[str], int]:
    header = pd.read_csv(path, sep="\t", nrows=0)
    source_columns = header.columns.astype(str).tolist()
    if "sig_id" not in source_columns:
        raise RuntimeError(f"Official siginfo lacks sig_id: {path}")
    usecols = [
        column for column in PREFERRED_SIGINFO_COLUMNS if column in source_columns
    ]

    matches: list[pd.DataFrame] = []
    rows_scanned = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        selected = chunk.loc[chunk["sig_id"].astype(str).isin(wanted_ids)]
        if not selected.empty:
            matches.append(selected.copy())

    if not matches:
        raise RuntimeError("No cache signatures matched official siginfo")
    matched = pd.concat(matches, ignore_index=True)
    matched["sig_id"] = matched["sig_id"].astype(str)
    return matched, source_columns, rows_scanned


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def classify_identifier_difference(
    row: pd.Series,
    cache_column: str,
    official_columns: list[str],
) -> tuple[str, str, str]:
    cache_value = str(row.get(cache_column, "")).strip()
    available = {
        column: str(row.get(column, "")).strip()
        for column in official_columns
        if pd.notna(row.get(column, pd.NA))
        and str(row.get(column, "")).strip()
    }
    exact_fields = [
        column for column, value in available.items() if value == cache_value
    ]
    normalized_cache = normalize_identifier(cache_value)
    normalized_fields = [
        column
        for column, value in available.items()
        if normalized_cache
        and normalize_identifier(value) == normalized_cache
    ]
    if exact_fields:
        classification = "matches_official_alternate_field"
        matched_fields = " | ".join(exact_fields)
    elif normalized_fields:
        classification = "normalized_official_alias_match"
        matched_fields = " | ".join(normalized_fields)
    else:
        classification = "unresolved_identifier_difference"
        matched_fields = ""
    alias_values = " | ".join(
        f"{column}={value}" for column, value in available.items()
    )
    return classification, matched_fields, alias_values


def join_unique(series: pd.Series) -> str:
    values = {
        str(value).strip()
        for value in series
        if pd.notna(value) and str(value).strip() and str(value).lower() != "nan"
    }
    return " | ".join(sorted(values))


def join_value_unit(
    frame: pd.DataFrame,
    value_column: str,
    unit_column: str,
) -> str:
    if value_column not in frame.columns:
        return ""
    values: set[str] = set()
    units = (
        clean_text(frame[unit_column])
        if unit_column in frame.columns
        else pd.Series("", index=frame.index)
    )
    for value, unit in zip(frame[value_column], units):
        if pd.isna(value) or not str(value).strip():
            continue
        value_text = str(value).strip()
        if value_text.endswith(".0"):
            value_text = value_text[:-2]
        values.add(f"{value_text} {unit}".strip())
    return " | ".join(sorted(values))


def true_count(series: pd.Series) -> int:
    return int(
        clean_text(series)
        .str.lower()
        .isin({"1", "1.0", "true", "yes", "y"})
        .sum()
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    siginfo_path = locate_siginfo(args.siginfo)

    cache = pd.read_csv(args.cache_metadata, low_memory=False)
    require_columns(
        cache,
        ["cache_row", "sig_id", "drug_id", "cell_id", "canonical_smiles"],
        "cache metadata",
    )
    if cache["sig_id"].duplicated().any():
        raise RuntimeError("cache_metadata.csv contains duplicate sig_id values")
    cache["sig_id"] = cache["sig_id"].astype(str)

    library = pd.read_csv(args.screening_library, low_memory=False)
    resolved = resolve_candidates(library)
    drug_map, smiles_map = candidate_maps(resolved)
    cache["candidate"] = assign_candidates(cache, drug_map, smiles_map)

    observed_counts = (
        cache.dropna(subset=["candidate"])
        .groupby("candidate")["sig_id"]
        .nunique()
        .to_dict()
    )
    observed_counts = {
        candidate: int(observed_counts.get(candidate, 0))
        for candidate in EXPECTED_CANDIDATE_COUNTS
    }
    if observed_counts != EXPECTED_CANDIDATE_COUNTS:
        raise RuntimeError(
            "Candidate signature counts changed unexpectedly: "
            f"expected={EXPECTED_CANDIDATE_COUNTS}, observed={observed_counts}"
        )

    wanted_ids = set(cache["sig_id"])
    official, source_columns, rows_scanned = scan_siginfo(
        siginfo_path, wanted_ids, args.chunksize
    )
    duplicate_official = int(official["sig_id"].duplicated().sum())
    if duplicate_official:
        raise RuntimeError(
            f"Official siginfo has {duplicate_official} duplicate matched sig_id rows"
        )

    matched_ids = set(official["sig_id"])
    missing_ids = wanted_ids - matched_ids
    extra_ids = matched_ids - wanted_ids
    if missing_ids or extra_ids:
        raise RuntimeError(
            "Official siginfo/cache mismatch: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    official = official.rename(
        columns={
            column: f"official_{column}"
            for column in official.columns
            if column != "sig_id"
        }
    )
    cache_columns = [
        column
        for column in [
            "cache_row",
            "sig_id",
            "candidate",
            "drug_id",
            "cell_id",
            "canonical_smiles",
            "tas",
        ]
        if column in cache.columns
    ]
    cache_export = cache[cache_columns].rename(
        columns={
            "drug_id": "cache_drug_id",
            "cell_id": "cache_cell_id",
            "canonical_smiles": "cache_canonical_smiles",
            "tas": "cache_tas",
        }
    )
    merged = cache_export.merge(
        official,
        on="sig_id",
        how="left",
        validate="one_to_one",
    )

    discrepancy_frames: list[pd.DataFrame] = []
    comparisons = [
        {
            "cache_column": "cache_drug_id",
            "primary_column": "official_pert_id",
            "official_columns": [
                "official_pert_id",
                "official_pert_iname",
                "official_cmap_name",
            ],
            "issue": "drug_id_vs_official_identity",
        },
        {
            "cache_column": "cache_cell_id",
            "primary_column": "official_cell_iname",
            "official_columns": [
                "official_cell_iname",
                "official_cell_mfc_name",
            ],
            "issue": "cell_id_vs_official_identity",
        },
    ]
    for comparison in comparisons:
        cache_column = comparison["cache_column"]
        primary_column = comparison["primary_column"]
        official_columns = [
            column
            for column in comparison["official_columns"]
            if column in merged.columns
        ]
        if cache_column not in merged.columns or primary_column not in merged.columns:
            continue
        left = clean_text(merged[cache_column])
        primary = clean_text(merged[primary_column])
        mask = left.ne("") & primary.ne("") & left.ne(primary)
        if not mask.any():
            continue
        selected_columns = list(
            dict.fromkeys(
                [
                    "sig_id",
                    "candidate",
                    cache_column,
                    primary_column,
                    *official_columns,
                ]
            )
        )
        discrepancy = merged.loc[mask, selected_columns].copy()
        classifications = discrepancy.apply(
            lambda row: classify_identifier_difference(
                row,
                cache_column,
                official_columns,
            ),
            axis=1,
            result_type="expand",
        )
        classifications.columns = [
            "classification",
            "matched_official_fields",
            "official_alias_values",
        ]
        discrepancy = pd.concat(
            [
                discrepancy.reset_index(drop=True),
                classifications.reset_index(drop=True),
            ],
            axis=1,
        )
        discrepancy.insert(0, "issue", comparison["issue"])
        discrepancy = discrepancy.rename(
            columns={
                cache_column: "cache_value",
                primary_column: "official_primary_value",
            }
        )
        keep_columns = [
            "issue",
            "classification",
            "sig_id",
            "candidate",
            "cache_value",
            "official_primary_value",
            "matched_official_fields",
            "official_alias_values",
        ]
        discrepancy_frames.append(discrepancy[keep_columns])

    discrepancy_columns = [
        "issue",
        "classification",
        "sig_id",
        "candidate",
        "cache_value",
        "official_primary_value",
        "matched_official_fields",
        "official_alias_values",
    ]
    discrepancies = (
        pd.concat(discrepancy_frames, ignore_index=True)
        if discrepancy_frames
        else pd.DataFrame(columns=discrepancy_columns)
    )
    discrepancy_summary = (
        discrepancies.groupby(["issue", "classification"], as_index=False)
        .agg(
            rows=("sig_id", "size"),
            unique_signatures=("sig_id", "nunique"),
            candidates_affected=("candidate", "nunique"),
        )
        if not discrepancies.empty
        else pd.DataFrame(
            columns=[
                "issue",
                "classification",
                "rows",
                "unique_signatures",
                "candidates_affected",
            ]
        )
    )
    candidate_discrepancy_summary = discrepancies.copy()
    if not candidate_discrepancy_summary.empty:
        candidate_discrepancy_summary["candidate"] = (
            candidate_discrepancy_summary["candidate"]
            .fillna("NON_CANDIDATE_CACHE_ROWS")
            .astype(str)
        )
        candidate_discrepancy_summary = (
            candidate_discrepancy_summary.groupby(
                ["candidate", "issue", "classification"], as_index=False
            )
            .agg(
                rows=("sig_id", "size"),
                unique_signatures=("sig_id", "nunique"),
                unique_cache_values=("cache_value", "nunique"),
            )
        )
    else:
        candidate_discrepancy_summary = pd.DataFrame(
            columns=[
                "candidate",
                "issue",
                "classification",
                "rows",
                "unique_signatures",
                "unique_cache_values",
            ]
        )
    unresolved_identifier_differences = int(
        discrepancies["classification"]
        .eq("unresolved_identifier_difference")
        .sum()
    )

    candidate_conditions = merged.dropna(subset=["candidate"]).copy()
    condition_fields = [
        "official_pert_id",
        "official_pert_type",
        "official_cell_iname",
        "official_pert_dose",
        "official_pert_dose_unit",
        "official_pert_time",
        "official_pert_time_unit",
        "official_nsample",
        "official_tas",
        "official_is_hiq",
        "official_qc_pass",
    ]
    completeness_rows: list[dict[str, object]] = []
    for field in condition_fields:
        if field not in candidate_conditions.columns:
            completeness_rows.append(
                {
                    "field": field,
                    "available_in_source": False,
                    "complete_rows": 0,
                    "missing_rows": len(candidate_conditions),
                    "complete_fraction": 0.0,
                }
            )
            continue
        present = clean_text(candidate_conditions[field]).ne("")
        completeness_rows.append(
            {
                "field": field,
                "available_in_source": True,
                "complete_rows": int(present.sum()),
                "missing_rows": int((~present).sum()),
                "complete_fraction": float(present.mean()),
            }
        )
    completeness = pd.DataFrame(completeness_rows)

    summary_rows: list[dict[str, object]] = []
    for resolved_row in resolved.itertuples(index=False):
        subset = candidate_conditions.loc[
            candidate_conditions["candidate"] == resolved_row.candidate
        ]
        name_columns = [
            column
            for column in ["official_pert_iname", "official_cmap_name"]
            if column in subset.columns
        ]
        names = " | ".join(
            value
            for value in [join_unique(subset[column]) for column in name_columns]
            if value
        )
        tas_column = next(
            (
                column
                for column in ["official_tas", "cache_tas"]
                if column in subset.columns
            ),
            None,
        )
        tas = (
            pd.to_numeric(subset[tas_column], errors="coerce")
            if tas_column is not None
            else pd.Series(dtype=float)
        )
        nsample = (
            pd.to_numeric(subset["official_nsample"], errors="coerce")
            if "official_nsample" in subset.columns
            else pd.Series(dtype=float)
        )
        if len(subset) == 0:
            status = "no_measured_LINCS_signature"
        elif bool(resolved_row.name_conflict):
            status = "identity_conflict_exclude_from_primary_evidence"
        else:
            status = "official_measured_conditions_available"
        summary_rows.append(
            {
                "candidate": resolved_row.candidate,
                "signatures": int(subset["sig_id"].nunique()),
                "cell_lines": (
                    int(subset["official_cell_iname"].nunique(dropna=True))
                    if "official_cell_iname" in subset.columns
                    else 0
                ),
                "official_names": names,
                "pert_types": (
                    join_unique(subset["official_pert_type"])
                    if "official_pert_type" in subset.columns
                    else ""
                ),
                "times": join_value_unit(
                    subset, "official_pert_time", "official_pert_time_unit"
                ),
                "doses": join_value_unit(
                    subset, "official_pert_dose", "official_pert_dose_unit"
                ),
                "tas_median": float(tas.median()) if tas.notna().any() else None,
                "nsample_median": (
                    float(nsample.median()) if nsample.notna().any() else None
                ),
                "hiq_rows": (
                    true_count(subset["official_is_hiq"])
                    if "official_is_hiq" in subset.columns
                    else None
                ),
                "qc_pass_rows": (
                    true_count(subset["official_qc_pass"])
                    if "official_qc_pass" in subset.columns
                    else None
                ),
                "name_conflict": bool(resolved_row.name_conflict),
                "metadata_status": status,
            }
        )
    candidate_summary = pd.DataFrame(summary_rows)

    coverage = pd.DataFrame(
        [
            {
                "cache_signatures": len(cache),
                "cache_unique_signatures": cache["sig_id"].nunique(),
                "siginfo_rows_scanned": rows_scanned,
                "official_matched_signatures": len(official),
                "unmatched_cache_signatures": len(missing_ids),
                "duplicate_official_signatures": duplicate_official,
                "identifier_primary_differences": len(discrepancies),
                "identifier_unresolved_differences": (
                    unresolved_identifier_differences
                ),
                "candidate_signatures": len(candidate_conditions),
            }
        ]
    )

    coverage.to_csv(args.output_dir / "cache_siginfo_coverage.csv", index=False)
    pd.DataFrame({"source_column": source_columns}).to_csv(
        args.output_dir / "siginfo_source_columns.csv", index=False
    )
    candidate_conditions.to_csv(
        args.output_dir / "official_candidate_conditions.csv", index=False
    )
    candidate_summary.to_csv(
        args.output_dir / "candidate_condition_summary.csv", index=False
    )
    completeness.to_csv(
        args.output_dir / "candidate_metadata_completeness.csv", index=False
    )
    discrepancies.to_csv(
        args.output_dir / "metadata_discrepancies.csv", index=False
    )
    discrepancy_summary.to_csv(
        args.output_dir / "metadata_discrepancy_summary.csv", index=False
    )
    candidate_discrepancy_summary.to_csv(
        args.output_dir / "candidate_metadata_discrepancy_summary.csv",
        index=False,
    )

    manifest = {
        "siginfo_path": str(siginfo_path),
        "siginfo_size_bytes": siginfo_path.stat().st_size,
        "cache_metadata": str(args.cache_metadata.resolve()),
        "screening_library": str(args.screening_library.resolve()),
        "cache_signatures": len(cache),
        "official_matched_signatures": len(official),
        "candidate_signatures": len(candidate_conditions),
        "candidate_count": len(candidate_summary),
        "candidates_with_measured_signatures": int(
            candidate_summary["signatures"].gt(0).sum()
        ),
        "identifier_primary_differences": len(discrepancies),
        "identifier_unresolved_differences": (
            unresolved_identifier_differences
        ),
        "identifier_audit_status": (
            "official_metadata_authoritative_differences_recorded"
            if len(discrepancies)
            else "exact_primary_identifier_match"
        ),
        "identifier_join_key": "sig_id",
        "condition_metadata_note": (
            "Dose, time, cell line, perturbation type, and QC fields are taken "
            "from official siginfo rather than parsed from sig_id."
        ),
        "identifier_difference_note": (
            "Every cache signature is joined one-to-one by official sig_id. "
            "Differences between cache labels and official primary identifiers "
            "are retained as diagnostics; downstream condition analyses use "
            "official identifiers and condition fields."
        ),
    }
    (args.output_dir / "condition_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_columns = [
        "candidate",
        "signatures",
        "cell_lines",
        "pert_types",
        "times",
        "doses",
        "tas_median",
        "hiq_rows",
        "qc_pass_rows",
        "name_conflict",
        "metadata_status",
    ]
    report = "\n".join(
        [
            "===== OFFICIAL SIGINFO SOURCE =====",
            json.dumps(
                {
                    "path": str(siginfo_path),
                    "rows_scanned": rows_scanned,
                    "source_columns": len(source_columns),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "",
            "===== CACHE / OFFICIAL COVERAGE =====",
            coverage.to_string(index=False),
            "",
            "===== IDENTIFIER DIFFERENCE CLASSIFICATION =====",
            (
                discrepancy_summary.to_string(index=False)
                if not discrepancy_summary.empty
                else "No cache/official primary-identifier differences."
            ),
            "",
            "===== CANDIDATE-AFFECTED IDENTIFIER DIFFERENCES =====",
            (
                candidate_discrepancy_summary.loc[
                    candidate_discrepancy_summary["candidate"]
                    != "NON_CANDIDATE_CACHE_ROWS"
                ].to_string(index=False)
                if (
                    not candidate_discrepancy_summary.empty
                    and candidate_discrepancy_summary["candidate"]
                    .ne("NON_CANDIDATE_CACHE_ROWS")
                    .any()
                )
                else "No candidate signatures have identifier differences."
            ),
            "",
            "===== CANDIDATE OFFICIAL CONDITIONS =====",
            candidate_summary[report_columns].round(6).to_string(index=False),
            "",
            "===== CONDITION METADATA COMPLETENESS =====",
            completeness.round(6).to_string(index=False),
            "",
            "===== MANIFEST =====",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "",
            "OFFICIAL LINCS CANDIDATE CONDITION AUDIT: PASSED",
        ]
    )
    (args.output_dir / "audit.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
