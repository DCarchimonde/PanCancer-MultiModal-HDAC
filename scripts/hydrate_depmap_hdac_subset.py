#!/usr/bin/env python3
"""Export a small, auditable HDAC subset from an exact DepMap release.

The preferred automated path obtains the official DepMap bulk-download
manifest, selects an exact release and exact source files, and verifies source
MD5 hashes when supplied.  A local-input path is also supported for files
downloaded interactively from the exact release page when the portal blocks
non-browser downloads.  Both paths export only the requested HDAC gene-effect
columns plus matching model metadata; signed URLs and raw bulk files are never
persisted in revision outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUEST = Path("revision/requests/depmap_26q1_hdac_subset.json")
DEFAULT_OUTPUT = Path("results/revision/depmap_hdac_background/source_subset")
USER_AGENT = (
    "PanCancer-MultiModal-HDAC-major-revision/1.0 "
    "(+https://github.com/DCarchimonde/PanCancer-MultiModal-HDAC)"
)
MODEL_ID_ALIASES = {"modelid", "depmapid", "achillesid"}
LINEAGE_ALIASES = {
    "oncotreelineage",
    "oncotreeprimarydisease",
    "lineage",
    "primarydisease",
}
METADATA_COLUMNS = {
    "modelid",
    "depmapid",
    "achillesid",
    "patientid",
    "celllinename",
    "strippedcelllinename",
    "cclelname",
    "cclename",
    "depmapmodeltype",
    "modeltype",
    "oncotreelineage",
    "oncotreeprimarydisease",
    "oncotreesubtype",
    "oncotreecode",
    "lineage",
    "primarydisease",
    "subtype",
    "primaryormetastasis",
    "samplecollectionsite",
    "sourcetype",
    "growthpattern",
    "rrid",
    "sangermodelid",
    "cosmicid",
    "age",
    "agecategory",
    "sex",
    "molecularsubtype",
    "legacymolecularsubtype",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-timeout", type=int, default=120)
    parser.add_argument("--download-timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--gene-effect",
        type=Path,
        help=(
            "Local CRISPRGeneEffect.csv downloaded from the frozen release page. "
            "Must be supplied together with --model-metadata."
        ),
    )
    parser.add_argument(
        "--model-metadata",
        type=Path,
        help=(
            "Local Model.csv/Models.csv downloaded from the frozen release page. "
            "Must be supplied together with --gene-effect."
        ),
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def safe_source_location(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def validate_https_url(url: str, label: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(f"{label} must be a complete HTTPS URL; got {url!r}")


def request_bytes(
    url: str,
    *,
    timeout: int,
    retries: int,
    accept: str,
    max_bytes: int = 25 * 1024 * 1024,
) -> tuple[bytes, dict[str, str]]:
    validate_https_url(url, "Source URL")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise RuntimeError(
                        f"Response exceeded the {max_bytes}-byte safety limit: {url}"
                    )
                headers = {key.lower(): value for key, value in response.headers.items()}
                return raw, headers
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(
        f"Request failed after {retries} attempts: {url}; error={last_error}"
    ) from last_error


def parse_download_manifest(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig", errors="strict")
    if "<html" in text[:1000].lower() or "quick check before you enter" in text.lower():
        raise RuntimeError(
            "DepMap returned an HTML verification page instead of the official "
            "bulk-download CSV. No source file was accepted."
        )
    reader = csv.DictReader(io.StringIO(text))
    required = {"release", "release_date", "filename", "url", "md5_hash"}
    fields = {str(value).strip() for value in (reader.fieldnames or [])}
    missing = required - fields
    if missing:
        raise RuntimeError(
            "Unexpected DepMap bulk-download manifest columns; "
            f"missing={sorted(missing)}, observed={sorted(fields)}"
        )
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({str(key).strip(): str(value or "").strip() for key, value in row.items()})
    if not rows:
        raise RuntimeError("The DepMap bulk-download manifest contained no rows")
    return rows


def unique_source_row(
    rows: Iterable[dict[str, str]],
    *,
    release: str,
    filenames: list[str],
    label: str,
) -> dict[str, str]:
    exact_release = [row for row in rows if row.get("release") == release]
    if not exact_release:
        observed = sorted({row.get("release", "") for row in rows if release[-4:] in row.get("release", "")})
        raise RuntimeError(
            f"Exact release {release!r} was absent from the official manifest; "
            f"nearby={observed[:20]}"
        )
    for filename in filenames:
        matches = [row for row in exact_release if row.get("filename") == filename]
        if len(matches) == 1:
            if not matches[0].get("url"):
                raise RuntimeError(f"Official {label} row has no download URL")
            validate_https_url(matches[0]["url"], f"Official {label} URL")
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Official manifest has {len(matches)} duplicate rows for "
                f"release={release!r}, filename={filename!r}"
            )
    available = sorted(row.get("filename", "") for row in exact_release)
    raise RuntimeError(
        f"Could not find exact {label} filename(s) {filenames!r} in release "
        f"{release!r}; available filenames include {available[:40]}"
    )


def download_file(
    row: dict[str, str],
    destination: Path,
    *,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    url = row["url"]
    expected_md5 = row.get("md5_hash", "").strip().lower()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        sha256 = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        byte_count = 0
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream,*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    handle.write(chunk)
                    sha256.update(chunk)
                    md5.update(chunk)
                    byte_count += len(chunk)
                    if byte_count and byte_count % (256 * 1024 * 1024) < len(chunk):
                        print(
                            f"download {row['filename']}: "
                            f"{byte_count / (1024**2):.1f} MiB",
                            flush=True,
                        )
            observed_md5 = md5.hexdigest()
            if expected_md5 and observed_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for {row['filename']}: "
                    f"expected={expected_md5}, observed={observed_md5}"
                )
            if byte_count == 0:
                raise RuntimeError(f"Downloaded source file is empty: {row['filename']}")
            return {
                "bytes": byte_count,
                "md5": observed_md5,
                "sha256": sha256.hexdigest(),
                "official_md5": expected_md5 or None,
            }
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(
        f"Download failed after {retries} attempts: {row['filename']}; "
        f"error={last_error}"
    ) from last_error


def find_model_id(columns: Iterable[str], label: str) -> str:
    matches = [column for column in columns if normalized(column) in MODEL_ID_ALIASES]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one model-ID column in {label}; observed={matches}"
        )
    return matches[0]


def find_gene_columns(columns: Iterable[str], genes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for gene in genes:
        pattern = re.compile(rf"^{re.escape(gene)}(?:\s*\(|$)", re.IGNORECASE)
        matches = [column for column in columns if pattern.match(str(column).strip())]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one source column for {gene}; observed={matches}"
            )
        mapping[gene] = matches[0]
    return mapping


def export_subsets(
    gene_effect_path: Path,
    model_path: Path,
    output_dir: Path,
    release_token: str,
    genes: list[str],
    min_rows: int,
) -> dict[str, Any]:
    import pandas as pd

    gene_header = list(pd.read_csv(gene_effect_path, nrows=0).columns)
    gene_id_column = find_model_id(gene_header, "CRISPR gene-effect table")
    gene_mapping = find_gene_columns(gene_header, genes)
    gene_usecols = [gene_id_column, *gene_mapping.values()]
    gene_effect = pd.read_csv(
        gene_effect_path,
        usecols=gene_usecols,
        dtype={gene_id_column: "string"},
        low_memory=False,
    )
    gene_effect = gene_effect.rename(
        columns={gene_id_column: "ModelID", **{source: gene for gene, source in gene_mapping.items()}}
    )
    gene_effect["ModelID"] = gene_effect["ModelID"].astype("string").str.strip()
    if gene_effect["ModelID"].isna().any() or (gene_effect["ModelID"] == "").any():
        raise RuntimeError("CRISPR gene-effect subset contains missing model IDs")
    if gene_effect["ModelID"].duplicated().any():
        duplicates = gene_effect.loc[gene_effect["ModelID"].duplicated(), "ModelID"].head().tolist()
        raise RuntimeError(f"CRISPR gene-effect subset has duplicate model IDs: {duplicates}")
    if len(gene_effect) < min_rows:
        raise RuntimeError(
            f"CRISPR gene-effect subset has only {len(gene_effect)} rows; "
            f"request minimum={min_rows}"
        )
    for gene in genes:
        gene_effect[gene] = pd.to_numeric(gene_effect[gene], errors="coerce")
        if int(gene_effect[gene].notna().sum()) < min_rows:
            raise RuntimeError(
                f"{gene} has fewer than {min_rows} finite gene-effect values"
            )
    gene_effect = gene_effect.sort_values("ModelID", kind="stable").reset_index(drop=True)

    model_header = list(pd.read_csv(model_path, nrows=0).columns)
    model_id_column = find_model_id(model_header, "model metadata table")
    selected_metadata = [
        column
        for column in model_header
        if normalized(column) in METADATA_COLUMNS or column == model_id_column
    ]
    if not any(normalized(column) in LINEAGE_ALIASES for column in selected_metadata):
        raise RuntimeError(
            "Model metadata table has no recognized lineage or primary-disease column"
        )
    model = pd.read_csv(
        model_path,
        usecols=selected_metadata,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    ).rename(columns={model_id_column: "ModelID"})
    model["ModelID"] = model["ModelID"].str.strip()
    model = model.loc[model["ModelID"].isin(set(gene_effect["ModelID"].tolist()))].copy()
    if model["ModelID"].duplicated().any():
        duplicates = model.loc[model["ModelID"].duplicated(), "ModelID"].head().tolist()
        raise RuntimeError(f"Model metadata subset has duplicate model IDs: {duplicates}")
    model = model.sort_values("ModelID", kind="stable").reset_index(drop=True)
    matched = int(gene_effect["ModelID"].isin(set(model["ModelID"])).sum())
    coverage = matched / len(gene_effect)
    if coverage < 0.98:
        raise RuntimeError(
            f"Model metadata join coverage is only {coverage:.3%} ({matched}/{len(gene_effect)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_output = output_dir / f"{release_token}_CRISPRGeneEffect_HDAC1_2_3_8.csv"
    model_output = output_dir / f"{release_token}_Model_metadata.csv"
    gene_effect.to_csv(gene_output, index=False, lineterminator="\n", float_format="%.10g")
    model.to_csv(model_output, index=False, lineterminator="\n")
    return {
        "gene_effect_output": gene_output,
        "model_output": model_output,
        "gene_effect_rows": int(len(gene_effect)),
        "model_metadata_rows": int(len(model)),
        "model_metadata_columns": list(model.columns),
        "join_matched_rows": matched,
        "join_coverage": coverage,
        "finite_gene_effect_values": {
            gene: int(gene_effect[gene].notna().sum()) for gene in genes
        },
    }


def source_record(row: dict[str, str], download: dict[str, Any]) -> dict[str, Any]:
    return {
        "release": row["release"],
        "release_date": row["release_date"],
        "filename": row["filename"],
        "source_location_without_query": safe_source_location(row["url"]),
        **download,
    }


def local_source_record(path: Path, *, release: str, expected_name: str) -> dict[str, Any]:
    """Record a browser-downloaded input without claiming official-MD5 verification."""
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Local DepMap source file is empty: {path}")
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
    return {
        "release": release,
        "filename": path.name,
        "expected_filename": expected_name,
        "source_mode": "local_browser_download",
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha256_file(path),
        "official_md5": None,
        "verification_note": (
            "Release and filename were frozen before download; CSV schema, row count, "
            "model join, and content hashes were verified locally. The official manifest "
            "MD5 was unavailable because the portal blocked automated manifest access."
        ),
    }


def main() -> None:
    args = parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    if request.get("schema_version") != 1:
        raise RuntimeError("Unsupported or missing request schema_version")
    release = str(request["release"])
    genes = [str(value) for value in request["gene_symbols"]]
    if genes != ["HDAC1", "HDAC2", "HDAC3", "HDAC8"]:
        raise RuntimeError(f"Frozen E2 request has unexpected genes: {genes}")
    manifest_url = str(request["source_manifest_url"])
    if manifest_url != "https://depmap.org/portal/api/download/files":
        raise RuntimeError(f"Unexpected DepMap source manifest URL: {manifest_url}")

    local_flags = (args.gene_effect is not None, args.model_metadata is not None)
    if any(local_flags) and not all(local_flags):
        raise RuntimeError(
            "Local-input mode requires both --gene-effect and --model-metadata"
        )

    manifest_raw: bytes | None = None
    manifest_headers: dict[str, str] = {}
    if all(local_flags):
        gene_path = args.gene_effect.expanduser().resolve()
        model_path = args.model_metadata.expanduser().resolve()
        missing = [str(path) for path in (gene_path, model_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing local DepMap input(s): {missing}")
        expected_gene_name = str(request["gene_effect_filename"])
        expected_model_names = [
            str(value) for value in request["model_metadata_filenames"]
        ]
        if gene_path.name != expected_gene_name:
            raise RuntimeError(
                f"Expected local gene-effect filename {expected_gene_name!r}; "
                f"observed={gene_path.name!r}"
            )
        if model_path.name not in expected_model_names:
            raise RuntimeError(
                f"Expected local model metadata filename in {expected_model_names!r}; "
                f"observed={model_path.name!r}"
            )
        source_mode = "local_browser_download"
        gene_source = local_source_record(
            gene_path, release=release, expected_name=expected_gene_name
        )
        model_source = local_source_record(
            model_path, release=release, expected_name=model_path.name
        )
        print(f"frozen release: {release}")
        print(f"local gene-effect file: {gene_path}")
        print(f"local model metadata file: {model_path}")
        subset = export_subsets(
            gene_path,
            model_path,
            args.output_dir,
            str(request["output_release_token"]),
            genes,
            int(request.get("minimum_gene_effect_rows", 500)),
        )
    else:
        source_mode = "official_manifest_download"
        manifest_raw, manifest_headers = request_bytes(
            manifest_url,
            timeout=args.manifest_timeout,
            retries=args.retries,
            accept="text/csv,text/plain;q=0.9,*/*;q=0.1",
        )
        rows = parse_download_manifest(manifest_raw)
        gene_row = unique_source_row(
            rows,
            release=release,
            filenames=[str(request["gene_effect_filename"])],
            label="CRISPR gene-effect source",
        )
        model_row = unique_source_row(
            rows,
            release=release,
            filenames=[str(value) for value in request["model_metadata_filenames"]],
            label="model metadata source",
        )
        print(f"official release: {release}")
        print(f"gene-effect file: {gene_row['filename']}")
        print(f"model metadata file: {model_row['filename']}")
        with tempfile.TemporaryDirectory(prefix="depmap-hdac-") as temporary:
            temporary_dir = Path(temporary)
            gene_path = temporary_dir / gene_row["filename"]
            model_path = temporary_dir / model_row["filename"]
            gene_download = download_file(
                gene_row,
                gene_path,
                timeout=args.download_timeout,
                retries=args.retries,
            )
            model_download = download_file(
                model_row,
                model_path,
                timeout=args.download_timeout,
                retries=args.retries,
            )
            gene_source = source_record(gene_row, gene_download)
            model_source = source_record(model_row, model_download)
            subset = export_subsets(
                gene_path,
                model_path,
                args.output_dir,
                str(request["output_release_token"]),
                genes,
                int(request.get("minimum_gene_effect_rows", 500)),
            )

    outputs = [subset["gene_effect_output"], subset["model_output"]]
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "request_path": str(args.request),
        "request_sha256": sha256_bytes(request_raw),
        "official_manifest_url": manifest_url,
        "official_manifest_sha256": (
            sha256_bytes(manifest_raw) if manifest_raw is not None else None
        ),
        "official_manifest_content_type": manifest_headers.get("content-type"),
        "release": release,
        "genes": genes,
        "source_files": [
            gene_source,
            model_source,
        ],
        "subset": {
            key: value
            for key, value in subset.items()
            if key not in {"gene_effect_output", "model_output"}
        },
        "outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
        "software": {
            "python": platform.python_version(),
            "pandas": __import__("pandas").__version__,
        },
        "interpretation_guardrail": (
            "DepMap CRISPR gene effect is genetic dependency evidence. It is not "
            "direct drug-response validation and must not be described as such."
        ),
    }
    manifest_path = args.output_dir / "depmap_26q1_hdac_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path = args.output_dir / "audit.txt"
    audit_lines = [
        "DEPMAP 26Q1 HDAC SOURCE SUBSET: PASSED",
        f"release={release}",
        f"source_mode={source_mode}",
        f"gene_effect_rows={subset['gene_effect_rows']}",
        f"model_metadata_rows={subset['model_metadata_rows']}",
        f"join_coverage={subset['join_coverage']:.6f}",
        *(f"{path.name}_sha256={sha256_file(path)}" for path in outputs),
        f"manifest_sha256={sha256_file(manifest_path)}",
    ]
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print("===== DEPMAP 26Q1 HDAC SUBSET =====")
    print(json.dumps(manifest["subset"], indent=2))
    print("\n".join(audit_lines))


if __name__ == "__main__":
    main()
