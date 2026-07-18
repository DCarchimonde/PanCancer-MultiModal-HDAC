from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib
import numpy as np
import pandas as pd

try:
    from build_reversal_gene_networks_and_pathways import (
        CANDIDATE_ORDER,
        ENRICHMENT_SOURCES,
        GENE_SET_SCOPES,
        gene_sets,
        map_gene_ids,
        require_columns,
        run_enrichment,
        run_string_networks,
        unique_strings,
    )
except ModuleNotFoundError:  # Supports import as ``scripts.hydrate_network_api_cache``.
    from scripts.build_reversal_gene_networks_and_pathways import (
        CANDIDATE_ORDER,
        ENRICHMENT_SOURCES,
        GENE_SET_SCOPES,
        gene_sets,
        map_gene_ids,
        require_columns,
        run_enrichment,
        run_string_networks,
        unique_strings,
    )


DEFAULT_BUNDLE = Path(
    "results/revision/reversal_gene_networks_pathways/"
    "network_api_request_bundle.json"
)
DEFAULT_RAW_DIR = Path(
    "results/revision/reversal_gene_networks_pathways/raw_api"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hydrate the versioned g:Profiler and STRING response cache from a "
            "frozen network-analysis request bundle."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-delay-seconds", type=float, default=0.5)
    parser.add_argument("--refresh-api", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(bundle: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    if bundle.get("schema_version") != 1:
        raise RuntimeError(
            f"Unsupported network API bundle schema: {bundle.get('schema_version')}"
        )
    if bundle.get("candidate_order") != CANDIDATE_ORDER:
        raise RuntimeError("Candidate order in the API bundle is not frozen")
    if bundle.get("gene_set_scopes") != GENE_SET_SCOPES:
        raise RuntimeError("Gene-set scopes in the API bundle changed")
    if bundle.get("enrichment_sources") != ENRICHMENT_SOURCES:
        raise RuntimeError("Enrichment sources in the API bundle changed")

    parameters = bundle.get("analysis_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("API bundle lacks analysis_parameters")
    expected_parameters = {
        "string_species": 9606,
        "string_network_types": ["physical", "functional"],
        "string_added_nodes": 0,
        "enrichment_user_threshold": 0.05,
        "enrichment_significance_threshold_method": "fdr",
        "enrichment_domain_scope": "custom",
    }
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise RuntimeError(
                f"API bundle parameter changed: {key}={parameters.get(key)!r}; "
                f"expected {expected!r}"
            )
    required_urls = ["gprofiler_base", "string_base"]
    if any(not str(parameters.get(key, "")).startswith("https://") for key in required_urls):
        raise RuntimeError("API bundle endpoints must use HTTPS")

    records = bundle.get("selected_consensus_records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("API bundle has no selected consensus records")
    consensus = pd.DataFrame(records)
    require_columns(
        consensus,
        [
            "candidate",
            "gene_id",
            "selected_network_gene",
            "consensus_rank",
            "reversal_pattern",
        ],
        "API-bundle selected consensus",
    )
    consensus["gene_id"] = consensus["gene_id"].astype(str).str.strip()
    consensus["selected_network_gene"] = (
        consensus["selected_network_gene"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if not consensus["selected_network_gene"].all():
        raise RuntimeError("API bundle contains a non-selected consensus row")
    if set(consensus["candidate"]) != set(CANDIDATE_ORDER):
        raise RuntimeError("API bundle candidate membership changed")
    if consensus.duplicated(["candidate", "gene_id"]).any():
        raise RuntimeError("API bundle contains duplicate candidate/gene rows")

    background_value = bundle.get("custom_background_gene_ids")
    if not isinstance(background_value, list):
        raise RuntimeError("API bundle lacks custom_background_gene_ids")
    background = unique_strings(background_value)
    if len(background) < 1_000:
        raise RuntimeError(
            f"Custom measured-gene background is unexpectedly small: {len(background)}"
        )
    if not set(consensus["gene_id"]).issubset(set(background)):
        raise RuntimeError("Selected genes are not a subset of the custom background")

    observed_counts = (
        consensus.groupby("candidate")["gene_id"]
        .nunique()
        .reindex(CANDIDATE_ORDER)
        .astype(int)
        .to_dict()
    )
    expected_counts = bundle.get("selected_genes_per_candidate")
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Selected-gene counts differ from bundle manifest: "
            f"{observed_counts} != {expected_counts}"
        )
    return consensus, background


def main() -> None:
    cli = parse_args()
    started = time.perf_counter()
    bundle = json.loads(cli.bundle.read_text(encoding="utf-8"))
    consensus, background = validate_bundle(bundle)
    parameters = bundle["analysis_parameters"]
    api_args = argparse.Namespace(
        gprofiler_base=str(parameters["gprofiler_base"]),
        string_base=str(parameters["string_base"]),
        required_score=int(parameters["string_required_score"]),
        timeout_seconds=cli.timeout_seconds,
        api_retries=cli.api_retries,
        api_delay_seconds=cli.api_delay_seconds,
        refresh_api=cli.refresh_api,
    )
    cli.raw_dir.mkdir(parents=True, exist_ok=True)

    sets = gene_sets(consensus)
    selected_union = unique_strings(
        gene
        for candidate in CANDIDATE_ORDER
        for gene in sets[(candidate, "all_selected")]
    )
    mapping, convert_metadata = map_gene_ids(selected_union, api_args, cli.raw_dir)
    selected_with_mapping = consensus[["candidate", "gene_id"]].merge(
        mapping,
        on="gene_id",
        how="left",
        validate="many_to_one",
    )
    mapping_coverage = (
        selected_with_mapping.groupby("candidate", as_index=False)
        .agg(
            selected_genes=("gene_id", "nunique"),
            string_mappable_genes=("string_mappable", "sum"),
        )
    )
    mapping_coverage["mapping_fraction"] = (
        mapping_coverage["string_mappable_genes"]
        / mapping_coverage["selected_genes"]
    )
    if (mapping_coverage["mapping_fraction"] < 0.90).any():
        raise RuntimeError(
            "STRING mapping coverage below 90%:\n"
            + mapping_coverage.to_string(index=False)
        )

    enrichment, enrichment_metadata, enrichment_inventory = run_enrichment(
        sets, background, api_args, cli.raw_dir
    )
    edges, nodes, network_summary = run_string_networks(
        consensus, mapping, api_args, cli.raw_dir
    )
    if enrichment.empty:
        raise RuntimeError("g:Profiler returned no significant result for any gene set")
    if nodes.empty or network_summary.empty:
        raise RuntimeError("STRING cache hydration produced no network nodes")

    cache_paths = sorted(
        path
        for path in cli.raw_dir.glob("*.json")
        if path.name != "network_api_cache_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "analysis_stage": "network_pathway_api_cache_hydration",
        "request_bundle": str(cli.bundle),
        "request_bundle_sha256": sha256_file(cli.bundle),
        "candidate_order": CANDIDATE_ORDER,
        "custom_background_genes": len(background),
        "selected_union_genes": len(selected_union),
        "mapping_coverage": mapping_coverage.to_dict(orient="records"),
        "enrichment_query_inventory": enrichment_inventory.to_dict(orient="records"),
        "significant_enrichment_rows": int(len(enrichment)),
        "network_summary": network_summary.to_dict(orient="records"),
        "network_edge_rows": int(len(edges)),
        "network_node_rows": int(len(nodes)),
        "gprofiler_response_metadata": [
            *convert_metadata,
            *enrichment_metadata,
        ],
        "api_parameters": parameters,
        "cache_files": {
            path.name: sha256_file(path) for path in cache_paths
        },
        "cache_file_count": len(cache_paths),
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "transport_note": (
            "Responses were hydrated from the versioned endpoints encoded in "
            "the request bundle. The downstream analysis verifies each cached "
            "request SHA-256 before use."
        ),
    }
    manifest_path = cli.raw_dir / "network_api_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("===== API CACHE MAPPING COVERAGE =====")
    print(mapping_coverage.to_string(index=False))
    print("\n===== API CACHE CONTENT =====")
    print(f"significant_enrichment_rows: {len(enrichment)}")
    print(f"network_edge_rows: {len(edges)}")
    print(f"network_node_rows: {len(nodes)}")
    print(f"cache_files: {len(cache_paths)}")
    print(f"manifest: {manifest_path}")
    print("NETWORK/PATHWAY API CACHE HYDRATION: PASSED")


if __name__ == "__main__":
    main()
