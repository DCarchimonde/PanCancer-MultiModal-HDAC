#!/usr/bin/env python3
"""Rebuild cached g:Profiler exports with true intersection gene identifiers.

The original parser flattened g:Profiler's per-query evidence-code arrays and
mistakenly labelled the resulting GO evidence codes as ``intersection_genes``.
This repair is network-free: it reads the frozen, request-hash-validated API
cache, aligns each evidence row to g:Profiler's mapped query order, verifies
every derived intersection count, and regenerates the representative-term
selection and pathway heatmap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import build_reversal_gene_networks_and_pathways as network_pipeline


CANDIDATE_SLUGS = {
    "belinostat": "Belinostat",
    "entinostat": "Entinostat",
    "mocetinostat": "Mocetinostat",
    "nch_51": "NCH-51",
    "tc_h_106": "TC-H-106",
    "vorinostat": "Vorinostat",
}
SCOPES = (
    "all_selected",
    "disease_down_drug_up",
    "disease_up_drug_down",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("results/revision/reversal_gene_networks_pathways/raw_api"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/reversal_gene_networks_pathways"),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identify(path: Path) -> tuple[str, str]:
    stem = path.stem.removeprefix("gost_")
    for scope in SCOPES:
        suffix = f"_{scope}"
        if stem.endswith(suffix):
            slug = stem[: -len(suffix)]
            if slug not in CANDIDATE_SLUGS:
                raise RuntimeError(f"Unknown candidate slug in {path.name}: {slug}")
            return CANDIDATE_SLUGS[slug], scope
    raise RuntimeError(f"Could not identify gene-set scope from {path.name}")


def load_cached_response(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = payload.get("response", payload)
    if not isinstance(response, dict) or "result" not in response or "meta" not in response:
        raise RuntimeError(f"Invalid frozen g:Profiler response: {path}")
    return response


def rebuild(raw_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    paths = sorted(raw_dir.glob("gost_*.json"))
    if len(paths) != len(CANDIDATE_SLUGS) * len(SCOPES):
        raise RuntimeError(
            f"Expected 18 frozen g:Profiler responses, observed {len(paths)} in {raw_dir}"
        )
    for path in paths:
        candidate, scope = identify(path)
        response = load_cached_response(path)
        genes_metadata = response["meta"].get("genes_metadata", {})
        query_metadata = genes_metadata.get("query", {})
        if not query_metadata:
            raise RuntimeError(f"Missing mapped-query metadata in {path}")
        query_entry = next(iter(query_metadata.values()))
        mapped_query_ids = [str(value) for value in query_entry.get("mapping", {}).keys()]
        if not mapped_query_ids:
            raise RuntimeError(f"Empty mapped-query order in {path}")
        for result in response["result"]:
            genes, evidence_codes = network_pipeline.intersection_values(
                result,
                mapped_query_ids,
            )
            row = dict(result)
            row["candidate"] = candidate
            row["gene_set_scope"] = scope
            row["intersection_genes"] = genes
            row["intersection_evidence_codes"] = evidence_codes
            rows.append(row)
        source_records.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "candidate": candidate,
                "gene_set_scope": scope,
                "mapped_query_ids": len(mapped_query_ids),
                "reported_terms": len(response["result"]),
                "gprofiler_version": response["meta"].get("version"),
                "timestamp": response["meta"].get("timestamp"),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 5037:
        raise RuntimeError(f"Expected 5,037 cached enrichment terms, observed {len(frame)}")
    return frame, source_records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full, source_records = rebuild(args.raw_dir)
    representative = network_pipeline.representative_terms(
        full,
        max_terms=5,
        min_term_size=10,
        max_term_size=2_000,
        redundancy_jaccard=0.50,
    )
    full_path = args.output_dir / "gprofiler_enrichment_full.csv"
    representative_path = args.output_dir / "gprofiler_enrichment_representative.csv"
    network_pipeline.serialize_intersections(full).to_csv(full_path, index=False)
    network_pipeline.serialize_intersections(representative).to_csv(
        representative_path,
        index=False,
    )
    network_pipeline.plot_pathway_heatmap(
        representative,
        args.output_dir,
        args.figure_dpi,
    )
    audit = {
        "repair": "gprofiler_intersection_gene_alignment",
        "network_requests_performed": False,
        "source_cache_files": source_records,
        "full_terms": len(full),
        "representative_terms": len(representative),
        "intersection_count_gate": "all derived gene counts equal g:Profiler intersection_size",
        "alignment_source": "meta.genes_metadata.query.<query>.mapping key order",
        "evidence_code_policy": (
            "GO evidence codes are retained separately as intersection_evidence_codes "
            "and are never labelled as genes."
        ),
        "representative_term_policy": {
            "max_terms_per_candidate_scope_source": 5,
            "term_size_min": 10,
            "term_size_max": 2_000,
            "intersection_gene_jaccard_exclusion_threshold": 0.50,
        },
        "outputs_sha256": {
            full_path.name: sha256_file(full_path),
            representative_path.name: sha256_file(representative_path),
            "reversal_pathway_enrichment_heatmap.pdf": sha256_file(
                args.output_dir / "reversal_pathway_enrichment_heatmap.pdf"
            ),
        },
    }
    audit_path = args.output_dir / "gprofiler_intersection_repair_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("GPROFILER INTERSECTION REPAIR: PASSED")
    print(f"full_terms={len(full)}")
    print(f"representative_terms={len(representative)}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
