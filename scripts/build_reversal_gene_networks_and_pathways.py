from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CANDIDATE_ORDER = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "Entinostat",
    "Vorinostat",
]
PRIMARY_CANDIDATES = ["Mocetinostat", "NCH-51", "TC-H-106"]
REFERENCE_CONTROLS = ["Entinostat", "Vorinostat"]
MAIN_NETWORK_ORDER = PRIMARY_CANDIDATES + REFERENCE_CONTROLS
ENRICHMENT_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
GENE_SET_SCOPES = [
    "all_selected",
    "disease_up_drug_down",
    "disease_down_drug_up",
]
SOURCE_COLORS = {
    "GO:BP": "#4C78A8",
    "GO:MF": "#72B7B2",
    "GO:CC": "#54A24B",
    "KEGG": "#F58518",
    "REAC": "#E45756",
}
REVERSAL_COLORS = {
    "disease_up_drug_down": "#D73027",
    "disease_down_drug_up": "#4575B4",
    "mixed_or_context_dependent": "#999999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build high-confidence STRING interaction networks and GO/KEGG/Reactome "
            "enrichment for frozen reversal-associated gene sets."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/revision/frozen_candidate_biology"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/reversal_gene_networks_pathways"),
    )
    parser.add_argument(
        "--gprofiler-base",
        default="https://biit.cs.ut.ee/gprofiler/api",
    )
    parser.add_argument(
        "--string-base",
        default="https://version-12-0.string-db.org/api",
    )
    parser.add_argument("--required-score", type=int, default=700)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-delay-seconds", type=float, default=0.5)
    parser.add_argument("--refresh-api", action="store_true")
    parser.add_argument("--max-representative-terms", type=int, default=5)
    parser.add_argument("--representative-min-term-size", type=int, default=10)
    parser.add_argument("--representative-max-term-size", type=int, default=2_000)
    parser.add_argument("--redundancy-jaccard", type=float, default=0.50)
    parser.add_argument("--plot-top-n", type=int, default=35)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=20260718)
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


def json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def unique_strings(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def post_bytes(
    url: str,
    data: bytes,
    content_type: str,
    timeout: int,
    retries: int,
) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": content_type,
                "Accept": "application/json, text/tab-separated-values, */*",
                "User-Agent": "PanCancer-MultiModal-HDAC-major-revision/2026",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), dict(response.headers.items())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            details = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    details = error.read().decode("utf-8", errors="replace")[:2_000]
                except Exception:
                    details = ""
            if attempt == retries:
                raise RuntimeError(
                    f"API request failed after {retries} attempts: {url}; "
                    f"error={error}; response={details}"
                ) from error
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"API request failed: {url}; {last_error}")


def cached_json_post(
    url: str,
    payload: dict[str, Any],
    cache_path: Path,
    timeout: int,
    retries: int,
    refresh: bool,
) -> dict[str, Any]:
    request_sha = json_hash({"url": url, "payload": payload})
    if cache_path.is_file() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_sha256") != request_sha:
            raise RuntimeError(
                f"Cached API request does not match current parameters: {cache_path}. "
                "Use --refresh-api to replace it."
            )
        return cached["response"]
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    raw, headers = post_bytes(
        url,
        body,
        "application/json; charset=utf-8",
        timeout,
        retries,
    )
    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"API did not return JSON: {url}; body={raw[:2_000]!r}"
        ) from error
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "url": url,
                "request_sha256": request_sha,
                "response_headers": headers,
                "response": response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return response


def cached_form_post(
    url: str,
    payload: dict[str, Any],
    cache_path: Path,
    timeout: int,
    retries: int,
    refresh: bool,
) -> str:
    request_sha = json_hash({"url": url, "payload": payload})
    if cache_path.is_file() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_sha256") != request_sha:
            raise RuntimeError(
                f"Cached STRING request does not match current parameters: {cache_path}. "
                "Use --refresh-api to replace it."
            )
        return str(cached["response_text"])
    body = urllib.parse.urlencode(payload).encode("utf-8")
    raw, headers = post_bytes(
        url,
        body,
        "application/x-www-form-urlencoded; charset=utf-8",
        timeout,
        retries,
    )
    text = raw.decode("utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "url": url,
                "request_sha256": request_sha,
                "response_headers": headers,
                "response_text": text,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return text


def numeric_namespace(gene_ids: list[str]) -> str:
    return "ENTREZGENE_ACC" if gene_ids and all(value.isdigit() for value in gene_ids) else ""


def normalize_incoming(value: object, expected: set[str]) -> str:
    text = str(value).strip()
    if text in expected:
        return text
    suffix = text.rsplit(":", 1)[-1]
    return suffix if suffix in expected else text


def map_gene_ids(
    gene_ids: list[str],
    args: argparse.Namespace,
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    expected = set(gene_ids)
    namespace = numeric_namespace(gene_ids)
    batch_size = 500
    for batch_index, start in enumerate(range(0, len(gene_ids), batch_size), start=1):
        batch = gene_ids[start : start + batch_size]
        payload: dict[str, Any] = {
            "organism": "hsapiens",
            "target": "ENSG",
            "query": batch,
        }
        if namespace:
            payload["numeric_namespace"] = namespace
        response = cached_json_post(
            f"{args.gprofiler_base.rstrip('/')}/convert/convert/",
            payload,
            raw_dir / f"gconvert_batch_{batch_index:03d}.json",
            args.timeout_seconds,
            args.api_retries,
            args.refresh_api,
        )
        if "result" not in response:
            raise RuntimeError(f"g:Convert response lacks result: batch {batch_index}")
        metadata.append(response.get("meta", {}))
        for result in response["result"]:
            row = dict(result)
            row["gene_id"] = normalize_incoming(row.get("incoming", ""), expected)
            rows.append(row)
        time.sleep(args.api_delay_seconds)

    converted = pd.DataFrame(rows)
    if converted.empty:
        raise RuntimeError("g:Convert returned no mappings")
    for column in ["converted", "name", "description", "namespaces"]:
        if column not in converted.columns:
            converted[column] = ""
    converted["mapped"] = (
        converted["converted"].notna()
        & converted["converted"].astype(str).ne("N/A")
        & converted["converted"].astype(str).ne("")
    )
    converted = converted.sort_values(
        ["gene_id", "mapped", "converted"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("gene_id", keep="first")
    missing = sorted(expected - set(converted["gene_id"]))
    if missing:
        converted = pd.concat(
            [
                converted,
                pd.DataFrame(
                    {
                        "gene_id": missing,
                        "incoming": missing,
                        "converted": "",
                        "name": "",
                        "description": "",
                        "namespaces": "",
                        "mapped": False,
                    }
                ),
            ],
            ignore_index=True,
        )
    converted["gene_symbol"] = converted["name"].fillna("").astype(str).str.strip()
    converted["ensembl_gene_id"] = (
        converted["converted"].fillna("").astype(str).str.strip()
    )
    converted["string_query_id"] = converted["gene_symbol"]
    use_ensembl = converted["string_query_id"].eq("")
    converted.loc[use_ensembl, "string_query_id"] = converted.loc[
        use_ensembl, "ensembl_gene_id"
    ]
    converted["string_mappable"] = (
        converted["mapped"] & converted["string_query_id"].ne("")
    )
    keep = [
        "gene_id",
        "incoming",
        "ensembl_gene_id",
        "gene_symbol",
        "description",
        "namespaces",
        "mapped",
        "string_query_id",
        "string_mappable",
    ]
    return converted[keep].sort_values("gene_id"), metadata


def gene_sets(consensus: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    selected = consensus.loc[consensus["selected_network_gene"]].copy()
    sets: dict[tuple[str, str], list[str]] = {}
    for candidate in CANDIDATE_ORDER:
        subset = selected.loc[selected["candidate"] == candidate].sort_values(
            "consensus_rank", kind="stable"
        )
        sets[(candidate, "all_selected")] = subset["gene_id"].astype(str).tolist()
        for direction in GENE_SET_SCOPES[1:]:
            sets[(candidate, direction)] = subset.loc[
                subset["reversal_pattern"] == direction, "gene_id"
            ].astype(str).tolist()
    return sets


def intersection_values(result: dict[str, Any]) -> list[str]:
    value = result.get("intersections", result.get("intersection", []))
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    flattened: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(part) for part in item if str(part))
            elif item is not None and str(item):
                flattened.append(str(item))
    return unique_strings(flattened)


def run_enrichment(
    sets: dict[tuple[str, str], list[str]],
    background: list[str],
    args: argparse.Namespace,
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    namespace = numeric_namespace(background)
    for candidate in CANDIDATE_ORDER:
        for scope in GENE_SET_SCOPES:
            query = sets[(candidate, scope)]
            if len(query) < 5:
                inventory.append(
                    {
                        "candidate": candidate,
                        "gene_set_scope": scope,
                        "query_genes": len(query),
                        "status": "skipped_fewer_than_5_genes",
                        "significant_terms": 0,
                    }
                )
                continue
            payload: dict[str, Any] = {
                "organism": "hsapiens",
                "query": query,
                "sources": ENRICHMENT_SOURCES,
                "user_threshold": 0.05,
                "significance_threshold_method": "fdr",
                "domain_scope": "custom",
                "background": background,
                "ordered": False,
                "no_evidences": False,
                "all_results": False,
                "measure_underrepresentation": False,
            }
            if namespace:
                payload["numeric_namespace"] = namespace
            response = cached_json_post(
                f"{args.gprofiler_base.rstrip('/')}/gost/profile/",
                payload,
                raw_dir / f"gost_{slug(candidate)}_{scope}.json",
                args.timeout_seconds,
                args.api_retries,
                args.refresh_api,
            )
            if "result" not in response:
                raise RuntimeError(
                    f"g:Profiler response lacks result: {candidate}/{scope}"
                )
            metadata.append(response.get("meta", {}))
            for result in response["result"]:
                row = dict(result)
                row["candidate"] = candidate
                row["gene_set_scope"] = scope
                row["intersection_genes"] = intersection_values(result)
                rows.append(row)
            inventory.append(
                {
                    "candidate": candidate,
                    "gene_set_scope": scope,
                    "query_genes": len(query),
                    "status": "completed",
                    "significant_terms": len(response["result"]),
                }
            )
            time.sleep(args.api_delay_seconds)
    return pd.DataFrame(rows), metadata, pd.DataFrame(inventory)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def representative_terms(
    enrichment: pd.DataFrame,
    max_terms: int,
    min_term_size: int,
    max_term_size: int,
    redundancy_jaccard: float,
) -> pd.DataFrame:
    if enrichment.empty:
        return enrichment.copy()
    require_columns(
        enrichment,
        [
            "candidate",
            "gene_set_scope",
            "source",
            "native",
            "name",
            "p_value",
            "term_size",
            "intersection_genes",
        ],
        "g:Profiler enrichment",
    )
    selected_indices: list[int] = []
    for _, subset in enrichment.groupby(
        ["candidate", "gene_set_scope", "source"], sort=False
    ):
        candidates = subset.loc[
            pd.to_numeric(subset["term_size"], errors="coerce").between(
                min_term_size, max_term_size
            )
        ].sort_values(["p_value", "native"], kind="stable")
        kept_sets: list[set[str]] = []
        for index, row in candidates.iterrows():
            genes = set(row["intersection_genes"])
            if genes and any(
                jaccard(genes, previous) >= redundancy_jaccard
                for previous in kept_sets
            ):
                continue
            selected_indices.append(index)
            kept_sets.append(genes)
            if len(kept_sets) >= max_terms:
                break
    representative = enrichment.loc[selected_indices].copy()
    representative["representative_rank"] = representative.groupby(
        ["candidate", "gene_set_scope", "source"]
    )["p_value"].rank(method="first")
    return representative.sort_values(
        ["candidate", "gene_set_scope", "source", "representative_rank"],
        kind="stable",
    )


def serialize_intersections(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "intersection_genes" in output.columns:
        output["intersection_genes"] = output["intersection_genes"].map(
            lambda values: " | ".join(str(value) for value in values)
        )
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (list, dict))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )
    return output


def parse_string_network(text: str, candidate: str, network_type: str) -> pd.DataFrame:
    stripped = text.strip()
    columns = [
        "stringId_A",
        "stringId_B",
        "preferredName_A",
        "preferredName_B",
        "ncbiTaxonId",
        "score",
        "nscore",
        "fscore",
        "pscore",
        "ascore",
        "escore",
        "dscore",
        "tscore",
    ]
    if not stripped:
        return pd.DataFrame(columns=["candidate", "network_type", *columns])
    if stripped.lower().startswith(("error", "problem")):
        raise RuntimeError(f"STRING returned an error for {candidate}/{network_type}: {stripped[:1000]}")
    frame = pd.read_csv(io.StringIO(stripped), sep="\t")
    require_columns(
        frame,
        ["preferredName_A", "preferredName_B", "score"],
        f"STRING {candidate}/{network_type}",
    )
    frame.insert(0, "network_type", network_type)
    frame.insert(0, "candidate", candidate)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    pair_keys = frame.apply(
        lambda row: "||".join(
            sorted(
                [
                    str(row["preferredName_A"]).upper(),
                    str(row["preferredName_B"]).upper(),
                ]
            )
        ),
        axis=1,
    )
    frame["_pair_key"] = pair_keys
    frame = frame.sort_values("score", ascending=False, kind="stable").drop_duplicates(
        "_pair_key", keep="first"
    )
    return frame.drop(columns="_pair_key").reset_index(drop=True)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def network_tables(
    candidate: str,
    network_type: str,
    edges: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = selected.copy()
    selected["node_key"] = selected["string_query_id"].astype(str).str.upper()
    selected = selected.loc[selected["node_key"].ne("")].drop_duplicates(
        "node_key", keep="first"
    )
    nodes = selected["node_key"].tolist()
    preferred_names = dict(
        zip(selected["node_key"], selected["string_query_id"].astype(str))
    )
    degree = {node: 0 for node in nodes}
    weighted_degree = {node: 0.0 for node in nodes}
    union_find = UnionFind(nodes)
    for row in edges.itertuples(index=False):
        left = str(row.preferredName_A).upper()
        right = str(row.preferredName_B).upper()
        for node, preferred in [
            (left, str(row.preferredName_A)),
            (right, str(row.preferredName_B)),
        ]:
            if node not in degree:
                degree[node] = 0
                weighted_degree[node] = 0.0
                union_find.parent[node] = node
                nodes.append(node)
            preferred_names[node] = preferred
        degree[left] += 1
        degree[right] += 1
        weighted_degree[left] += float(row.score)
        weighted_degree[right] += float(row.score)
        union_find.union(left, right)
    component_members: dict[str, list[str]] = {}
    for node in nodes:
        component_members.setdefault(union_find.find(node), []).append(node)
    component_order = sorted(
        component_members,
        key=lambda root: (-len(component_members[root]), min(component_members[root])),
    )
    component_id = {
        node: index + 1
        for index, root in enumerate(component_order)
        for node in component_members[root]
    }
    component_size = {
        node: len(component_members[union_find.find(node)]) for node in nodes
    }
    node_frame = pd.DataFrame(
        {
            "candidate": candidate,
            "network_type": network_type,
            "node_key": nodes,
            "preferred_name": [preferred_names[node] for node in nodes],
            "degree": [degree[node] for node in nodes],
            "weighted_degree": [weighted_degree[node] for node in nodes],
            "component_id": [component_id[node] for node in nodes],
            "component_size": [component_size[node] for node in nodes],
        }
    )
    annotation = selected.drop(columns=["node_key", "candidate"], errors="ignore").copy()
    annotation["join_key"] = annotation["string_query_id"].astype(str).str.upper()
    node_frame["join_key"] = node_frame["node_key"]
    node_frame = node_frame.merge(
        annotation,
        on="join_key",
        how="left",
        validate="one_to_one",
    ).drop(columns="join_key")
    node_count = len(node_frame)
    edge_count = len(edges)
    summary = {
        "candidate": candidate,
        "network_type": network_type,
        "selected_genes": int(selected["gene_id"].nunique()),
        "network_nodes": node_count,
        "nodes_with_edges": int((node_frame["degree"] > 0).sum()),
        "network_edges": edge_count,
        "density": (
            2.0 * edge_count / (node_count * (node_count - 1))
            if node_count > 1
            else 0.0
        ),
        "components": len(component_members),
        "largest_component_nodes": max(map(len, component_members.values()), default=0),
    }
    return node_frame, summary


def run_string_networks(
    consensus: pd.DataFrame,
    mapping: pd.DataFrame,
    args: argparse.Namespace,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = consensus.loc[consensus["selected_network_gene"]].copy()
    selected["gene_id"] = selected["gene_id"].astype(str)
    selected = selected.merge(
        mapping,
        on="gene_id",
        how="left",
        validate="many_to_one",
    )
    edge_frames: list[pd.DataFrame] = []
    node_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for candidate in CANDIDATE_ORDER:
        candidate_genes = selected.loc[
            (selected["candidate"] == candidate)
            & selected["string_mappable"].fillna(False)
        ].sort_values("consensus_rank", kind="stable")
        identifiers = unique_strings(candidate_genes["string_query_id"])
        if len(identifiers) < 2:
            raise RuntimeError(f"Fewer than two STRING-mappable genes for {candidate}")
        for network_type in ["physical", "functional"]:
            payload = {
                "identifiers": "\r".join(identifiers),
                "species": 9606,
                "required_score": args.required_score,
                "network_type": network_type,
                "add_nodes": 0,
                "caller_identity": "PanCancer-MultiModal-HDAC-major-revision",
            }
            text = cached_form_post(
                f"{args.string_base.rstrip('/')}/tsv/network",
                payload,
                raw_dir / f"string_{slug(candidate)}_{network_type}.json",
                args.timeout_seconds,
                args.api_retries,
                args.refresh_api,
            )
            edges = parse_string_network(text, candidate, network_type)
            nodes, summary = network_tables(
                candidate, network_type, edges, candidate_genes
            )
            edge_frames.append(edges)
            node_frames.append(nodes)
            summaries.append(summary)
            time.sleep(args.api_delay_seconds)
    return (
        pd.concat(edge_frames, ignore_index=True),
        pd.concat(node_frames, ignore_index=True),
        pd.DataFrame(summaries),
    )


def pairwise_overlap(
    named_sets: dict[str, set[str]], value_label: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in combinations_with_replacement(CANDIDATE_ORDER, 2):
        left_set, right_set = named_sets[left], named_sets[right]
        intersection = sorted(left_set & right_set)
        union = left_set | right_set
        rows.append(
            {
                "candidate_a": left,
                "candidate_b": right,
                f"{value_label}_a": len(left_set),
                f"{value_label}_b": len(right_set),
                f"shared_{value_label}": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                f"shared_{value_label}_values": " | ".join(intersection),
            }
        )
        if left != right:
            reverse = rows[-1].copy()
            reverse["candidate_a"], reverse["candidate_b"] = right, left
            reverse[f"{value_label}_a"], reverse[f"{value_label}_b"] = (
                reverse[f"{value_label}_b"],
                reverse[f"{value_label}_a"],
            )
            rows.append(reverse)
    return pd.DataFrame(rows)


def force_layout(
    nodes: list[str],
    edges: pd.DataFrame,
    seed: int,
    iterations: int = 250,
) -> dict[str, np.ndarray]:
    count = len(nodes)
    if count == 0:
        return {}
    if count == 1:
        return {nodes[0]: np.zeros(2)}
    rng = np.random.default_rng(seed)
    positions = rng.normal(0, 0.25, size=(count, 2))
    lookup = {node.upper(): index for index, node in enumerate(nodes)}
    edge_indices: list[tuple[int, int, float]] = []
    for row in edges.itertuples(index=False):
        left = lookup.get(str(row.preferredName_A).upper())
        right = lookup.get(str(row.preferredName_B).upper())
        if left is not None and right is not None and left != right:
            edge_indices.append((left, right, float(row.score)))
    k = 1.0 / math.sqrt(count)
    temperature = 0.20
    for _ in range(iterations):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)
        repulsion = (k * k / np.maximum(distance, 1e-6) ** 2)[:, :, None]
        displacement = np.sum(
            np.where(np.isfinite(distance)[:, :, None], delta * repulsion, 0.0),
            axis=1,
        )
        for left, right, weight in edge_indices:
            vector = positions[left] - positions[right]
            length = max(float(np.linalg.norm(vector)), 1e-6)
            attraction = vector / length * (length * length / k) * weight
            displacement[left] -= attraction
            displacement[right] += attraction
        norms = np.linalg.norm(displacement, axis=1)
        step = displacement / np.maximum(norms[:, None], 1e-9)
        positions += step * np.minimum(norms, temperature)[:, None]
        positions -= positions.mean(axis=0, keepdims=True)
        temperature *= 0.985
    scale = np.max(np.abs(positions))
    if scale > 0:
        positions /= scale
    return {node: positions[index] for index, node in enumerate(nodes)}


def plot_overlap_heatmap(
    overlap: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    matrix = overlap.pivot(
        index="candidate_a", columns="candidate_b", values="jaccard"
    ).reindex(index=CANDIDATE_ORDER, columns=CANDIDATE_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CANDIDATE_ORDER)), CANDIDATE_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(CANDIDATE_ORDER)), CANDIDATE_ORDER)
    for row in range(len(CANDIDATE_ORDER)):
        for column in range(len(CANDIDATE_ORDER)):
            value = float(matrix.iloc[row, column])
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    ax.set_title("Overlap of frozen reversal-associated gene sets")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Jaccard index")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"candidate_gene_overlap_heatmap.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def truncate(text: object, length: int = 68) -> str:
    value = str(text)
    return value if len(value) <= length else value[: length - 1] + "…"


def plot_pathway_heatmap(
    representative: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    subset = representative.loc[
        (representative["gene_set_scope"] == "all_selected")
        & representative["source"].isin(["GO:BP", "KEGG", "REAC"])
    ].copy()
    if subset.empty:
        return
    term_summary = (
        subset.groupby(["source", "native", "name"], as_index=False)
        .agg(min_fdr=("p_value", "min"), candidates=("candidate", "nunique"))
        .sort_values(["candidates", "min_fdr"], ascending=[False, True], kind="stable")
    )
    selected_terms = []
    for source in ["GO:BP", "KEGG", "REAC"]:
        selected_terms.extend(
            term_summary.loc[term_summary["source"] == source].head(5).to_dict("records")
        )
    if not selected_terms:
        return
    selected = pd.DataFrame(selected_terms)
    selected["term_key"] = selected["source"] + " | " + selected["native"]
    subset["term_key"] = subset["source"] + " | " + subset["native"]
    subset = subset.loc[subset["term_key"].isin(selected["term_key"])].copy()
    subset["minus_log10_fdr"] = -np.log10(
        np.clip(pd.to_numeric(subset["p_value"], errors="coerce"), 1e-300, 1.0)
    )
    matrix = subset.pivot_table(
        index="term_key",
        columns="candidate",
        values="minus_log10_fdr",
        aggfunc="max",
        fill_value=0.0,
    )
    ordered_keys = selected["term_key"].tolist()
    matrix = matrix.reindex(index=ordered_keys, columns=CANDIDATE_ORDER, fill_value=0.0)
    label_lookup = {
        row["term_key"]: f"{row['source']} | {truncate(row['name'])}"
        for row in selected_terms
    }
    displayed = np.clip(matrix.to_numpy(dtype=float), 0, 20)
    fig_height = max(6.5, 0.42 * len(matrix))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(displayed, cmap="YlOrRd", aspect="auto", vmin=0, vmax=20)
    ax.set_xticks(range(len(CANDIDATE_ORDER)), CANDIDATE_ORDER, rotation=45, ha="right")
    ax.set_yticks(
        range(len(matrix)),
        [label_lookup.get(key, key) for key in matrix.index],
        fontsize=8,
    )
    ax.set_title("Representative pathway enrichment of reversal-associated genes")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(r"$-\log_{10}$(FDR), capped at 20")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"reversal_pathway_enrichment_heatmap.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_networks(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    output_dir: Path,
    plot_top_n: int,
    dpi: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    axes_flat = axes.ravel()
    for panel, candidate in enumerate(MAIN_NETWORK_ORDER):
        ax = axes_flat[panel]
        physical_nodes = nodes.loc[
            (nodes["candidate"] == candidate) & (nodes["network_type"] == "physical")
        ].copy()
        physical_edges = edges.loc[
            (edges["candidate"] == candidate) & (edges["network_type"] == "physical")
        ].copy()
        network_type = "physical"
        if physical_edges.empty:
            physical_nodes = nodes.loc[
                (nodes["candidate"] == candidate)
                & (nodes["network_type"] == "functional")
            ].copy()
            physical_edges = edges.loc[
                (edges["candidate"] == candidate)
                & (edges["network_type"] == "functional")
            ].copy()
            network_type = "functional"
        selected_nodes = physical_nodes.sort_values(
            ["weighted_degree", "degree", "preferred_name"],
            ascending=[False, False, True],
            kind="stable",
        ).head(plot_top_n)
        node_names = selected_nodes["preferred_name"].astype(str).tolist()
        node_keys = {name.upper() for name in node_names}
        selected_edges = physical_edges.loc[
            physical_edges["preferredName_A"].astype(str).str.upper().isin(node_keys)
            & physical_edges["preferredName_B"].astype(str).str.upper().isin(node_keys)
        ]
        positions = force_layout(node_names, selected_edges, seed + panel)
        for row in selected_edges.itertuples(index=False):
            left = next(
                (name for name in node_names if name.upper() == str(row.preferredName_A).upper()),
                None,
            )
            right = next(
                (name for name in node_names if name.upper() == str(row.preferredName_B).upper()),
                None,
            )
            if left is None or right is None:
                continue
            x = [positions[left][0], positions[right][0]]
            y = [positions[left][1], positions[right][1]]
            ax.plot(
                x,
                y,
                color="#9E9E9E",
                alpha=0.35,
                linewidth=0.4 + 1.4 * float(row.score),
                zorder=1,
            )
        degree_lookup = selected_nodes.set_index(
            selected_nodes["preferred_name"].astype(str).str.upper()
        )
        for name in node_names:
            row = degree_lookup.loc[name.upper()]
            pattern = str(row.get("reversal_pattern", "mixed_or_context_dependent"))
            size = 45 + 18 * math.sqrt(max(float(row["degree"]), 0.0) + 1.0)
            ax.scatter(
                positions[name][0],
                positions[name][1],
                s=size,
                color=REVERSAL_COLORS.get(pattern, "#999999"),
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
        label_names = set(
            selected_nodes.sort_values(
                ["weighted_degree", "degree"], ascending=False
            ).head(12)["preferred_name"].astype(str)
        )
        for name in label_names:
            ax.text(
                positions[name][0],
                positions[name][1],
                name,
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=3,
            )
        ax.set_title(
            f"{candidate}\nSTRING {network_type}, nodes={len(node_names)}, edges={len(selected_edges)}"
        )
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.axis("off")
    axes_flat[-1].axis("off")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Disease-up / drug-down",
            markerfacecolor=REVERSAL_COLORS["disease_up_drug_down"],
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Disease-down / drug-up",
            markerfacecolor=REVERSAL_COLORS["disease_down_drug_up"],
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Context-dependent",
            markerfacecolor=REVERSAL_COLORS["mixed_or_context_dependent"],
            markersize=9,
        ),
    ]
    axes_flat[-1].legend(handles=legend, loc="center", frameon=False)
    fig.suptitle(
        "High-confidence interactions among frozen reversal-associated genes",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"reversal_associated_gene_networks.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.required_score <= 1_000:
        raise ValueError("--required-score must be between 0 and 1000")
    if not 0 < args.redundancy_jaccard <= 1:
        raise ValueError("--redundancy-jaccard must be in (0, 1]")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_api"
    raw_dir.mkdir(parents=True, exist_ok=True)

    consensus_path = args.input_dir / "candidate_reversal_gene_consensus.csv"
    freeze_path = args.input_dir / "frozen_candidate_manifest.csv"
    upstream_manifest_path = (
        args.input_dir / "frozen_candidate_biology_manifest.json"
    )
    consensus = pd.read_csv(consensus_path, dtype={"gene_id": str}, low_memory=False)
    freeze = pd.read_csv(freeze_path, low_memory=False)
    require_columns(
        consensus,
        [
            "candidate",
            "candidate_role",
            "gene_id",
            "selected_network_gene",
            "consensus_rank",
            "consensus_rank_score",
            "reversal_pattern",
            "disease_observed_cancers",
        ],
        "frozen candidate gene consensus",
    )
    observed_candidates = consensus["candidate"].drop_duplicates().tolist()
    if observed_candidates != CANDIDATE_ORDER:
        raise RuntimeError(
            f"Frozen gene candidates changed: {observed_candidates} != {CANDIDATE_ORDER}"
        )
    consensus["selected_network_gene"] = (
        consensus["selected_network_gene"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    selected_counts = (
        consensus.loc[consensus["selected_network_gene"]]
        .groupby("candidate")["gene_id"]
        .nunique()
        .reindex(CANDIDATE_ORDER, fill_value=0)
    )
    if (selected_counts == 0).any():
        missing = selected_counts.loc[selected_counts == 0].index.tolist()
        raise RuntimeError(f"Candidates without selected network genes: {missing}")

    background = unique_strings(
        consensus.loc[
            pd.to_numeric(
                consensus["disease_observed_cancers"], errors="coerce"
            ).gt(0),
            "gene_id",
        ]
    )
    sets = gene_sets(consensus)
    selected_union = unique_strings(
        gene
        for candidate in CANDIDATE_ORDER
        for gene in sets[(candidate, "all_selected")]
    )
    mapping, convert_metadata = map_gene_ids(selected_union, args, raw_dir)
    selected_with_mapping = consensus.loc[
        consensus["selected_network_gene"], ["candidate", "gene_id"]
    ].merge(mapping, on="gene_id", how="left", validate="many_to_one")
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
        failed = mapping_coverage.loc[
            mapping_coverage["mapping_fraction"] < 0.90
        ]
        raise RuntimeError(
            "STRING mapping coverage below 90%:\n"
            f"{failed.to_string(index=False)}"
        )

    enrichment, enrichment_metadata, enrichment_inventory = run_enrichment(
        sets, background, args, raw_dir
    )
    representative = representative_terms(
        enrichment,
        args.max_representative_terms,
        args.representative_min_term_size,
        args.representative_max_term_size,
        args.redundancy_jaccard,
    )
    edges, nodes, network_summary = run_string_networks(
        consensus, mapping, args, raw_dir
    )

    selected_sets = {
        candidate: set(sets[(candidate, "all_selected")])
        for candidate in CANDIDATE_ORDER
    }
    gene_overlap = pairwise_overlap(selected_sets, "genes")
    all_scope_enrichment = enrichment.loc[
        enrichment["gene_set_scope"] == "all_selected"
    ]
    pathway_sets = {
        candidate: set(
            all_scope_enrichment.loc[
                all_scope_enrichment["candidate"] == candidate, "native"
            ].astype(str)
        )
        for candidate in CANDIDATE_ORDER
    }
    pathway_overlap = pairwise_overlap(pathway_sets, "pathways")
    primary_control_overlap = gene_overlap.loc[
        gene_overlap["candidate_a"].isin(PRIMARY_CANDIDATES)
        & gene_overlap["candidate_b"].isin(REFERENCE_CONTROLS)
    ].copy()
    hub_genes = (
        nodes.loc[nodes["network_type"] == "physical"]
        .sort_values(
            ["candidate", "weighted_degree", "degree"],
            ascending=[True, False, False],
            kind="stable",
        )
        .groupby("candidate", as_index=False, group_keys=False)
        .head(20)
    )

    mapping.to_csv(args.output_dir / "gene_id_mapping.csv", index=False)
    mapping_coverage.to_csv(
        args.output_dir / "gene_id_mapping_coverage.csv", index=False
    )
    enrichment_inventory.to_csv(
        args.output_dir / "enrichment_query_inventory.csv", index=False
    )
    serialize_intersections(enrichment).to_csv(
        args.output_dir / "gprofiler_enrichment_full.csv", index=False
    )
    serialize_intersections(representative).to_csv(
        args.output_dir / "gprofiler_enrichment_representative.csv", index=False
    )
    edges.to_csv(args.output_dir / "string_network_edges.csv", index=False)
    nodes.to_csv(args.output_dir / "string_network_nodes.csv", index=False)
    network_summary.to_csv(
        args.output_dir / "string_network_summary.csv", index=False
    )
    hub_genes.to_csv(args.output_dir / "string_hub_genes.csv", index=False)
    gene_overlap.to_csv(args.output_dir / "candidate_gene_overlap.csv", index=False)
    pathway_overlap.to_csv(
        args.output_dir / "candidate_pathway_overlap.csv", index=False
    )
    primary_control_overlap.to_csv(
        args.output_dir / "primary_control_gene_overlap.csv", index=False
    )

    plot_overlap_heatmap(gene_overlap, args.output_dir, args.figure_dpi)
    plot_pathway_heatmap(representative, args.output_dir, args.figure_dpi)
    plot_networks(
        edges,
        nodes,
        args.output_dir,
        args.plot_top_n,
        args.figure_dpi,
        args.random_seed,
    )

    output_names = [
        "gene_id_mapping.csv",
        "gene_id_mapping_coverage.csv",
        "enrichment_query_inventory.csv",
        "gprofiler_enrichment_full.csv",
        "gprofiler_enrichment_representative.csv",
        "string_network_edges.csv",
        "string_network_nodes.csv",
        "string_network_summary.csv",
        "string_hub_genes.csv",
        "candidate_gene_overlap.csv",
        "candidate_pathway_overlap.csv",
        "primary_control_gene_overlap.csv",
        "candidate_gene_overlap_heatmap.png",
        "candidate_gene_overlap_heatmap.pdf",
        "reversal_pathway_enrichment_heatmap.png",
        "reversal_pathway_enrichment_heatmap.pdf",
        "reversal_associated_gene_networks.png",
        "reversal_associated_gene_networks.pdf",
        "network_pathway_manifest.json",
        "audit.log",
        "raw_api/*.json",
    ]
    gprofiler_versions = []
    for metadata in [*convert_metadata, *enrichment_metadata]:
        for key in ["data_version", "version", "timestamp"]:
            if key in metadata:
                gprofiler_versions.append({key: metadata[key]})
    manifest = {
        "analysis_stage": "frozen_reversal_gene_networks_and_pathways",
        "candidate_order": CANDIDATE_ORDER,
        "primary_candidates": PRIMARY_CANDIDATES,
        "reference_controls": REFERENCE_CONTROLS,
        "gene_set_scopes": GENE_SET_SCOPES,
        "selected_genes_per_candidate": selected_counts.astype(int).to_dict(),
        "custom_background_genes": len(background),
        "enrichment_sources": ENRICHMENT_SOURCES,
        "enrichment_method": (
            "g:Profiler g:GOSt one-tailed hypergeometric over-representation "
            "analysis with custom measured gene background and Benjamini-Hochberg FDR."
        ),
        "gprofiler_api": args.gprofiler_base,
        "gprofiler_response_versions": gprofiler_versions,
        "representative_term_policy": {
            "max_terms_per_candidate_scope_source": args.max_representative_terms,
            "term_size_min": args.representative_min_term_size,
            "term_size_max": args.representative_max_term_size,
            "intersection_jaccard_exclusion_threshold": args.redundancy_jaccard,
            "note": "Full unfiltered significant results are always exported.",
        },
        "string_api": args.string_base,
        "string_version": "12.0 (version-pinned endpoint)",
        "string_species": 9606,
        "string_required_score": args.required_score,
        "string_network_types": ["physical", "functional"],
        "string_added_nodes": 0,
        "evidence_note": (
            "Networks connect frozen reversal-associated genes. They are not "
            "drug-target networks and do not establish direct binding, efficacy, "
            "or causal mechanism. Functional networks are sensitivity analyses; "
            "physical high-confidence networks are primary."
        ),
        "input_sha256": {
            "candidate_reversal_gene_consensus": sha256_file(consensus_path),
            "frozen_candidate_manifest": sha256_file(freeze_path),
            "upstream_manifest": (
                sha256_file(upstream_manifest_path)
                if upstream_manifest_path.is_file()
                else None
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "random_seed": args.random_seed,
        "random_seed_scope": "deterministic network figure layout only",
        "outputs": output_names,
    }
    (args.output_dir / "network_pathway_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    enrichment_display = (
        enrichment.loc[enrichment["gene_set_scope"] == "all_selected"]
        .groupby(["candidate", "source"], as_index=False)
        .size()
        .rename(columns={"size": "significant_terms_fdr_lt_0_05"})
    )
    sections = [
        "===== GENE-ID MAPPING COVERAGE =====",
        mapping_coverage.to_string(index=False),
        "",
        "===== SIGNIFICANT ENRICHMENT TERMS: ALL SELECTED GENES =====",
        (
            enrichment_display.to_string(index=False)
            if len(enrichment_display)
            else "No significant terms"
        ),
        "",
        "===== STRING NETWORK SUMMARY =====",
        network_summary.round(6).to_string(index=False),
        "",
        "===== PRIMARY-CONTROL GENE-SET OVERLAP =====",
        primary_control_overlap[
            ["candidate_a", "candidate_b", "shared_genes", "jaccard"]
        ].round(6).to_string(index=False),
        "",
        "REVERSAL-ASSOCIATED GENE NETWORK/PATHWAY ANALYSIS: PASSED",
    ]
    audit_text = "\n".join(sections) + "\n"
    (args.output_dir / "audit.log").write_text(audit_text, encoding="utf-8")
    print(audit_text, end="")


if __name__ == "__main__":
    main()
