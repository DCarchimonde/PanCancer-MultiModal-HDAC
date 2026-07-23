#!/usr/bin/env python3
"""Regenerate main Figure 7 and Supplementary Figure S4 with safe labels.

The renderer uses only the frozen STRING response caches and the frozen
candidate-gene request bundle already stored in the repository.  It does not
query an API or recompute any biological result.  The top-weighted node
subsets, induced physical edges, reversal-pattern colours, and hub rankings
are preserved; only component packing, typography, and label placement change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CANDIDATES = [
    "Mocetinostat",
    "NCH-51",
    "TC-H-106",
    "Belinostat",
    "Entinostat",
    "Vorinostat",
]
CACHE_STEMS = {
    "Mocetinostat": "mocetinostat",
    "NCH-51": "nch_51",
    "TC-H-106": "tc_h_106",
    "Belinostat": "belinostat",
    "Entinostat": "entinostat",
    "Vorinostat": "vorinostat",
}
REVERSAL_COLORS = {
    "disease_up_drug_down": "#D84A3A",
    "disease_down_drug_up": "#3E78B2",
    "mixed_or_context_dependent": "#8D9196",
}
DISPLAY_LABELS = {
    "disease_up_drug_down": "Disease-up / drug-down",
    "disease_down_drug_up": "Disease-down / drug-up",
    "mixed_or_context_dependent": "Context-dependent",
}
SEED = 20260719
PDF_METADATA = {
    "Creator": "Matplotlib collision-aware frozen-network renderer",
    "Keywords": "network-layout-qa-v2",
}


@dataclass
class CandidateGraph:
    candidate: str
    nodes: pd.DataFrame
    edges: pd.DataFrame


@dataclass
class PanelState:
    axis: plt.Axes
    annotations: list[plt.Annotation]
    node_positions: dict[str, np.ndarray]
    node_sizes: dict[str, float]
    candidate: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root. Auto-detected from this script when omitted.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for parent in [start.resolve(), *start.resolve().parents]:
        if (
            (parent / "results/revision/reversal_gene_networks_pathways").is_dir()
            and (parent / "revision/rework_20260720").is_dir()
        ):
            return parent
    raise FileNotFoundError(
        "Could not locate the repository root containing the frozen network caches."
    )


def load_candidate_graphs(repo_root: Path) -> dict[str, CandidateGraph]:
    network_dir = repo_root / "results/revision/reversal_gene_networks_pathways"
    raw_dir = network_dir / "raw_api"
    bundle = json.loads((network_dir / "network_api_request_bundle.json").read_text())
    records = pd.DataFrame(bundle["selected_consensus_records"])

    converted: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("gconvert_batch_*.json")):
        payload = json.loads(path.read_text())
        converted.extend(payload["response"]["result"])
    conversion = pd.DataFrame(converted)
    if conversion.empty:
        raise RuntimeError("Frozen g:Profiler conversion cache is empty.")
    conversion["incoming"] = conversion["incoming"].astype(str)
    gene_to_name = (
        conversion.drop_duplicates("incoming")
        .set_index("incoming")["name"]
        .astype(str)
    )
    records["preferred_name"] = records["gene_id"].astype(str).map(gene_to_name)
    if records["preferred_name"].isna().any():
        missing = records.loc[records["preferred_name"].isna(), "gene_id"].tolist()
        raise RuntimeError(f"Unmapped frozen gene identifiers: {missing[:10]}")

    graphs: dict[str, CandidateGraph] = {}
    for candidate in CANDIDATES:
        cache = raw_dir / f"string_{CACHE_STEMS[candidate]}_physical.json"
        payload = json.loads(cache.read_text())
        edges = pd.read_csv(io.StringIO(payload["response_text"]), sep="\t")
        required = {"preferredName_A", "preferredName_B", "score"}
        if not required.issubset(edges.columns):
            raise RuntimeError(f"{cache} lacks required STRING fields.")
        edges = edges.copy()
        edges["preferredName_A"] = edges["preferredName_A"].astype(str)
        edges["preferredName_B"] = edges["preferredName_B"].astype(str)
        edges["score"] = pd.to_numeric(edges["score"], errors="raise")

        endpoints = pd.concat(
            [
                edges[["preferredName_A", "score"]].rename(
                    columns={"preferredName_A": "preferred_name"}
                ),
                edges[["preferredName_B", "score"]].rename(
                    columns={"preferredName_B": "preferred_name"}
                ),
            ],
            ignore_index=True,
        )
        metrics = (
            endpoints.groupby("preferred_name", as_index=False)
            .agg(degree=("score", "size"), weighted_degree=("score", "sum"))
        )
        metadata = records.loc[
            records["candidate"] == candidate,
            ["preferred_name", "reversal_pattern", "consensus_rank"],
        ].drop_duplicates("preferred_name")
        nodes = metrics.merge(
            metadata, on="preferred_name", how="left", validate="one_to_one"
        )
        if nodes["reversal_pattern"].isna().any():
            missing = nodes.loc[
                nodes["reversal_pattern"].isna(), "preferred_name"
            ].tolist()
            raise RuntimeError(f"Missing reversal pattern for {candidate}: {missing}")
        nodes = nodes.sort_values(
            ["weighted_degree", "degree", "preferred_name"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        graphs[candidate] = CandidateGraph(candidate, nodes, edges)
    return graphs


def induced_subset(
    graph: CandidateGraph, node_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_nodes = graph.nodes.head(node_limit).copy()
    selected_names = set(selected_nodes["preferred_name"].astype(str))
    selected_edges = graph.edges.loc[
        graph.edges["preferredName_A"].isin(selected_names)
        & graph.edges["preferredName_B"].isin(selected_names)
    ].copy()
    return selected_nodes, selected_edges


def connected_components(
    node_names: Iterable[str], edges: pd.DataFrame
) -> list[list[str]]:
    names = list(node_names)
    adjacency = {name: set() for name in names}
    for row in edges.itertuples(index=False):
        left = str(row.preferredName_A)
        right = str(row.preferredName_B)
        if left in adjacency and right in adjacency and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen: set[str] = set()
    components: list[list[str]] = []
    for name in names:
        if name in seen:
            continue
        stack = [name]
        seen.add(name)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def force_layout(
    node_names: list[str], edges: pd.DataFrame, seed: int
) -> dict[str, np.ndarray]:
    count = len(node_names)
    if count == 1:
        return {node_names[0]: np.zeros(2)}
    if count == 2:
        return {
            node_names[0]: np.array([-0.60, 0.0]),
            node_names[1]: np.array([0.60, 0.0]),
        }

    rng = np.random.default_rng(seed)
    positions = rng.normal(0, 0.45, size=(count, 2))
    index = {name: position for position, name in enumerate(node_names)}
    pairs = [
        (index[str(row.preferredName_A)], index[str(row.preferredName_B)])
        for row in edges.itertuples(index=False)
        if str(row.preferredName_A) in index
        and str(row.preferredName_B) in index
        and str(row.preferredName_A) != str(row.preferredName_B)
    ]
    ideal = 1.25 / math.sqrt(count)
    temperature = 0.16
    for _ in range(420):
        delta = positions[:, None, :] - positions[None, :, :]
        distance2 = np.sum(delta * delta, axis=2) + np.eye(count)
        distance = np.sqrt(distance2)
        repulsion = (
            (ideal * ideal / distance2)[:, :, None]
            * (delta / distance[:, :, None])
        )
        displacement = repulsion.sum(axis=1)
        for left, right in pairs:
            vector = positions[left] - positions[right]
            length = max(float(np.linalg.norm(vector)), 1e-4)
            attraction = (length * length / ideal) * (vector / length)
            displacement[left] -= attraction
            displacement[right] += attraction
        displacement -= 0.10 * positions
        norms = np.linalg.norm(displacement, axis=1)
        step = displacement / np.maximum(norms[:, None], 1e-9)
        step *= np.minimum(norms, temperature)[:, None]
        positions += step
        temperature *= 0.991
    positions -= positions.mean(axis=0)
    scale = np.max(np.abs(positions), axis=0)
    scale[scale < 1e-6] = 1.0
    positions /= scale
    return {name: positions[index[name]] for name in node_names}


def split_rectangles(
    weighted_components: list[tuple[list[str], float]],
    rectangle: tuple[float, float, float, float],
) -> list[tuple[list[str], tuple[float, float, float, float]]]:
    if len(weighted_components) == 1:
        return [(weighted_components[0][0], rectangle)]

    total = sum(weight for _, weight in weighted_components)
    cumulative = 0.0
    best_index = 1
    best_gap = float("inf")
    for index in range(1, len(weighted_components)):
        cumulative += weighted_components[index - 1][1]
        gap = abs(cumulative - total / 2.0)
        if gap < best_gap:
            best_gap = gap
            best_index = index
    first = weighted_components[:best_index]
    second = weighted_components[best_index:]
    first_weight = sum(weight for _, weight in first)
    ratio = first_weight / total
    x, y, width, height = rectangle
    if width >= height:
        first_rect = (x, y, width * ratio, height)
        second_rect = (x + width * ratio, y, width * (1 - ratio), height)
    else:
        first_rect = (x, y, width, height * ratio)
        second_rect = (x, y + height * ratio, width, height * (1 - ratio))
    return split_rectangles(first, first_rect) + split_rectangles(second, second_rect)


def packed_component_layout(
    node_names: list[str],
    edges: pd.DataFrame,
    seed: int,
    bounds: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    components = connected_components(node_names, edges)
    weighted = [
        (component, max(1.0, float(len(component)) ** 0.86))
        for component in components
    ]
    rectangles = split_rectangles(weighted, bounds)
    final: dict[str, np.ndarray] = {}
    component_centres: dict[str, np.ndarray] = {}

    for component_index, (component, rectangle) in enumerate(rectangles):
        component_set = set(component)
        component_edges = edges.loc[
            edges["preferredName_A"].isin(component_set)
            & edges["preferredName_B"].isin(component_set)
        ]
        local = force_layout(component, component_edges, seed + component_index * 101)
        x, y, width, height = rectangle
        padding = min(0.12, 0.09 * min(width, height))
        usable_width = max(width - 2 * padding, width * 0.66)
        usable_height = max(height - 2 * padding, height * 0.66)
        centre = np.array([x + width / 2.0, y + height / 2.0])
        for name in component:
            point = local[name]
            final[name] = centre + np.array(
                [point[0] * usable_width * 0.43, point[1] * usable_height * 0.43]
            )
            component_centres[name] = centre
    return final, component_centres


def initial_label_position(
    node_position: np.ndarray,
    component_centre: np.ndarray,
    index: int,
) -> np.ndarray:
    direction = node_position - component_centre
    norm = float(np.linalg.norm(direction))
    if norm < 0.06:
        angle = 2 * math.pi * (index + 0.5) / 11.0
        direction = np.array([math.cos(angle), math.sin(angle)])
    else:
        direction /= norm
    return node_position + direction * 0.105


def add_network_panel(
    axis: plt.Axes,
    graph: CandidateGraph,
    node_limit: int,
    label_limit: int,
    seed: int,
    label_fontsize: float,
    bounds: tuple[float, float, float, float],
    node_size_scale: float = 1.0,
) -> tuple[PanelState, dict[str, object]]:
    nodes, edges = induced_subset(graph, node_limit)
    node_names = nodes["preferred_name"].astype(str).tolist()
    positions, component_centres = packed_component_layout(
        node_names, edges, seed, bounds
    )
    node_lookup = nodes.set_index("preferred_name")

    for row in edges.itertuples(index=False):
        left = str(row.preferredName_A)
        right = str(row.preferredName_B)
        axis.plot(
            [positions[left][0], positions[right][0]],
            [positions[left][1], positions[right][1]],
            color="#AAB3BE",
            alpha=0.55,
            linewidth=0.35 + 1.35 * float(row.score),
            solid_capstyle="round",
            zorder=1,
        )

    node_sizes: dict[str, float] = {}
    for name in node_names:
        row = node_lookup.loc[name]
        weighted_degree = max(float(row["weighted_degree"]), 0.0)
        size = node_size_scale * (
            28.0 + 34.0 * math.sqrt(weighted_degree + 0.25)
        )
        node_sizes[name] = size
        pattern = str(row["reversal_pattern"])
        axis.scatter(
            positions[name][0],
            positions[name][1],
            s=size,
            color=REVERSAL_COLORS.get(pattern, REVERSAL_COLORS["mixed_or_context_dependent"]),
            edgecolors="white",
            linewidths=0.75,
            zorder=3,
        )

    label_names = (
        nodes.sort_values(
            ["weighted_degree", "degree", "preferred_name"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(label_limit)["preferred_name"]
        .astype(str)
        .tolist()
    )
    annotations: list[plt.Annotation] = []
    for index, name in enumerate(label_names):
        label_position = initial_label_position(
            positions[name], component_centres[name], index
        )
        annotation = axis.annotate(
            name,
            xy=positions[name],
            xytext=label_position,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            weight="semibold",
            color="#18212A",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#7D8792",
                "linewidth": 0.65,
                "shrinkA": 2.0,
                "shrinkB": 3.5,
            },
            zorder=5,
        )
        annotations.append(annotation)

    x, y, width, height = bounds
    axis.set_xlim(x, x + width)
    axis.set_ylim(y, y + height)
    axis.set_facecolor("#FAFBFC")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#D9DEE5")
        spine.set_linewidth(0.8)

    state = PanelState(
        axis=axis,
        annotations=annotations,
        node_positions=positions,
        node_sizes=node_sizes,
        candidate=graph.candidate,
    )
    inventory = {
        "candidate": graph.candidate,
        "displayed_nodes": len(node_names),
        "displayed_edges": len(edges),
        "labelled_hubs": len(label_names),
        "selection": f"top {min(node_limit, len(graph.nodes))} physical-network nodes by weighted degree",
    }
    return state, inventory


def _text_patch_bbox(
    annotation: plt.Annotation, renderer: matplotlib.backend_bases.RendererBase
) -> matplotlib.transforms.Bbox:
    patch = annotation.get_bbox_patch()
    if patch is None:
        return annotation.get_window_extent(renderer)
    return patch.get_window_extent(renderer)


def relax_panel_labels(
    figure: plt.Figure,
    state: PanelState,
    *,
    iterations: int = 700,
) -> dict[str, int]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_bbox = state.axis.get_window_extent(renderer)
    count = len(state.annotations)
    if count == 0:
        return {"label_label_overlaps": 0, "label_node_overlaps": 0}

    boxes = [_text_patch_bbox(annotation, renderer) for annotation in state.annotations]
    widths = np.array([box.width for box in boxes], dtype=float)
    heights = np.array([box.height for box in boxes], dtype=float)
    centres = state.axis.transData.transform(
        np.array([annotation.get_position() for annotation in state.annotations])
    )
    anchors = state.axis.transData.transform(
        np.array([annotation.xy for annotation in state.annotations])
    )
    node_names = list(state.node_positions)
    node_points = state.axis.transData.transform(
        np.array([state.node_positions[name] for name in node_names])
    )
    node_radii = np.array(
        [max(4.0, math.sqrt(state.node_sizes[name]) * figure.dpi / 120.0) for name in node_names]
    )

    for iteration in range(iterations):
        forces = (anchors - centres) * (0.005 if iteration < 240 else 0.0)

        for left in range(count):
            for right in range(left + 1, count):
                delta = centres[left] - centres[right]
                overlap_x = (
                    (widths[left] + widths[right]) / 2.0 + 4.0 - abs(delta[0])
                )
                overlap_y = (
                    (heights[left] + heights[right]) / 2.0 + 3.0 - abs(delta[1])
                )
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if overlap_x < overlap_y:
                    sign = 1.0 if delta[0] >= 0 else -1.0
                    push = sign * (overlap_x / 2.0 + 0.8)
                    forces[left, 0] += push
                    forces[right, 0] -= push
                else:
                    sign = 1.0 if delta[1] >= 0 else -1.0
                    push = sign * (overlap_y / 2.0 + 0.8)
                    forces[left, 1] += push
                    forces[right, 1] -= push

        for label_index in range(count):
            half_width = widths[label_index] / 2.0 + 2.0
            half_height = heights[label_index] / 2.0 + 1.5
            for point, radius in zip(node_points, node_radii):
                delta = centres[label_index] - point
                overlap_x = half_width + radius - abs(delta[0])
                overlap_y = half_height + radius - abs(delta[1])
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if overlap_x < overlap_y:
                    sign = 1.0 if delta[0] >= 0 else -1.0
                    forces[label_index, 0] += sign * 1.5 * (overlap_x + 1.2)
                else:
                    sign = 1.0 if delta[1] >= 0 else -1.0
                    forces[label_index, 1] += sign * 1.5 * (overlap_y + 1.2)

        max_step = 8.0 if iteration < 250 else 4.0
        forces = np.clip(forces, -max_step, max_step)
        centres += forces
        centres[:, 0] = np.clip(
            centres[:, 0],
            axis_bbox.x0 + widths / 2.0 + 4.0,
            axis_bbox.x1 - widths / 2.0 - 4.0,
        )
        centres[:, 1] = np.clip(
            centres[:, 1],
            axis_bbox.y0 + heights / 2.0 + 4.0,
            axis_bbox.y1 - heights / 2.0 - 4.0,
        )

    def placement_score(
        label_index: int,
        candidate_centre: np.ndarray,
        current_centres: np.ndarray,
    ) -> float:
        half_width = widths[label_index] / 2.0
        half_height = heights[label_index] / 2.0
        left = candidate_centre[0] - half_width
        right = candidate_centre[0] + half_width
        bottom = candidate_centre[1] - half_height
        top = candidate_centre[1] + half_height
        if (
            left < axis_bbox.x0 + 4.0
            or right > axis_bbox.x1 - 4.0
            or bottom < axis_bbox.y0 + 4.0
            or top > axis_bbox.y1 - 4.0
        ):
            return 1e12
        score = 0.015 * float(np.sum((candidate_centre - anchors[label_index]) ** 2))
        for other in range(count):
            if other == label_index:
                continue
            other_left = current_centres[other, 0] - widths[other] / 2.0
            other_right = current_centres[other, 0] + widths[other] / 2.0
            other_bottom = current_centres[other, 1] - heights[other] / 2.0
            other_top = current_centres[other, 1] + heights[other] / 2.0
            overlap_width = min(right + 3.0, other_right) - max(
                left - 3.0, other_left
            )
            overlap_height = min(top + 2.0, other_top) - max(
                bottom - 2.0, other_bottom
            )
            if overlap_width > 0 and overlap_height > 0:
                score += 1e7 + overlap_width * overlap_height * 1e4
        for point, radius in zip(node_points, node_radii):
            overlap_width = min(right + radius, point[0]) - max(
                left - radius, point[0]
            )
            overlap_height = min(top + radius, point[1]) - max(
                bottom - radius, point[1]
            )
            if overlap_width >= 0 and overlap_height >= 0:
                score += 1e6 + (overlap_width + 1.0) * (overlap_height + 1.0) * 1e3
        return score

    # A deterministic radial search removes the rare residual collision left
    # by continuous repulsion in very dense S4 modules.
    radii = [18.0, 26.0, 36.0, 48.0, 62.0, 78.0, 96.0, 118.0]
    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    for _ in range(5):
        changed = False
        for label_index in range(count):
            candidates = [centres[label_index].copy()]
            candidates.extend(
                anchors[label_index]
                + radius * np.array([math.cos(angle), math.sin(angle)])
                for radius in radii
                for angle in angles
            )
            scores = [
                placement_score(label_index, candidate, centres)
                for candidate in candidates
            ]
            best = candidates[int(np.argmin(scores))]
            if float(np.linalg.norm(best - centres[label_index])) > 0.5:
                centres[label_index] = best
                changed = True
        if not changed:
            break

    inverse = state.axis.transData.inverted()
    for annotation, centre in zip(state.annotations, centres):
        annotation.set_position(inverse.transform(centre))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    final_boxes = [_text_patch_bbox(annotation, renderer) for annotation in state.annotations]

    label_overlaps = 0
    for left in range(count):
        for right in range(left + 1, count):
            intersection = matplotlib.transforms.Bbox.intersection(
                final_boxes[left], final_boxes[right]
            )
            if intersection is not None and intersection.width > 0.5 and intersection.height > 0.5:
                label_overlaps += 1

    node_overlaps = 0
    for box in final_boxes:
        for point, radius in zip(node_points, node_radii):
            if (
                box.x0 - radius < point[0] < box.x1 + radius
                and box.y0 - radius < point[1] < box.y1 + radius
            ):
                node_overlaps += 1
    return {
        "label_label_overlaps": label_overlaps,
        "label_node_overlaps": node_overlaps,
    }


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REVERSAL_COLORS[key],
            markeredgecolor="white",
            markersize=8.5,
            label=DISPLAY_LABELS[key],
        )
        for key in [
            "disease_up_drug_down",
            "disease_down_drug_up",
            "mixed_or_context_dependent",
        ]
    ]


def build_main_figure(
    graphs: dict[str, CandidateGraph],
    destination: Path,
    dpi: int,
) -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    bounds = (-1.0, -0.78, 2.0, 1.56)
    panels: list[PanelState] = []
    inventory: list[dict[str, object]] = []
    for index, (axis, candidate) in enumerate(zip(axes, CANDIDATES[:3])):
        state, item = add_network_panel(
            axis,
            graphs[candidate],
            node_limit=25,
            label_limit=7,
            seed=SEED + index * 1000,
            label_fontsize=6.8,
            bounds=bounds,
            node_size_scale=0.76,
        )
        axis.set_title(
            f"{candidate}\n"
            f"25 nodes · {item['displayed_edges']} edges",
            fontsize=7.5,
            weight="bold",
            color="#17202A",
            pad=8,
        )
        panels.append(state)
        inventory.append(item)

    figure.suptitle(
        "High-confidence physical STRING networks of prioritized candidates",
        fontsize=10.4,
        weight="bold",
        color="#111820",
        y=0.982,
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.7,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=1.1,
        handletextpad=0.3,
    )
    figure.subplots_adjust(
        left=0.018,
        right=0.992,
        top=0.74,
        bottom=0.18,
        wspace=0.065,
    )
    qa = [relax_panel_labels(figure, panel) for panel in panels]
    if any(result["label_label_overlaps"] for result in qa):
        raise RuntimeError(f"Figure 7 label collision remained after relaxation: {qa}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, metadata=PDF_METADATA)
    plt.close(figure)
    return inventory, qa


def draw_interpretation_key(axis: plt.Axes, page_label: str) -> None:
    axis.axis("off")
    axis.text(
        0.02,
        0.92,
        "Interpretation key",
        transform=axis.transAxes,
        fontsize=11.3,
        weight="bold",
        color="#17202A",
        va="top",
    )
    y = 0.74
    for key in [
        "disease_up_drug_down",
        "disease_down_drug_up",
        "mixed_or_context_dependent",
    ]:
        axis.scatter(
            0.055,
            y,
            s=90,
            color=REVERSAL_COLORS[key],
            edgecolors="white",
            linewidths=0.8,
            transform=axis.transAxes,
            clip_on=False,
        )
        axis.text(
            0.105,
            y,
            DISPLAY_LABELS[key],
            transform=axis.transAxes,
            fontsize=8.6,
            va="center",
            color="#222B33",
        )
        y -= 0.115
    axis.text(
        0.02,
        0.365,
        "Node area ∝ physical-network weighted degree\n"
        "Edge width ∝ STRING physical score\n"
        "Labels identify the nine leading hubs",
        transform=axis.transAxes,
        fontsize=8.0,
        va="top",
        color="#27323C",
        linespacing=1.35,
    )
    axis.text(
        0.02,
        0.205,
        "Layout note",
        transform=axis.transAxes,
        fontsize=8.5,
        weight="bold",
        va="top",
        color="#17202A",
    )
    axis.text(
        0.02,
        0.155,
        "Components are packed separately for legibility.\n"
        "Inter-component distance has no biological meaning.\n"
        "Singletons have no retained within-subset edge.",
        transform=axis.transAxes,
        fontsize=7.2,
        va="top",
        color="#4A5661",
        linespacing=1.28,
    )
def build_supplementary_figure(
    graphs: dict[str, CandidateGraph],
    destination: Path,
    dpi: int,
) -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    all_qa: list[dict[str, int]] = []
    bounds = (-1.0, -0.72, 2.0, 1.44)
    with PdfPages(destination, metadata=PDF_METADATA) as pdf:
        pages = [
            (CANDIDATES[:3], "A  Prioritized candidates"),
            (CANDIDATES[3:], "B  HDAC controls"),
        ]
        for page_index, (page_candidates, page_label) in enumerate(pages):
            figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
            axes_flat = axes.flatten()
            panels: list[PanelState] = []
            for panel_index, candidate in enumerate(page_candidates):
                state, item = add_network_panel(
                    axes_flat[panel_index],
                    graphs[candidate],
                    node_limit=45,
                    label_limit=9,
                    seed=SEED + page_index * 5000 + panel_index * 1000,
                    label_fontsize=7.6,
                    bounds=bounds,
                )
                axes_flat[panel_index].set_title(
                    f"{candidate}\n"
                    f"{item['displayed_nodes']} selected nodes · "
                    f"{item['displayed_edges']} within-subset edges",
                    fontsize=9.5,
                    weight="bold",
                    color="#17202A",
                    pad=7,
                )
                panels.append(state)
                inventory.append(item)
            draw_interpretation_key(axes_flat[3], page_label)
            figure.suptitle(
                "Expanded high-confidence physical STRING networks",
                fontsize=13.2,
                weight="bold",
                color="#111820",
                y=0.985,
            )
            figure.text(
                0.5,
                0.927,
                page_label,
                ha="center",
                va="center",
                fontsize=9.6,
                weight="semibold",
                color="#4A5661",
            )
            figure.subplots_adjust(
                left=0.035,
                right=0.985,
                top=0.855,
                bottom=0.045,
                wspace=0.09,
                hspace=0.22,
            )
            page_qa = [relax_panel_labels(figure, panel) for panel in panels]
            if any(result["label_label_overlaps"] for result in page_qa):
                raise RuntimeError(
                    f"Figure S4 page {page_index + 1} label collision remained: {page_qa}"
                )
            all_qa.extend(page_qa)
            pdf.savefig(figure, dpi=dpi)
            plt.close(figure)
    return inventory, all_qa


def main() -> None:
    args = parse_args()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root
        else find_repo_root(Path(__file__).parent)
    )
    graphs = load_candidate_graphs(repo_root)
    source_root = repo_root / "revision/rework_20260720"
    main_path = (
        source_root / "manuscript/figures/primary_candidate_physical_networks.pdf"
    )
    supplementary_path = (
        source_root
        / "supplement/figures/supplementary_expanded_physical_networks.pdf"
    )
    main_inventory, main_qa = build_main_figure(graphs, main_path, args.dpi)
    supplementary_inventory, supplementary_qa = build_supplementary_figure(
        graphs, supplementary_path, args.dpi
    )
    summary = {
        "main_figure": str(main_path.relative_to(repo_root)),
        "supplementary_figure": str(supplementary_path.relative_to(repo_root)),
        "main_inventory": main_inventory,
        "supplementary_inventory": supplementary_inventory,
        "main_qa": main_qa,
        "supplementary_qa": supplementary_qa,
        "data_policy": "frozen cache only; no API calls; no result recomputation",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
