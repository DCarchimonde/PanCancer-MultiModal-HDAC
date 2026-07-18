#!/usr/bin/env python3
"""Run the frozen 4LXZ AutoDock4Zn control/candidate docking panel.

This script is intentionally blocked unless the 4LXZ redocking gate has
already passed.  It reuses that pre-specified receptor, box, seeds, search
depth, and all-seed pose-consensus rule.  The panel and the designed negative
control are declared in this source before any candidate score is observed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from e3_ad4zn_support import sha256_file, write_ad4zn_gpf


DEFAULT_ENV_DIR = Path(".venv/e3_docking")
DEFAULT_REDOCK_DIR = Path("results/revision/docking_controls/redocking_4lxz")
DEFAULT_OUTPUT_DIR = Path("results/revision/docking_controls/controlled_panel_4lxz")
SPACING = 0.375
CONFORMER_SEED = 20260718
CONSENSUS_TOP_K = 5
CLUSTER_RMSD_A = 2.0

# Frozen before candidate docking.  RG2833 is prediction-only and the
# Tianeptinaline/BG-1010 identity-conflict row is deliberately absent.
PANEL: tuple[dict[str, str], ...] = (
    {
        "compound": "Vorinostat",
        "role": "crystallographic_positive_control",
        "smiles": "ONC(=O)CCCCCCC(=O)Nc1ccccc1",
        "zbg_class": "hydroxamate",
        "input_policy": "reuse_passed_SHH_redocking",
    },
    {
        "compound": "Vorinostat_O_methyl_decoy",
        "role": "designed_zinc_chelation_sensitivity_control",
        "smiles": "CONC(=O)CCCCCCC(=O)Nc1ccccc1",
        "zbg_class": "O_methyl_hydroxamate_decoy",
        "input_policy": "deterministic_RDKit_conformer",
    },
    {
        "compound": "Mocetinostat",
        "role": "core_candidate",
        "smiles": "Nc1ccccc1NC(=O)c1ccc(CNc2nccc(n2)-c2cccnc2)cc1",
        "zbg_class": "ortho_amino_benzamide",
        "input_policy": "deterministic_RDKit_conformer",
    },
    {
        "compound": "NCH-51",
        "role": "secondary_candidate",
        "smiles": "CC(C)C(=O)SCCCCCCC(=O)Nc1nc(cs1)-c1ccccc1",
        "zbg_class": "thioester_thiazolyl_amide_parent",
        "input_policy": "deterministic_RDKit_conformer",
    },
    {
        "compound": "TC-H-106",
        "role": "original_method_sensitive_candidate",
        "smiles": "Cc1ccc(NC(=O)CCCCCC(=O)Nc2ccccc2N)cc1",
        "zbg_class": "ortho_amino_benzamide",
        "input_policy": "deterministic_RDKit_conformer",
    },
    {
        "compound": "Entinostat",
        "role": "reference_control",
        "smiles": "Nc1ccccc1NC(=O)c1ccc(CNC(=O)OCc2cccnc2)cc1",
        "zbg_class": "ortho_amino_benzamide",
        "input_policy": "deterministic_RDKit_conformer",
    },
    {
        "compound": "Belinostat",
        "role": "class_support_control",
        "smiles": "ONC(=O)C=Cc1cccc(c1)S(=O)(=O)Nc1ccccc1",
        "zbg_class": "hydroxamate",
        "input_policy": "deterministic_RDKit_conformer",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--redocking-dir", type=Path, default=DEFAULT_REDOCK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cpu",
        type=int,
        help="CPU threads per Vina run; defaults to the passed redocking protocol value.",
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> str:
    print("+ " + " ".join(command), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command={' '.join(command)}\nruntime_seconds={elapsed:.6f}\n"
        + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}\n"
            + completed.stdout[-4000:]
        )
    return completed.stdout


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required controlled-docking input(s): {missing}")


def load_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise RuntimeError("Candidate docking is blocked: 4LXZ redocking gate is not passed")
    if gate.get("primary_protocol") != "AutoDock4Zn seed-consensus":
        raise RuntimeError("Unexpected redocking primary protocol")
    seeds = gate.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Invalid frozen redocking seeds: {seeds}")
    selection_rule = str(gate.get("selection_rule", ""))
    expected_fragments = (
        f"top {CONSENSUS_TOP_K} poses per seed",
        f"{CLUSTER_RMSD_A:.1f} Å",
        "require all-seed coverage",
    )
    if not all(fragment in selection_rule for fragment in expected_fragments):
        raise RuntimeError(
            "The redocking consensus rule does not match the frozen candidate-panel rule"
        )
    return gate


def deterministic_conformer(smiles: str, destination: Path, compound: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError(f"Invalid frozen SMILES for {compound}: {smiles}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = CONFORMER_SEED
    parameters.useRandomCoords = False
    status = AllChem.EmbedMolecule(molecule, parameters)
    if status != 0:
        parameters.useRandomCoords = True
        status = AllChem.EmbedMolecule(molecule, parameters)
    if status != 0:
        raise RuntimeError(f"RDKit could not generate a 3D conformer for {compound}")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        optimization = "MMFF94"
        optimization_status = int(AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000))
    else:
        optimization = "UFF"
        optimization_status = int(AllChem.UFFOptimizeMolecule(molecule, maxIters=1000))
    molecule.SetProp("compound", compound)
    molecule.SetProp("canonical_smiles", canonical)
    molecule.SetProp("conformer_protocol", f"ETKDGv3_seed_{CONFORMER_SEED}_{optimization}")
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Failed to write deterministic conformer for {compound}")
    return {
        "canonical_smiles": canonical,
        "heavy_atoms": int(Chem.RemoveHs(molecule).GetNumAtoms()),
        "optimization": optimization,
        "optimization_status": optimization_status,
        "input_sha256": sha256_file(destination),
    }


def score_from_molecule(molecule: Chem.Mol) -> float:
    if not molecule.HasProp("meeko"):
        raise RuntimeError("Meeko score metadata is missing from exported pose")
    return float(json.loads(molecule.GetProp("meeko"))["free_energy"])


def heavy_pose(raw_molecule: Chem.Mol) -> tuple[Chem.Mol, np.ndarray]:
    molecule = Chem.RemoveHs(raw_molecule)
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    return molecule, coordinates


def generic_heteroatom_distances(molecule: Chem.Mol, zinc: np.ndarray) -> tuple[float, float]:
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    distances = sorted(
        float(np.linalg.norm(coordinates[atom.GetIdx()] - zinc))
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in {7, 8, 16}
    )
    if not distances:
        return math.nan, math.nan
    if len(distances) == 1:
        return distances[0], math.nan
    return distances[0], distances[1]


def hydroxamate_distances(molecule: Chem.Mol, zinc: np.ndarray) -> tuple[float, float]:
    pattern = Chem.MolFromSmarts("[O:1]-[N:2]-[C:3](=[O:4])")
    match = molecule.GetSubstructMatch(pattern)
    if len(match) != 4:
        return math.nan, math.nan
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    values = sorted(
        (
            float(np.linalg.norm(coordinates[match[0]] - zinc)),
            float(np.linalg.norm(coordinates[match[3]] - zinc)),
        )
    )
    return values[0], values[1]


def ortho_amino_benzamide_distances(
    molecule: Chem.Mol, zinc: np.ndarray
) -> tuple[float, float]:
    """Find the carbonyl O and ortho exocyclic amino N without name heuristics."""
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    matches: list[tuple[int, int]] = []
    for carbon in molecule.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        carbonyl_oxygens = [
            bond.GetOtherAtom(carbon).GetIdx()
            for bond in carbon.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(carbon).GetAtomicNum() == 8
        ]
        amide_nitrogens = [
            bond.GetOtherAtom(carbon)
            for bond in carbon.GetBonds()
            if bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetOtherAtom(carbon).GetAtomicNum() == 7
        ]
        for amide_nitrogen in amide_nitrogens:
            aromatic_anchors = [
                neighbor
                for neighbor in amide_nitrogen.GetNeighbors()
                if neighbor.GetIdx() != carbon.GetIdx() and neighbor.GetIsAromatic()
            ]
            for anchor in aromatic_anchors:
                for ring_neighbor in anchor.GetNeighbors():
                    if not ring_neighbor.GetIsAromatic():
                        continue
                    exocyclic_nitrogens = [
                        neighbor
                        for neighbor in ring_neighbor.GetNeighbors()
                        if not neighbor.GetIsAromatic()
                        and neighbor.GetAtomicNum() == 7
                        and neighbor.GetIdx() != amide_nitrogen.GetIdx()
                    ]
                    for amino in exocyclic_nitrogens:
                        for oxygen_index in carbonyl_oxygens:
                            matches.append((oxygen_index, amino.GetIdx()))
    if not matches:
        return math.nan, math.nan
    values = min(
        (
            sorted(
                (
                    float(np.linalg.norm(coordinates[oxygen] - zinc)),
                    float(np.linalg.norm(coordinates[nitrogen] - zinc)),
                )
            )
            for oxygen, nitrogen in matches
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    return values[0], values[1]


def zbg_distances(
    molecule: Chem.Mol, zinc: np.ndarray, zbg_class: str
) -> tuple[float, float, str]:
    if zbg_class in {"hydroxamate", "O_methyl_hydroxamate_decoy"}:
        first, second = hydroxamate_distances(molecule, zinc)
        return first, second, "N-O and carbonyl oxygens"
    if zbg_class == "ortho_amino_benzamide":
        first, second = ortho_amino_benzamide_distances(molecule, zinc)
        return first, second, "ortho amino nitrogen and amide carbonyl oxygen"
    return math.nan, math.nan, "not pre-specified for the intact parent structure"


def pair_pose_rmsd(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_molecule = first["molecule"]
    second_molecule = second["molecule"]
    first_coordinates = first["coordinates"]
    second_coordinates = second["coordinates"]
    matches = second_molecule.GetSubstructMatches(
        first_molecule, uniquify=False, maxMatches=10000
    )
    if not matches:
        raise RuntimeError("Docked poses are not graph-isomorphic within a compound")
    return min(
        float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (second_coordinates[list(match)] - first_coordinates) ** 2,
                        axis=1,
                    )
                )
            )
        )
        for match in matches
    )


def load_poses(
    compound: dict[str, str],
    seeds: list[int],
    compound_dir: Path,
    zinc: np.ndarray,
) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    compound_slug = slug(compound["compound"])
    for seed in seeds:
        path = compound_dir / f"{compound_slug}_ad4zn_seed{seed}.sdf"
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        if not supplier:
            raise RuntimeError(f"No exported poses in {path}")
        for rank, raw_molecule in enumerate(supplier, start=1):
            if raw_molecule is None:
                raise RuntimeError(f"Unreadable pose in {path} at rank {rank}")
            molecule, coordinates = heavy_pose(raw_molecule)
            hetero_min, hetero_second = generic_heteroatom_distances(molecule, zinc)
            zbg_min, zbg_max, zbg_definition = zbg_distances(
                molecule, zinc, compound["zbg_class"]
            )
            poses.append(
                {
                    "compound": compound["compound"],
                    "role": compound["role"],
                    "zbg_class": compound["zbg_class"],
                    "zbg_definition": zbg_definition,
                    "seed": seed,
                    "rank": rank,
                    "score_kcal_mol": score_from_molecule(raw_molecule),
                    "nearest_heteroatom_zn_a": hetero_min,
                    "second_heteroatom_zn_a": hetero_second,
                    "zbg_min_zn_a": zbg_min,
                    "zbg_max_zn_a": zbg_max,
                    "source_sdf": str(path),
                    "raw_molecule": raw_molecule,
                    "molecule": molecule,
                    "coordinates": coordinates,
                }
            )
    return poses


def cluster_consensus(
    poses: list[dict[str, Any]], seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    selected_poses = [pose for pose in poses if pose["rank"] <= CONSENSUS_TOP_K]
    parent = list(range(len(selected_poses)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    pairwise: dict[tuple[int, int], float] = {}
    for first in range(len(selected_poses)):
        for second in range(first):
            value = pair_pose_rmsd(selected_poses[first], selected_poses[second])
            pairwise[(second, first)] = value
            if value <= CLUSTER_RMSD_A:
                union(first, second)

    members: dict[int, list[int]] = {}
    for index in range(len(selected_poses)):
        members.setdefault(find(index), []).append(index)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_number, indices in enumerate(
        sorted(members.values(), key=lambda values: min(values)), start=1
    ):
        cluster_poses = [selected_poses[index] for index in indices]
        cluster_rows.append(
            {
                "compound": cluster_poses[0]["compound"],
                "role": cluster_poses[0]["role"],
                "cluster_id": cluster_number,
                "indices": indices,
                "seed_coverage": len({pose["seed"] for pose in cluster_poses}),
                "pose_count": len(indices),
                "median_score_kcal_mol": float(
                    median(pose["score_kcal_mol"] for pose in cluster_poses)
                ),
                "median_rank": float(median(pose["rank"] for pose in cluster_poses)),
                "seeds": "|".join(
                    str(seed) for seed in sorted({pose["seed"] for pose in cluster_poses})
                ),
                "poses": "|".join(
                    f"s{pose['seed']}r{pose['rank']}" for pose in cluster_poses
                ),
            }
        )
    eligible = [row for row in cluster_rows if row["seed_coverage"] == len(seeds)]
    if not eligible:
        for row in cluster_rows:
            row["selected"] = False
        return cluster_rows, None, None
    chosen = min(
        eligible,
        key=lambda row: (
            row["median_score_kcal_mol"],
            row["median_rank"],
            row["cluster_id"],
        ),
    )
    chosen_indices = chosen["indices"]

    def distance(first: int, second: int) -> float:
        if first == second:
            return 0.0
        return pairwise[(min(first, second), max(first, second))]

    medoid_index = min(
        chosen_indices,
        key=lambda index: (
            median(distance(index, other) for other in chosen_indices),
            selected_poses[index]["score_kcal_mol"],
            selected_poses[index]["rank"],
            selected_poses[index]["seed"],
        ),
    )
    medoid = selected_poses[medoid_index]
    chosen["selected"] = True
    chosen["medoid_seed"] = medoid["seed"]
    chosen["medoid_rank"] = medoid["rank"]
    for row in cluster_rows:
        row.setdefault("selected", False)
        row.setdefault("medoid_seed", None)
        row.setdefault("medoid_rank", None)
    return cluster_rows, chosen, medoid


def read_zinc(receptor_pdb: Path) -> np.ndarray:
    zinc: list[list[float]] = []
    for line in receptor_pdb.read_text(encoding="utf-8").splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == "ZN":
            zinc.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    if len(zinc) != 1:
        raise RuntimeError(f"Expected one catalytic zinc in frozen receptor; observed={len(zinc)}")
    return np.asarray(zinc[0], dtype=float)


def write_medoid(medoid: dict[str, Any], destination: Path) -> None:
    molecule = Chem.Mol(medoid["raw_molecule"])
    molecule.SetProp("consensus_medoid", "true")
    molecule.SetProp("seed", str(medoid["seed"]))
    molecule.SetProp("rank", str(medoid["rank"]))
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()


def positive_control_row(
    panel_entry: dict[str, str],
    redocking_dir: Path,
    zinc: np.ndarray,
    medoid_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = pd.read_csv(redocking_dir / "redocking_method_summary.csv")
    row = method.loc[method["method"] == "ad4zn"]
    if len(row) != 1:
        raise RuntimeError("Passed redocking output has no unique AutoDock4Zn summary row")
    summary = row.iloc[0]
    seed = int(summary["consensus_medoid_seed"])
    rank = int(summary["consensus_medoid_rank"])
    sdf_path = redocking_dir / f"SHH_ad4zn_seed{seed}.sdf"
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    if not supplier or len(supplier) < rank or supplier[rank - 1] is None:
        raise RuntimeError("Could not load the passed redocking consensus medoid")
    raw_molecule = supplier[rank - 1]
    molecule, _ = heavy_pose(raw_molecule)
    hetero_min, hetero_second = generic_heteroatom_distances(molecule, zinc)
    zbg_min, zbg_max, zbg_definition = zbg_distances(
        molecule, zinc, panel_entry["zbg_class"]
    )
    medoid = {
        "raw_molecule": raw_molecule,
        "seed": seed,
        "rank": rank,
    }
    write_medoid(medoid, medoid_dir / "vorinostat_consensus_medoid.sdf")
    pose_metrics = pd.read_csv(redocking_dir / "redocking_pose_metrics.csv")
    pose_row = pose_metrics.loc[
        (pose_metrics["method"] == "ad4zn")
        & (pose_metrics["seed"] == seed)
        & (pose_metrics["rank"] == rank)
    ]
    if len(pose_row) != 1:
        raise RuntimeError("Passed redocking pose table does not identify its medoid uniquely")
    result = {
        **panel_entry,
        "stable_all_seed_consensus": True,
        "seeds": int(summary["seeds"]),
        "selected_cluster_seed_coverage": int(summary["selected_cluster_seed_coverage"]),
        "selected_cluster_pose_count": int(summary["selected_cluster_pose_count"]),
        "selected_cluster_median_score_kcal_mol": float(
            summary["selected_cluster_median_score_kcal_mol"]
        ),
        "consensus_medoid_seed": seed,
        "consensus_medoid_rank": rank,
        "consensus_medoid_score_kcal_mol": float(pose_row.iloc[0]["score_kcal_mol"]),
        "nearest_heteroatom_zn_a": hetero_min,
        "second_heteroatom_zn_a": hetero_second,
        "zbg_min_zn_a": zbg_min,
        "zbg_max_zn_a": zbg_max,
        "zbg_definition": zbg_definition,
        "redocking_crystal_rmsd_a": float(summary["consensus_medoid_rmsd_a"]),
        "input_sha256": None,
        "medoid_sdf": str(medoid_dir / "vorinostat_consensus_medoid.sdf"),
    }
    cluster = {
        "compound": panel_entry["compound"],
        "role": panel_entry["role"],
        "cluster_id": "imported_passed_redocking_cluster",
        "seed_coverage": int(summary["selected_cluster_seed_coverage"]),
        "pose_count": int(summary["selected_cluster_pose_count"]),
        "median_score_kcal_mol": float(summary["selected_cluster_median_score_kcal_mol"]),
        "selected": True,
        "medoid_seed": seed,
        "medoid_rank": rank,
    }
    return result, cluster


def plot_panel(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    display = summary.copy()
    display["label"] = display["compound"].replace(
        {"Vorinostat_O_methyl_decoy": "Vorinostat O-methyl\ndesigned decoy"}
    )
    colors = {
        "crystallographic_positive_control": "#176B87",
        "designed_zinc_chelation_sensitivity_control": "#888888",
        "core_candidate": "#C0392B",
        "secondary_candidate": "#D68910",
        "original_method_sensitive_candidate": "#8E44AD",
        "reference_control": "#2874A6",
        "class_support_control": "#1E8449",
    }
    x = np.arange(len(display))
    figure, axes = plt.subplots(2, 1, figsize=(10.4, 7.5), sharex=True)
    stable = display["stable_all_seed_consensus"].astype(bool)
    for index, row in display.iterrows():
        color = colors.get(str(row["role"]), "#555555")
        marker = "o" if bool(row["stable_all_seed_consensus"]) else "x"
        score = row["selected_cluster_median_score_kcal_mol"]
        distance = row["zbg_min_zn_a"]
        if pd.isna(distance):
            distance = row["nearest_heteroatom_zn_a"]
        if not pd.isna(score):
            axes[0].scatter(index, score, color=color, marker=marker, s=75, zorder=3)
        if not pd.isna(distance):
            axes[1].scatter(index, distance, color=color, marker=marker, s=75, zorder=3)
    axes[0].axhline(0, color="#CCCCCC", linewidth=0.8)
    axes[0].set_ylabel("Consensus-cluster score\n(kcal/mol; descriptive)")
    axes[1].axhspan(1.8, 2.8, color="#176B87", alpha=0.10, label="Zn-distance reference band")
    axes[1].set_ylabel("Nearest pre-specified ZBG atom\n(or nearest N/O/S) to Zn (Å)")
    axes[1].set_xticks(x, display["label"], rotation=30, ha="right")
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    if (~stable).any():
        axes[0].text(
            0.99,
            0.02,
            "× = no all-seed consensus",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
    figure.suptitle("4LXZ AutoDock4Zn controlled structural-plausibility panel")
    figure.text(
        0.5,
        0.005,
        "Docking scores are not binding affinities and are not efficacy or target validation.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(output_dir / "controlled_docking_panel_4lxz.png", dpi=dpi)
    figure.savefig(output_dir / "controlled_docking_panel_4lxz.pdf")
    plt.close(figure)


def clean_pose_row(pose: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in pose.items()
        if key not in {"raw_molecule", "molecule", "coordinates"}
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.env_dir = args.env_dir.expanduser().resolve()
    args.redocking_dir = args.redocking_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    ligand_dir = args.output_dir / "ligands"
    medoid_dir = args.output_dir / "consensus_medoids"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    medoid_dir.mkdir(parents=True, exist_ok=True)

    gate_path = args.redocking_dir / "redocking_gate_manifest.json"
    gate = load_gate(gate_path)
    seeds = [int(value) for value in gate["seeds"]]
    exhaustiveness = int(gate["exhaustiveness"])
    num_modes = int(gate["num_modes"])
    cpu = int(args.cpu if args.cpu is not None else gate["cpu_per_run"])
    if cpu < 1:
        raise ValueError("--cpu must be positive")
    box = gate["box"]
    center = tuple(float(value) for value in box["center"])
    npts = tuple(int(value) for value in box["autogrid_npts"])
    if len(center) != 3 or len(npts) != 3:
        raise RuntimeError("Frozen redocking box is malformed")

    vina = args.env_dir / "bin/vina"
    mk_ligand = args.env_dir / "bin/mk_prepare_ligand.py"
    mk_export = args.env_dir / "bin/mk_export.py"
    autogrid = args.env_dir / "bin/autogrid4"
    receptor_pdb = args.redocking_dir / "4LXZ_chainA_receptor.pdb"
    receptor_tz = args.redocking_dir / "4LXZ_chainA_receptor_tz.pdbqt"
    parameter_source = args.redocking_dir / "AD4Zn.dat"
    require_files(
        [
            vina,
            mk_ligand,
            mk_export,
            autogrid,
            receptor_pdb,
            receptor_tz,
            parameter_source,
            gate_path,
            args.redocking_dir / "redocking_method_summary.csv",
            args.redocking_dir / "redocking_pose_metrics.csv",
        ]
    )
    zinc = read_zinc(receptor_pdb)

    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []

    positive = PANEL[0]
    positive_row, positive_cluster = positive_control_row(
        dict(positive), args.redocking_dir, zinc, medoid_dir
    )
    summary_rows.append(positive_row)
    cluster_rows.append(positive_cluster)
    input_rows.append(
        {
            **positive,
            "canonical_smiles": Chem.MolToSmiles(
                Chem.MolFromSmiles(positive["smiles"]), isomericSmiles=True
            ),
            "conformer_protocol": "PDB CCD ideal SHH coordinates used by passed redocking",
        }
    )

    for entry in PANEL[1:]:
        compound = dict(entry)
        compound_slug = slug(compound["compound"])
        compound_dir = ligand_dir / compound_slug
        compound_dir.mkdir(parents=True, exist_ok=True)
        input_sdf = compound_dir / f"{compound_slug}_input.sdf"
        conformer = deterministic_conformer(
            compound["smiles"], input_sdf, compound["compound"]
        )
        input_rows.append(
            {
                **compound,
                **conformer,
                "conformer_seed": CONFORMER_SEED,
            }
        )
        ligand_pdbqt = compound_dir / f"{compound_slug}.pdbqt"
        run(
            [
                str(mk_ligand),
                "-i",
                str(input_sdf),
                "-o",
                str(ligand_pdbqt),
                "--add_index_map",
            ],
            cwd=compound_dir,
            log_path=log_dir / f"prepare_{compound_slug}.log",
        )
        local_receptor = compound_dir / receptor_tz.name
        local_parameter = compound_dir / parameter_source.name
        shutil.copy2(receptor_tz, local_receptor)
        shutil.copy2(parameter_source, local_parameter)
        gpf = compound_dir / f"{compound_slug}.gpf"
        maps_prefix = write_ad4zn_gpf(
            gpf,
            local_receptor,
            ligand_pdbqt,
            local_parameter.name,
            center,
            npts,
            SPACING,
        )
        autogrid_output = run(
            [str(autogrid), "-p", gpf.name, "-l", f"{compound_slug}.glg"],
            cwd=compound_dir,
            log_path=log_dir / f"autogrid_{compound_slug}.log",
            env={**os.environ, "OMP_NUM_THREADS": str(cpu)},
        )
        if "Successful Completion" not in autogrid_output:
            glg = compound_dir / f"{compound_slug}.glg"
            if not glg.is_file() or "Successful Completion" not in glg.read_text(
                encoding="utf-8", errors="replace"
            ):
                raise RuntimeError(f"AutoGrid failed for {compound['compound']}")

        for seed in seeds:
            output_pdbqt = compound_dir / f"{compound_slug}_ad4zn_seed{seed}.pdbqt"
            output_sdf = compound_dir / f"{compound_slug}_ad4zn_seed{seed}.sdf"
            run(
                [
                    str(vina),
                    "--ligand",
                    str(ligand_pdbqt),
                    "--maps",
                    maps_prefix,
                    "--scoring",
                    "ad4",
                    "--exhaustiveness",
                    str(exhaustiveness),
                    "--num_modes",
                    str(num_modes),
                    "--seed",
                    str(seed),
                    "--cpu",
                    str(cpu),
                    "--out",
                    str(output_pdbqt),
                ],
                cwd=compound_dir,
                log_path=log_dir / f"dock_{compound_slug}_seed{seed}.log",
            )
            run(
                [str(mk_export), str(output_pdbqt), "-s", str(output_sdf)],
                cwd=compound_dir,
                log_path=log_dir / f"export_{compound_slug}_seed{seed}.log",
            )

        poses = load_poses(compound, seeds, compound_dir, zinc)
        pose_rows.extend(clean_pose_row(pose) for pose in poses)
        clusters, chosen, medoid = cluster_consensus(poses, seeds)
        for cluster in clusters:
            cluster.pop("indices", None)
            cluster_rows.append(cluster)
        top_one = [pose for pose in poses if pose["rank"] == 1]
        if chosen is None or medoid is None:
            summary_rows.append(
                {
                    **compound,
                    "stable_all_seed_consensus": False,
                    "seeds": len(seeds),
                    "selected_cluster_seed_coverage": max(
                        (row["seed_coverage"] for row in clusters), default=0
                    ),
                    "selected_cluster_pose_count": None,
                    "selected_cluster_median_score_kcal_mol": None,
                    "consensus_medoid_seed": None,
                    "consensus_medoid_rank": None,
                    "consensus_medoid_score_kcal_mol": None,
                    "nearest_heteroatom_zn_a": None,
                    "second_heteroatom_zn_a": None,
                    "zbg_min_zn_a": None,
                    "zbg_max_zn_a": None,
                    "zbg_definition": poses[0]["zbg_definition"],
                    "redocking_crystal_rmsd_a": None,
                    "top1_score_median_kcal_mol": float(
                        median(pose["score_kcal_mol"] for pose in top_one)
                    ),
                    "input_sha256": conformer["input_sha256"],
                    "medoid_sdf": None,
                }
            )
            continue
        medoid_path = medoid_dir / f"{compound_slug}_consensus_medoid.sdf"
        write_medoid(medoid, medoid_path)
        summary_rows.append(
            {
                **compound,
                "stable_all_seed_consensus": True,
                "seeds": len(seeds),
                "selected_cluster_seed_coverage": chosen["seed_coverage"],
                "selected_cluster_pose_count": chosen["pose_count"],
                "selected_cluster_median_score_kcal_mol": chosen[
                    "median_score_kcal_mol"
                ],
                "consensus_medoid_seed": medoid["seed"],
                "consensus_medoid_rank": medoid["rank"],
                "consensus_medoid_score_kcal_mol": medoid["score_kcal_mol"],
                "nearest_heteroatom_zn_a": medoid["nearest_heteroatom_zn_a"],
                "second_heteroatom_zn_a": medoid["second_heteroatom_zn_a"],
                "zbg_min_zn_a": medoid["zbg_min_zn_a"],
                "zbg_max_zn_a": medoid["zbg_max_zn_a"],
                "zbg_definition": medoid["zbg_definition"],
                "redocking_crystal_rmsd_a": None,
                "top1_score_median_kcal_mol": float(
                    median(pose["score_kcal_mol"] for pose in top_one)
                ),
                "input_sha256": conformer["input_sha256"],
                "medoid_sdf": str(medoid_path),
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    pose_frame = pd.DataFrame(pose_rows)
    input_frame = pd.DataFrame(input_rows)
    summary_frame.to_csv(args.output_dir / "controlled_docking_summary.csv", index=False)
    cluster_frame.to_csv(args.output_dir / "controlled_docking_clusters.csv", index=False)
    pose_frame.to_csv(args.output_dir / "controlled_docking_pose_metrics.csv", index=False)
    input_frame.to_csv(args.output_dir / "controlled_docking_input_manifest.csv", index=False)
    plot_panel(summary_frame, args.output_dir, args.figure_dpi)

    decoy = summary_frame.loc[
        summary_frame["role"] == "designed_zinc_chelation_sensitivity_control"
    ].iloc[0]
    positive_result = summary_frame.loc[
        summary_frame["role"] == "crystallographic_positive_control"
    ].iloc[0]
    descriptive_control = {
        "positive_consensus": bool(positive_result["stable_all_seed_consensus"]),
        "designed_decoy_consensus": bool(decoy["stable_all_seed_consensus"]),
        "positive_score_kcal_mol": float(
            positive_result["selected_cluster_median_score_kcal_mol"]
        ),
        "designed_decoy_score_kcal_mol": (
            None
            if pd.isna(decoy["selected_cluster_median_score_kcal_mol"])
            else float(decoy["selected_cluster_median_score_kcal_mol"])
        ),
        "positive_zbg_min_zn_a": float(positive_result["zbg_min_zn_a"]),
        "designed_decoy_zbg_min_zn_a": (
            None if pd.isna(decoy["zbg_min_zn_a"]) else float(decoy["zbg_min_zn_a"])
        ),
        "policy": (
            "Descriptive sensitivity control only; no post-hoc pass threshold and no "
            "compound is removed because of this comparison."
        ),
    }
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel_state": "frozen_before_candidate_scores",
        "primary_structure": "4LXZ chain A HDAC2",
        "primary_protocol": "passed AutoDock4Zn all-seed consensus",
        "redocking_gate_manifest_sha256": sha256_file(gate_path),
        "panel": [dict(value) for value in PANEL],
        "exclusions": {
            "RG2833": "prediction-only; no measured LINCS support",
            "Tianeptinaline_or_BG-1010": "identity-conflict excluded",
            "PCI-24781_and_Panobinostat": (
                "measured-validation/class-support rows, not frozen biological candidates "
                "or reference controls for this focused panel"
            ),
        },
        "designed_negative_control": (
            "Vorinostat hydroxamate O-methyl analogue; predeclared to remove the "
            "hydroxamate O-H state as a zinc-chelation sensitivity control, not claimed "
            "to have experimental inactivity."
        ),
        "seeds": seeds,
        "exhaustiveness": exhaustiveness,
        "num_modes": num_modes,
        "cpu_per_run": cpu,
        "consensus_top_k": CONSENSUS_TOP_K,
        "cluster_rmsd_a": CLUSTER_RMSD_A,
        "selection_rule": gate["selection_rule"],
        "conformer_protocol": {
            "method": "RDKit ETKDGv3 followed by MMFF94 when parameterized, otherwise UFF",
            "seed": CONFORMER_SEED,
            "one_starting_conformer_per_noncrystal_ligand": True,
        },
        "descriptive_control_result": descriptive_control,
        "stable_all_seed_consensus_compounds": summary_frame.loc[
            summary_frame["stable_all_seed_consensus"].astype(bool), "compound"
        ].tolist(),
        "unstable_compounds": summary_frame.loc[
            ~summary_frame["stable_all_seed_consensus"].astype(bool), "compound"
        ].tolist(),
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "rdkit_version": rdBase.rdkitVersion,
        "platform": platform.platform(),
        "gpu_required": False,
        "interpretation_guardrail": (
            "Docking is a controlled structural-plausibility analysis. Scores are not "
            "binding affinities and cannot establish efficacy, selectivity, direct target "
            "engagement, or superiority between different zinc-binding chemotypes."
        ),
        "outputs": [
            "controlled_docking_summary.csv",
            "controlled_docking_clusters.csv",
            "controlled_docking_pose_metrics.csv",
            "controlled_docking_input_manifest.csv",
            "consensus_medoids/*.sdf",
            "controlled_docking_panel_4lxz.png",
            "controlled_docking_panel_4lxz.pdf",
            "controlled_docking_manifest.json",
            "audit.log",
        ],
    }
    manifest_path = args.output_dir / "controlled_docking_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("===== FROZEN 4LXZ CONTROL/CANDIDATE PANEL =====")
    display_columns = [
        "compound",
        "role",
        "stable_all_seed_consensus",
        "selected_cluster_pose_count",
        "selected_cluster_median_score_kcal_mol",
        "consensus_medoid_score_kcal_mol",
        "zbg_min_zn_a",
        "zbg_max_zn_a",
        "nearest_heteroatom_zn_a",
    ]
    print(summary_frame[display_columns].to_string(index=False))
    print("\n===== DESCRIPTIVE CONTROL COMPARISON =====")
    print(json.dumps(descriptive_control, indent=2))
    final_line = "4LXZ CONTROLLED DOCKING PANEL: PASSED"
    print("\n" + final_line)
    audit = (
        summary_frame[display_columns].to_string(index=False)
        + "\n\n"
        + json.dumps(descriptive_control, indent=2)
        + "\n\n"
        + final_line
        + "\n"
        + f"manifest_sha256={sha256_file(manifest_path)}\n"
    )
    (args.output_dir / "audit.log").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
