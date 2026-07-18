#!/usr/bin/env python3
"""Run the frozen 4LXZ-SHH redocking gate before any candidate docking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from e3_ad4zn_support import sha256_file, write_ad4zn_gpf


DEFAULT_ENV_DIR = Path(".venv/e3_docking")
DEFAULT_SOURCE_DIR = Path("results/revision/docking_controls/source_structures")
DEFAULT_OUTPUT_DIR = Path("results/revision/docking_controls/redocking_4lxz")
METHODS = ("vina_standard", "ad4zn")
SPACING = 0.375


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--consensus-top-k", type=int, default=5)
    parser.add_argument("--cluster-rmsd", type=float, default=2.0)
    parser.add_argument("--pass-rmsd", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


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


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Redocking requires at least three unique integer seeds")
    return seeds


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required E3 paths: {missing}")


def extract_chain_a_receptor(source: Path, destination: Path) -> np.ndarray:
    lines: list[str] = []
    zinc: list[list[float]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        record = line[:6].strip()
        chain = line[21:22]
        residue = line[17:20].strip()
        alternate = line[16:17]
        if record == "ATOM" and chain == "A" and alternate in {" ", "A"}:
            lines.append(line[:16] + " " + line[17:])
        elif record == "HETATM" and chain == "A" and residue == "ZN":
            lines.append(line)
            zinc.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if len(zinc) != 1:
        raise RuntimeError(f"Expected exactly one chain-A catalytic zinc; observed={len(zinc)}")
    lines.extend(["TER", "END"])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return np.asarray(zinc[0], dtype=float)


def read_first_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecule = supplier[0] if supplier else None
    if molecule is None:
        raise RuntimeError(f"Could not read SDF molecule: {path}")
    return Chem.RemoveHs(molecule)


def box_from_ligand(path: Path, padding: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    molecule = read_first_sdf(path)
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = (minimum + maximum) / 2.0
    size = maximum - minimum + 2.0 * padding
    npts: list[int] = []
    for dimension in size:
        points = int(math.ceil(float(dimension) / SPACING))
        if points % 2:
            points += 1
        npts.append(points)
    return center, size, tuple(npts)


def direct_symmetry_rmsd(reference: Chem.Mol, moving: Chem.Mol) -> float:
    reference_coordinates = np.asarray(reference.GetConformer().GetPositions(), dtype=float)
    moving_coordinates = np.asarray(moving.GetConformer().GetPositions(), dtype=float)
    matches = moving.GetSubstructMatches(reference, uniquify=False, maxMatches=10000)
    if not matches:
        raise RuntimeError("Docked pose is not graph-isomorphic to the reference ligand")
    return min(
        float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (moving_coordinates[list(match)] - reference_coordinates) ** 2,
                        axis=1,
                    )
                )
            )
        )
        for match in matches
    )


def pair_pose_rmsd(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_molecule = first["molecule"]
    second_molecule = second["molecule"]
    first_coordinates = first["coordinates"]
    second_coordinates = second["coordinates"]
    matches = second_molecule.GetSubstructMatches(
        first_molecule, uniquify=False, maxMatches=10000
    )
    if not matches:
        raise RuntimeError("Redocking poses are not graph-isomorphic")
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


def hydroxamate_oxygen_distances(molecule: Chem.Mol, zinc: np.ndarray) -> tuple[float, float]:
    pattern = Chem.MolFromSmarts("[O:1]=[C:2]([N:3][O:4])")
    match = molecule.GetSubstructMatch(pattern)
    if len(match) != 4:
        raise RuntimeError("Could not identify the SHH hydroxamate group")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    distances = sorted(
        [
            float(np.linalg.norm(coordinates[match[0]] - zinc)),
            float(np.linalg.norm(coordinates[match[3]] - zinc)),
        ]
    )
    return distances[0], distances[1]


def score_from_molecule(molecule: Chem.Mol) -> float:
    if not molecule.HasProp("meeko"):
        raise RuntimeError("Meeko score metadata is missing from exported pose")
    return float(json.loads(molecule.GetProp("meeko"))["free_energy"])


def load_method_poses(
    method: str,
    seeds: list[int],
    output_dir: Path,
    crystal: Chem.Mol,
    zinc: np.ndarray,
) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    for seed in seeds:
        path = output_dir / f"SHH_{method}_seed{seed}.sdf"
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for rank, raw_molecule in enumerate(supplier, start=1):
            if raw_molecule is None:
                raise RuntimeError(f"Unreadable pose in {path} at rank {rank}")
            molecule = Chem.RemoveHs(raw_molecule)
            closest, farther = hydroxamate_oxygen_distances(molecule, zinc)
            poses.append(
                {
                    "method": method,
                    "seed": seed,
                    "rank": rank,
                    "score_kcal_mol": score_from_molecule(raw_molecule),
                    "crystal_rmsd_a": direct_symmetry_rmsd(crystal, molecule),
                    "hydroxamate_zn_min_a": closest,
                    "hydroxamate_zn_max_a": farther,
                    "molecule": molecule,
                    "coordinates": np.asarray(
                        molecule.GetConformer().GetPositions(), dtype=float
                    ),
                    "source_sdf": str(path),
                }
            )
    return poses


def cluster_consensus(
    poses: list[dict[str, Any]],
    seeds: list[int],
    top_k: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected_poses = [pose for pose in poses if pose["rank"] <= top_k]
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
            if value <= threshold:
                union(first, second)

    members: dict[int, list[int]] = {}
    for index in range(len(selected_poses)):
        members.setdefault(find(index), []).append(index)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_number, indices in enumerate(
        sorted(members.values(), key=lambda values: min(values)), start=1
    ):
        cluster_poses = [selected_poses[index] for index in indices]
        coverage = len({pose["seed"] for pose in cluster_poses})
        cluster_rows.append(
            {
                "cluster_id": cluster_number,
                "indices": indices,
                "seed_coverage": coverage,
                "pose_count": len(indices),
                "median_score_kcal_mol": float(
                    median(pose["score_kcal_mol"] for pose in cluster_poses)
                ),
                "median_rank": float(median(pose["rank"] for pose in cluster_poses)),
                "median_crystal_rmsd_a": float(
                    median(pose["crystal_rmsd_a"] for pose in cluster_poses)
                ),
                "max_crystal_rmsd_a": float(
                    max(pose["crystal_rmsd_a"] for pose in cluster_poses)
                ),
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
        raise RuntimeError("No top-k pose cluster recurred across every random seed")
    chosen_cluster = min(
        eligible,
        key=lambda row: (
            row["median_score_kcal_mol"],
            row["median_rank"],
            row["cluster_id"],
        ),
    )
    chosen_indices = chosen_cluster["indices"]

    def distance(first: int, second: int) -> float:
        if first == second:
            return 0.0
        key = (min(first, second), max(first, second))
        return pairwise[key]

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
    chosen_cluster["selected"] = True
    chosen_cluster["medoid_seed"] = medoid["seed"]
    chosen_cluster["medoid_rank"] = medoid["rank"]
    chosen_cluster["medoid_crystal_rmsd_a"] = medoid["crystal_rmsd_a"]
    for row in cluster_rows:
        row.setdefault("selected", False)
        row.setdefault("medoid_seed", None)
        row.setdefault("medoid_rank", None)
        row.setdefault("medoid_crystal_rmsd_a", None)
    return cluster_rows, chosen_cluster, medoid


def clean_pose_row(pose: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in pose.items() if key not in {"molecule", "coordinates"}}


def plot_control(summary: pd.DataFrame, pose_frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    labels = {"vina_standard": "Standard Vina", "ad4zn": "AutoDock4Zn"}
    colors = {"vina_standard": "#777777", "ad4zn": "#176B87"}
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for x_position, method in enumerate(METHODS):
        top = pose_frame[(pose_frame["method"] == method) & (pose_frame["rank"] == 1)]
        jitter = np.linspace(-0.10, 0.10, len(top))
        axis.scatter(
            x_position + jitter,
            top["crystal_rmsd_a"],
            color=colors[method],
            s=48,
            alpha=0.75,
            label="Rank-1 poses" if x_position == 0 else None,
        )
        value = float(
            summary.loc[summary["method"] == method, "consensus_medoid_rmsd_a"].iloc[0]
        )
        axis.scatter(
            [x_position],
            [value],
            marker="*",
            s=180,
            color="#C0392B",
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
            label="Seed-consensus medoid" if x_position == 0 else None,
        )
    axis.axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="2 Å gate")
    axis.set_xticks(range(len(METHODS)), [labels[method] for method in METHODS])
    axis.set_ylabel("Heavy-atom RMSD to 4LXZ crystal pose (Å)")
    axis.set_title("4LXZ–SAHA redocking protocol control")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "redocking_protocol_control.png", dpi=dpi)
    figure.savefig(output_dir / "redocking_protocol_control.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.env_dir = args.env_dir.expanduser().resolve()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"

    env_python = args.env_dir / "bin/python"
    vina = args.env_dir / "bin/vina"
    mk_receptor = args.env_dir / "bin/mk_prepare_receptor.py"
    mk_ligand = args.env_dir / "bin/mk_prepare_ligand.py"
    mk_export = args.env_dir / "bin/mk_export.py"
    autogrid = args.env_dir / "bin/autogrid4"
    zinc_pseudo = args.env_dir / "share/ad4zn/zinc_pseudo.py"
    parameter_source = args.env_dir / "share/ad4zn/AD4Zn.dat"
    structure = args.source_dir / "4LXZ.pdb"
    ideal_ligand = args.source_dir / "SHH_ideal.sdf"
    crystal_ligand = args.source_dir / "4LXZ_chainA_SHH_bound.sdf"
    require_paths(
        [
            env_python,
            vina,
            mk_receptor,
            mk_ligand,
            mk_export,
            autogrid,
            zinc_pseudo,
            parameter_source,
            structure,
            ideal_ligand,
            crystal_ligand,
        ]
    )

    receptor_pdb = args.output_dir / "4LXZ_chainA_receptor.pdb"
    receptor_basename = args.output_dir / "4LXZ_chainA_receptor"
    receptor_pdbqt = args.output_dir / "4LXZ_chainA_receptor.pdbqt"
    receptor_tz = args.output_dir / "4LXZ_chainA_receptor_tz.pdbqt"
    ligand_pdbqt = args.output_dir / "SHH_ideal.pdbqt"
    parameter_file = args.output_dir / "AD4Zn.dat"
    zinc = extract_chain_a_receptor(structure, receptor_pdb)
    center, size, npts = box_from_ligand(crystal_ligand, args.padding)

    run(
        [
            str(mk_receptor),
            "--read_pdb",
            str(receptor_pdb),
            "-o",
            str(receptor_basename),
            "-p",
            "-v",
            "--box_center",
            *[f"{value:.6f}" for value in center],
            "--box_size",
            *[f"{value:.6f}" for value in size],
        ],
        cwd=args.output_dir,
        log_path=log_dir / "prepare_receptor.log",
    )
    run(
        [str(mk_ligand), "-i", str(ideal_ligand), "-o", str(ligand_pdbqt), "--add_index_map"],
        cwd=args.output_dir,
        log_path=log_dir / "prepare_ligand.log",
    )
    run(
        [str(env_python), str(zinc_pseudo), "-r", str(receptor_pdbqt), "-o", str(receptor_tz)],
        cwd=args.output_dir,
        log_path=log_dir / "zinc_pseudo.log",
    )
    shutil.copy2(parameter_source, parameter_file)
    gpf = args.output_dir / "4LXZ_chainA_receptor_tz.gpf"
    maps_prefix = write_ad4zn_gpf(
        gpf,
        receptor_tz,
        ligand_pdbqt,
        parameter_file.name,
        tuple(float(value) for value in center),
        npts,
        SPACING,
    )
    autogrid_output = run(
        [str(autogrid), "-p", gpf.name, "-l", "4LXZ_chainA_receptor_tz.glg"],
        cwd=args.output_dir,
        log_path=log_dir / "autogrid4.log",
        env={**os.environ, "OMP_NUM_THREADS": str(args.cpu)},
    )
    if "Successful Completion" not in autogrid_output:
        glg = args.output_dir / "4LXZ_chainA_receptor_tz.glg"
        if not glg.is_file() or "Successful Completion" not in glg.read_text(
            encoding="utf-8", errors="replace"
        ):
            raise RuntimeError("AutoGrid did not report Successful Completion")

    for method in METHODS:
        for seed in seeds:
            output_pdbqt = args.output_dir / f"SHH_{method}_seed{seed}.pdbqt"
            if method == "vina_standard":
                command = [
                    str(vina),
                    "--receptor",
                    str(receptor_pdbqt),
                    "--ligand",
                    str(ligand_pdbqt),
                    "--scoring",
                    "vina",
                    "--center_x",
                    f"{center[0]:.6f}",
                    "--center_y",
                    f"{center[1]:.6f}",
                    "--center_z",
                    f"{center[2]:.6f}",
                    "--size_x",
                    f"{size[0]:.6f}",
                    "--size_y",
                    f"{size[1]:.6f}",
                    "--size_z",
                    f"{size[2]:.6f}",
                ]
            else:
                command = [
                    str(vina),
                    "--ligand",
                    str(ligand_pdbqt),
                    "--maps",
                    maps_prefix,
                    "--scoring",
                    "ad4",
                ]
            command.extend(
                [
                    "--exhaustiveness",
                    str(args.exhaustiveness),
                    "--num_modes",
                    str(args.num_modes),
                    "--seed",
                    str(seed),
                    "--cpu",
                    str(args.cpu),
                    "--out",
                    str(output_pdbqt),
                ]
            )
            run(
                command,
                cwd=args.output_dir,
                log_path=log_dir / f"dock_{method}_seed{seed}.log",
            )
            output_sdf = args.output_dir / f"SHH_{method}_seed{seed}.sdf"
            run(
                [str(mk_export), str(output_pdbqt), "-s", str(output_sdf)],
                cwd=args.output_dir,
                log_path=log_dir / f"export_{method}_seed{seed}.log",
            )

    crystal = read_first_sdf(crystal_ligand)
    crystal_zn_min, crystal_zn_max = hydroxamate_oxygen_distances(crystal, zinc)
    all_poses: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    medoids: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        poses = load_method_poses(method, seeds, args.output_dir, crystal, zinc)
        all_poses.extend(poses)
        clusters, selected_cluster, medoid = cluster_consensus(
            poses, seeds, args.consensus_top_k, args.cluster_rmsd
        )
        medoids[method] = medoid
        for row in clusters:
            row["method"] = method
            row.pop("indices", None)
            cluster_rows.append(row)
        top_one = [pose for pose in poses if pose["rank"] == 1]
        method_rows.append(
            {
                "method": method,
                "seeds": len(seeds),
                "top1_rmsd_median_a": float(
                    median(pose["crystal_rmsd_a"] for pose in top_one)
                ),
                "top1_rmsd_min_a": float(min(pose["crystal_rmsd_a"] for pose in top_one)),
                "top1_rmsd_max_a": float(max(pose["crystal_rmsd_a"] for pose in top_one)),
                "selected_cluster_seed_coverage": selected_cluster["seed_coverage"],
                "selected_cluster_pose_count": selected_cluster["pose_count"],
                "selected_cluster_median_score_kcal_mol": selected_cluster[
                    "median_score_kcal_mol"
                ],
                "consensus_medoid_seed": medoid["seed"],
                "consensus_medoid_rank": medoid["rank"],
                "consensus_medoid_rmsd_a": medoid["crystal_rmsd_a"],
                "consensus_medoid_zn_min_a": medoid["hydroxamate_zn_min_a"],
                "consensus_medoid_zn_max_a": medoid["hydroxamate_zn_max_a"],
                "selected_cluster_max_crystal_rmsd_a": selected_cluster[
                    "max_crystal_rmsd_a"
                ],
            }
        )

    pose_frame = pd.DataFrame([clean_pose_row(pose) for pose in all_poses])
    cluster_frame = pd.DataFrame(cluster_rows)
    method_frame = pd.DataFrame(method_rows)
    pose_frame.to_csv(args.output_dir / "redocking_pose_metrics.csv", index=False)
    cluster_frame.to_csv(args.output_dir / "redocking_cluster_summary.csv", index=False)
    method_frame.to_csv(args.output_dir / "redocking_method_summary.csv", index=False)
    plot_control(method_frame, pose_frame, args.output_dir, args.figure_dpi)

    ad4zn_summary = method_frame.loc[method_frame["method"] == "ad4zn"].iloc[0]
    gate_checks = {
        "all_seeds_in_selected_consensus_cluster": bool(
            ad4zn_summary["selected_cluster_seed_coverage"] == len(seeds)
        ),
        "consensus_medoid_rmsd_le_threshold": bool(
            ad4zn_summary["consensus_medoid_rmsd_a"] <= args.pass_rmsd
        ),
        "all_selected_cluster_poses_rmsd_le_threshold": bool(
            ad4zn_summary["selected_cluster_max_crystal_rmsd_a"] <= args.pass_rmsd
        ),
        "consensus_medoid_nearest_zn_distance_1_8_to_2_8": bool(
            1.8 <= ad4zn_summary["consensus_medoid_zn_min_a"] <= 2.8
        ),
        "consensus_medoid_farther_zn_distance_le_3_6": bool(
            ad4zn_summary["consensus_medoid_zn_max_a"] <= 3.6
        ),
    }
    passed = all(gate_checks.values())
    manifest = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_state": "frozen_before_candidate_docking",
        "primary_protocol": "AutoDock4Zn seed-consensus",
        "sensitivity_protocol": "standard Vina",
        "primary_structure": "4LXZ chain A HDAC2-SHH/SAHA/vorinostat",
        "receptor_policy": (
            "Chain A protein plus catalytic Zn only; crystallographic ligand, water, "
            "polyethylene glycol, calcium, and sodium removed."
        ),
        "ligand_starting_coordinates": "PDB CCD ideal coordinates, not bound coordinates",
        "selection_rule": (
            f"Cluster top {args.consensus_top_k} poses per seed at {args.cluster_rmsd:.1f} Å "
            "direct symmetry-aware RMSD; require all-seed coverage; select the eligible "
            "cluster with the most favorable median docking score; choose its geometric medoid."
        ),
        "rmsd_policy": (
            "Symmetry-aware heavy-atom RMSD in the fixed receptor coordinate frame; no "
            "post-docking ligand superposition."
        ),
        "gate_thresholds": {
            "rmsd_a": args.pass_rmsd,
            "nearest_hydroxamate_o_zn_a": [1.8, 2.8],
            "farther_hydroxamate_o_zn_max_a": 3.6,
        },
        "gate_checks": gate_checks,
        "seeds": seeds,
        "exhaustiveness": args.exhaustiveness,
        "cpu_per_run": args.cpu,
        "num_modes": args.num_modes,
        "box": {
            "center": center.tolist(),
            "size_a": size.tolist(),
            "spacing_a": SPACING,
            "autogrid_npts": list(npts),
            "padding_a": args.padding,
        },
        "crystal_hydroxamate_zn_distances_a": [crystal_zn_min, crystal_zn_max],
        "method_summary": method_rows,
        "source_sha256": {
            "4LXZ.pdb": sha256_file(structure),
            "SHH_ideal.sdf": sha256_file(ideal_ligand),
            "4LXZ_chainA_SHH_bound.sdf": sha256_file(crystal_ligand),
            "AD4Zn.dat": sha256_file(parameter_file),
            "zinc_pseudo.py": sha256_file(zinc_pseudo),
            "vina": sha256_file(vina),
            "autogrid4": sha256_file(autogrid),
        },
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_required": False,
        "interpretation_guardrail": (
            "Passing redocking validates this frozen pose-generation protocol for controlled "
            "structural plausibility analyses. It does not establish efficacy, binding affinity, "
            "selectivity, or a direct target for any candidate."
        ),
        "outputs": [
            "redocking_pose_metrics.csv",
            "redocking_cluster_summary.csv",
            "redocking_method_summary.csv",
            "redocking_protocol_control.png",
            "redocking_protocol_control.pdf",
            "redocking_gate_manifest.json",
            "audit.log",
        ],
    }
    manifest_path = args.output_dir / "redocking_gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("===== 4LXZ REDOCKING METHOD SUMMARY =====")
    print(method_frame.to_string(index=False))
    print("\n===== REDOCKING GATE CHECKS =====")
    print(json.dumps(gate_checks, indent=2))
    print("\n===== CRYSTAL Zn COORDINATION =====")
    print(f"hydroxamate_O_Zn_A={crystal_zn_min:.6f}, {crystal_zn_max:.6f}")
    final_line = f"4LXZ REDOCKING GATE: {'PASSED' if passed else 'FAILED'}"
    print("\n" + final_line)
    audit_text = (
        method_frame.to_string(index=False)
        + "\n\n"
        + json.dumps(gate_checks, indent=2)
        + "\n\n"
        + final_line
        + "\n"
        + f"manifest_sha256={sha256_file(manifest_path)}\n"
    )
    (args.output_dir / "audit.log").write_text(audit_text, encoding="utf-8")
    if not passed:
        raise RuntimeError(
            "The frozen AutoDock4Zn redocking protocol did not pass. Candidate docking is blocked."
        )


if __name__ == "__main__":
    main()
