#!/usr/bin/env python3
"""Run the frozen docking panel on aligned 4BKX/HDAC1 as receptor sensitivity.

4BKX chain B is aligned to 4LXZ chain A before receptor preparation so that
the already-frozen 4LXZ docking box can be reused without selecting a new
pocket after observing candidate results.  Candidate interpretation remains
descriptive; alignment quality is the only pre-docking gate in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from e3_ad4zn_support import sha256_file, write_ad4zn_gpf
from run_e3_controlled_docking_panel import (
    CLUSTER_RMSD_A,
    CONSENSUS_TOP_K,
    PANEL,
    SPACING,
    clean_pose_row,
    cluster_consensus,
    load_gate,
    load_poses,
    read_zinc,
    require_files,
    run,
    slug,
    write_medoid,
)


DEFAULT_ENV_DIR = Path(".venv/e3_docking")
DEFAULT_SOURCE_DIR = Path("results/revision/docking_controls/source_structures")
DEFAULT_REDOCK_DIR = Path("results/revision/docking_controls/redocking_4lxz")
DEFAULT_PANEL_4LXZ_DIR = Path(
    "results/revision/docking_controls/controlled_panel_4lxz"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/revision/docking_controls/receptor_sensitivity_4bkx"
)
REFERENCE_CHAIN = "A"
MOVING_CHAIN = "B"
MIN_SEQUENCE_IDENTITY = 0.85
MAX_CA_RMSD_A = 1.50
MAX_ALIGNED_ZN_DISTANCE_A = 0.75

AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--redocking-dir", type=Path, default=DEFAULT_REDOCK_DIR)
    parser.add_argument("--panel-4lxz-dir", type=Path, default=DEFAULT_PANEL_4LXZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cpu",
        type=int,
        help="CPU threads per Vina run; defaults to the passed redocking value.",
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def parse_chain_ca(path: Path, chain: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    residues: dict[str, dict[str, Any]] = {}
    zinc: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line[:6].strip()
        if line[21:22] != chain:
            continue
        residue_name = line[17:20].strip()
        alternate = line[16:17]
        if record == "ATOM" and line[12:16].strip() == "CA":
            if alternate not in {" ", "A"} or residue_name not in AMINO_ACIDS:
                continue
            residue_key = line[22:27]
            residues.setdefault(
                residue_key,
                {
                    "key": residue_key.strip(),
                    "residue_name": residue_name,
                    "amino_acid": AMINO_ACIDS[residue_name],
                    "coordinate": np.asarray(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    ),
                },
            )
        elif record == "HETATM" and residue_name == "ZN":
            zinc.append(
                np.asarray(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            )
    if len(residues) < 250:
        raise RuntimeError(
            f"Too few C-alpha residues in {path.name} chain {chain}: {len(residues)}"
        )
    if len(zinc) != 1:
        raise RuntimeError(
            f"Expected one zinc in {path.name} chain {chain}; observed={len(zinc)}"
        )
    return list(residues.values()), zinc[0]


def global_sequence_alignment(
    reference: str, moving: str
) -> list[tuple[int, int]]:
    """Needleman-Wunsch alignment with deterministic tie breaking."""
    rows = len(reference) + 1
    columns = len(moving) + 1
    score = np.zeros((rows, columns), dtype=np.int32)
    trace = np.zeros((rows, columns), dtype=np.int8)
    for row in range(1, rows):
        score[row, 0] = score[row - 1, 0] - 2
        trace[row, 0] = 1
    for column in range(1, columns):
        score[0, column] = score[0, column - 1] - 2
        trace[0, column] = 2
    for row in range(1, rows):
        for column in range(1, columns):
            diagonal = score[row - 1, column - 1] + (
                2 if reference[row - 1] == moving[column - 1] else -1
            )
            up = score[row - 1, column] - 2
            left = score[row, column - 1] - 2
            if diagonal >= up and diagonal >= left:
                score[row, column] = diagonal
                trace[row, column] = 0
            elif up >= left:
                score[row, column] = up
                trace[row, column] = 1
            else:
                score[row, column] = left
                trace[row, column] = 2
    row = len(reference)
    column = len(moving)
    pairs: list[tuple[int, int]] = []
    while row or column:
        direction = int(trace[row, column])
        if row and column and direction == 0:
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and (not column or direction == 1):
            row -= 1
        else:
            column -= 1
    return list(reversed(pairs))


def kabsch_transform(
    reference_coordinates: np.ndarray, moving_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if reference_coordinates.shape != moving_coordinates.shape:
        raise ValueError("Reference and moving alignment coordinates differ in shape")
    reference_center = reference_coordinates.mean(axis=0)
    moving_center = moving_coordinates.mean(axis=0)
    reference_centered = reference_coordinates - reference_center
    moving_centered = moving_coordinates - moving_center
    covariance = moving_centered.T @ reference_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1, :] *= -1
        rotation = right_transpose.T @ left.T
    transformed = (rotation @ moving_centered.T).T + reference_center
    rmsd = float(
        np.sqrt(np.mean(np.sum((transformed - reference_coordinates) ** 2, axis=1)))
    )
    return rotation, moving_center, reference_center, rmsd


def transform_coordinate(
    coordinate: np.ndarray,
    rotation: np.ndarray,
    moving_center: np.ndarray,
    reference_center: np.ndarray,
) -> np.ndarray:
    return rotation @ (coordinate - moving_center) + reference_center


def write_aligned_receptor(
    source: Path,
    destination: Path,
    chain: str,
    rotation: np.ndarray,
    moving_center: np.ndarray,
    reference_center: np.ndarray,
) -> None:
    output: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        record = line[:6].strip()
        if line[21:22] != chain:
            continue
        residue_name = line[17:20].strip()
        alternate = line[16:17]
        keep = (
            record == "ATOM" and alternate in {" ", "A"}
        ) or (record == "HETATM" and residue_name == "ZN")
        if not keep:
            continue
        coordinate = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
        transformed = transform_coordinate(
            coordinate, rotation, moving_center, reference_center
        )
        cleaned = line[:16] + (" " if alternate == "A" else alternate) + line[17:]
        cleaned = (
            cleaned[:30]
            + f"{transformed[0]:8.3f}{transformed[1]:8.3f}{transformed[2]:8.3f}"
            + cleaned[54:]
        )
        output.append(cleaned)
    output.extend(["TER", "END"])
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def align_4bkx_to_4lxz(
    reference_path: Path, moving_path: Path, destination: Path
) -> dict[str, Any]:
    reference_residues, reference_zinc = parse_chain_ca(
        reference_path, REFERENCE_CHAIN
    )
    moving_residues, moving_zinc = parse_chain_ca(moving_path, MOVING_CHAIN)
    pairs = global_sequence_alignment(
        "".join(value["amino_acid"] for value in reference_residues),
        "".join(value["amino_acid"] for value in moving_residues),
    )
    if len(pairs) < 250:
        raise RuntimeError(f"Too few aligned residues between 4LXZ and 4BKX: {len(pairs)}")
    identical = [
        pair
        for pair in pairs
        if reference_residues[pair[0]]["amino_acid"]
        == moving_residues[pair[1]]["amino_acid"]
    ]
    identity = len(identical) / len(pairs)
    # Use identical aligned residues for the transform so residue substitutions
    # cannot pull the structural reference fit.
    reference_coordinates = np.vstack(
        [reference_residues[first]["coordinate"] for first, _ in identical]
    )
    moving_coordinates = np.vstack(
        [moving_residues[second]["coordinate"] for _, second in identical]
    )
    rotation, moving_center, reference_center, ca_rmsd = kabsch_transform(
        reference_coordinates, moving_coordinates
    )
    aligned_zinc = transform_coordinate(
        moving_zinc, rotation, moving_center, reference_center
    )
    zinc_distance = float(np.linalg.norm(aligned_zinc - reference_zinc))
    gates = {
        "sequence_identity_ge_0_85": bool(identity >= MIN_SEQUENCE_IDENTITY),
        "ca_rmsd_le_1_50_a": bool(ca_rmsd <= MAX_CA_RMSD_A),
        "aligned_catalytic_zn_distance_le_0_75_a": bool(
            zinc_distance <= MAX_ALIGNED_ZN_DISTANCE_A
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"4BKX-to-4LXZ alignment gate failed: {gates}")
    write_aligned_receptor(
        moving_path,
        destination,
        MOVING_CHAIN,
        rotation,
        moving_center,
        reference_center,
    )
    return {
        "reference": "4LXZ chain A HDAC2",
        "moving": "4BKX chain B HDAC1",
        "reference_ca_residues": len(reference_residues),
        "moving_ca_residues": len(moving_residues),
        "aligned_residue_pairs": len(pairs),
        "identical_aligned_residues_used_for_fit": len(identical),
        "sequence_identity": identity,
        "ca_rmsd_a": ca_rmsd,
        "reference_zinc_xyz": reference_zinc.tolist(),
        "aligned_4bkx_zinc_xyz": aligned_zinc.tolist(),
        "aligned_zinc_distance_a": zinc_distance,
        "rotation_matrix": rotation.tolist(),
        "moving_centroid": moving_center.tolist(),
        "reference_centroid": reference_center.tolist(),
        "gates": gates,
    }


def fixed_frame_symmetry_rmsd(first_path: Path, second_path: Path) -> float:
    first_supplier = Chem.SDMolSupplier(str(first_path), removeHs=False)
    second_supplier = Chem.SDMolSupplier(str(second_path), removeHs=False)
    first_raw = first_supplier[0] if first_supplier else None
    second_raw = second_supplier[0] if second_supplier else None
    if first_raw is None or second_raw is None:
        raise RuntimeError(
            f"Could not read cross-receptor medoids: {first_path}, {second_path}"
        )
    first = Chem.RemoveHs(first_raw)
    second = Chem.RemoveHs(second_raw)
    first_coordinates = np.asarray(first.GetConformer().GetPositions(), dtype=float)
    second_coordinates = np.asarray(second.GetConformer().GetPositions(), dtype=float)
    matches = second.GetSubstructMatches(first, uniquify=False, maxMatches=10000)
    if not matches:
        raise RuntimeError(
            f"Cross-receptor medoids are not graph-isomorphic: {first_path.name}"
        )
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


def known_medoid_path(directory: Path, compound: str) -> Path:
    return directory / "consensus_medoids" / f"{slug(compound)}_consensus_medoid.sdf"


def build_cross_receptor_comparison(
    summary_4lxz: pd.DataFrame,
    summary_4bkx: pd.DataFrame,
    panel_4lxz_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in PANEL:
        compound = entry["compound"]
        first_rows = summary_4lxz.loc[summary_4lxz["compound"] == compound]
        second_rows = summary_4bkx.loc[summary_4bkx["compound"] == compound]
        if len(first_rows) != 1 or len(second_rows) != 1:
            raise RuntimeError(f"Non-unique cross-receptor summary for {compound}")
        first = first_rows.iloc[0]
        second = second_rows.iloc[0]
        stable_first = bool(first["stable_all_seed_consensus"])
        stable_second = bool(second["stable_all_seed_consensus"])
        rmsd = math.nan
        if stable_first and stable_second:
            first_medoid = known_medoid_path(panel_4lxz_dir, compound)
            second_medoid = known_medoid_path(output_dir, compound)
            require_files([first_medoid, second_medoid])
            rmsd = fixed_frame_symmetry_rmsd(first_medoid, second_medoid)

        def numeric(row: pd.Series, key: str) -> float:
            value = row.get(key)
            return math.nan if pd.isna(value) else float(value)

        score_first = numeric(first, "selected_cluster_median_score_kcal_mol")
        score_second = numeric(second, "selected_cluster_median_score_kcal_mol")
        zbg_first = numeric(first, "zbg_min_zn_a")
        zbg_second = numeric(second, "zbg_min_zn_a")
        hetero_first = numeric(first, "nearest_heteroatom_zn_a")
        hetero_second = numeric(second, "nearest_heteroatom_zn_a")
        rows.append(
            {
                "compound": compound,
                "role": entry["role"],
                "stable_consensus_4lxz_hdac2": stable_first,
                "stable_consensus_4bkx_hdac1": stable_second,
                "fixed_frame_medoid_rmsd_a": rmsd,
                "cluster_median_score_4lxz_kcal_mol": score_first,
                "cluster_median_score_4bkx_kcal_mol": score_second,
                "score_delta_4bkx_minus_4lxz_kcal_mol": (
                    score_second - score_first
                    if np.isfinite(score_first) and np.isfinite(score_second)
                    else math.nan
                ),
                "zbg_min_zn_4lxz_a": zbg_first,
                "zbg_min_zn_4bkx_a": zbg_second,
                "nearest_heteroatom_zn_4lxz_a": hetero_first,
                "nearest_heteroatom_zn_4bkx_a": hetero_second,
            }
        )
    return pd.DataFrame(rows)


def plot_receptor_sensitivity(
    comparison: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    display = comparison.copy()
    display["label"] = display["compound"].replace(
        {"Vorinostat_O_methyl_decoy": "Vorinostat O-methyl\ndesigned control"}
    )
    x = np.arange(len(display))
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True)
    axes[0].scatter(
        x,
        display["fixed_frame_medoid_rmsd_a"],
        color="#6C3483",
        s=58,
        zorder=3,
    )
    axes[0].axhline(2.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("4LXZ–4BKX medoid\nRMSD (Å; fixed frame)")

    for index, row in display.iterrows():
        values = [
            row["cluster_median_score_4lxz_kcal_mol"],
            row["cluster_median_score_4bkx_kcal_mol"],
        ]
        axes[1].plot(
            [index - 0.10, index + 0.10],
            values,
            color="#999999",
            linewidth=0.8,
            zorder=1,
        )
        axes[1].scatter(index - 0.10, values[0], color="#176B87", s=42, zorder=3)
        axes[1].scatter(index + 0.10, values[1], color="#C0392B", s=42, zorder=3)
    axes[1].set_ylabel("Consensus-cluster score\n(kcal/mol; descriptive)")

    for index, row in display.iterrows():
        first = row["zbg_min_zn_4lxz_a"]
        second = row["zbg_min_zn_4bkx_a"]
        if pd.isna(first):
            first = row["nearest_heteroatom_zn_4lxz_a"]
        if pd.isna(second):
            second = row["nearest_heteroatom_zn_4bkx_a"]
        axes[2].plot(
            [index - 0.10, index + 0.10],
            [first, second],
            color="#999999",
            linewidth=0.8,
            zorder=1,
        )
        axes[2].scatter(index - 0.10, first, color="#176B87", s=42, zorder=3)
        axes[2].scatter(index + 0.10, second, color="#C0392B", s=42, zorder=3)
    axes[2].axhspan(1.8, 2.8, color="#176B87", alpha=0.10)
    axes[2].set_ylabel("Pre-specified ZBG atom\n(or nearest N/O/S) to Zn (Å)")
    axes[2].set_xticks(x, display["label"], rotation=30, ha="right")
    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].scatter([], [], color="#176B87", label="4LXZ / HDAC2")
    axes[1].scatter([], [], color="#C0392B", label="4BKX / HDAC1")
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle("Frozen docking-panel sensitivity to HDAC receptor structure")
    figure.text(
        0.5,
        0.005,
        "Within-compound sensitivity only; scores are not binding affinities.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    figure.savefig(output_dir / "docking_receptor_sensitivity_4bkx.png", dpi=dpi)
    figure.savefig(output_dir / "docking_receptor_sensitivity_4bkx.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for attribute in (
        "env_dir",
        "source_dir",
        "redocking_dir",
        "panel_4lxz_dir",
        "output_dir",
    ):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    ligand_dir = args.output_dir / "ligands"
    medoid_dir = args.output_dir / "consensus_medoids"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    medoid_dir.mkdir(parents=True, exist_ok=True)

    structure_4lxz = args.source_dir / "4LXZ.pdb"
    structure_4bkx = args.source_dir / "4BKX.pdb"
    source_positive_ligand = args.source_dir / "SHH_ideal.sdf"
    gate_path = args.redocking_dir / "redocking_gate_manifest.json"
    panel_manifest_path = args.panel_4lxz_dir / "controlled_docking_manifest.json"
    panel_summary_path = args.panel_4lxz_dir / "controlled_docking_summary.csv"
    require_files(
        [
            structure_4lxz,
            structure_4bkx,
            source_positive_ligand,
            gate_path,
            panel_manifest_path,
            panel_summary_path,
        ]
    )
    gate = load_gate(gate_path)
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("status") != "passed":
        raise RuntimeError("The frozen 4LXZ controlled panel is not passed")
    if panel_manifest.get("panel") != [dict(value) for value in PANEL]:
        raise RuntimeError("4LXZ panel manifest differs from the frozen source panel")
    seeds = [int(value) for value in gate["seeds"]]
    exhaustiveness = int(gate["exhaustiveness"])
    num_modes = int(gate["num_modes"])
    cpu = int(args.cpu if args.cpu is not None else gate["cpu_per_run"])
    if cpu < 1:
        raise ValueError("--cpu must be positive")
    box = gate["box"]
    center = tuple(float(value) for value in box["center"])
    size = tuple(float(value) for value in box["size_a"])
    npts = tuple(int(value) for value in box["autogrid_npts"])

    env_python = args.env_dir / "bin/python"
    vina = args.env_dir / "bin/vina"
    mk_receptor = args.env_dir / "bin/mk_prepare_receptor.py"
    mk_ligand = args.env_dir / "bin/mk_prepare_ligand.py"
    mk_export = args.env_dir / "bin/mk_export.py"
    autogrid = args.env_dir / "bin/autogrid4"
    zinc_pseudo = args.env_dir / "share/ad4zn/zinc_pseudo.py"
    parameter_source = args.env_dir / "share/ad4zn/AD4Zn.dat"
    require_files(
        [
            env_python,
            vina,
            mk_receptor,
            mk_ligand,
            mk_export,
            autogrid,
            zinc_pseudo,
            parameter_source,
        ]
    )

    aligned_receptor = args.output_dir / "4BKX_chainB_aligned_to_4LXZ_chainA.pdb"
    alignment = align_4bkx_to_4lxz(
        structure_4lxz, structure_4bkx, aligned_receptor
    )
    zinc = read_zinc(aligned_receptor)
    receptor_basename = args.output_dir / "4BKX_chainB_aligned_receptor"
    receptor_pdbqt = args.output_dir / "4BKX_chainB_aligned_receptor.pdbqt"
    receptor_tz = args.output_dir / "4BKX_chainB_aligned_receptor_tz.pdbqt"
    parameter_file = args.output_dir / "AD4Zn.dat"
    run(
        [
            str(mk_receptor),
            "--read_pdb",
            str(aligned_receptor),
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
        log_path=log_dir / "prepare_4bkx_receptor.log",
    )
    run(
        [
            str(env_python),
            str(zinc_pseudo),
            "-r",
            str(receptor_pdbqt),
            "-o",
            str(receptor_tz),
        ],
        cwd=args.output_dir,
        log_path=log_dir / "zinc_pseudo_4bkx.log",
    )
    shutil.copy2(parameter_source, parameter_file)

    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for entry_value in PANEL:
        entry = dict(entry_value)
        compound = entry["compound"]
        compound_slug = slug(compound)
        compound_dir = ligand_dir / compound_slug
        compound_dir.mkdir(parents=True, exist_ok=True)
        if compound == "Vorinostat":
            input_sdf = source_positive_ligand
            input_origin = "PDB CCD ideal SHH coordinates used by passed redocking"
        else:
            input_sdf = (
                args.panel_4lxz_dir
                / "ligands"
                / compound_slug
                / f"{compound_slug}_input.sdf"
            )
            input_origin = "exact deterministic conformer reused from frozen 4LXZ panel"
        require_files([input_sdf])
        input_rows.append(
            {
                **entry,
                "input_origin": input_origin,
                "input_path": str(input_sdf),
                "input_sha256": sha256_file(input_sdf),
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
        local_parameter = compound_dir / parameter_file.name
        shutil.copy2(receptor_tz, local_receptor)
        shutil.copy2(parameter_file, local_parameter)
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
                raise RuntimeError(f"AutoGrid failed for {compound} on 4BKX")
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

        poses = load_poses(entry, seeds, compound_dir, zinc)
        pose_rows.extend(clean_pose_row(pose) for pose in poses)
        clusters, chosen, medoid = cluster_consensus(poses, seeds)
        for cluster in clusters:
            cluster.pop("indices", None)
            cluster_rows.append(cluster)
        top_one = [pose for pose in poses if pose["rank"] == 1]
        base = {
            **entry,
            "seeds": len(seeds),
            "top1_score_median_kcal_mol": float(
                median(pose["score_kcal_mol"] for pose in top_one)
            ),
            "input_sha256": sha256_file(input_sdf),
        }
        if chosen is None or medoid is None:
            summary_rows.append(
                {
                    **base,
                    "stable_all_seed_consensus": False,
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
                    "medoid_sdf": None,
                }
            )
            continue
        medoid_path = medoid_dir / f"{compound_slug}_consensus_medoid.sdf"
        write_medoid(medoid, medoid_path)
        summary_rows.append(
            {
                **base,
                "stable_all_seed_consensus": True,
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
                "medoid_sdf": str(medoid_path),
            }
        )

    summary_4bkx = pd.DataFrame(summary_rows)
    clusters_4bkx = pd.DataFrame(cluster_rows)
    poses_4bkx = pd.DataFrame(pose_rows)
    inputs_4bkx = pd.DataFrame(input_rows)
    summary_4bkx.to_csv(args.output_dir / "docking_summary_4bkx.csv", index=False)
    clusters_4bkx.to_csv(args.output_dir / "docking_clusters_4bkx.csv", index=False)
    poses_4bkx.to_csv(args.output_dir / "docking_pose_metrics_4bkx.csv", index=False)
    inputs_4bkx.to_csv(args.output_dir / "docking_input_manifest_4bkx.csv", index=False)

    summary_4lxz = pd.read_csv(panel_summary_path)
    comparison = build_cross_receptor_comparison(
        summary_4lxz, summary_4bkx, args.panel_4lxz_dir, args.output_dir
    )
    comparison.to_csv(args.output_dir / "cross_receptor_comparison.csv", index=False)
    plot_receptor_sensitivity(comparison, args.output_dir, args.figure_dpi)

    manifest = {
        "schema_version": 1,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "HDAC1 receptor-structure sensitivity",
        "alignment": alignment,
        "receptor_policy": (
            "4BKX chain B protein plus catalytic Zn only; acetate, potassium, sulfate, "
            "chain A, and other heteroatoms removed. Receptor rigidly aligned to 4LXZ "
            "chain A before preparation."
        ),
        "box_policy": (
            "Exact center, size, grid spacing, and grid-point counts reused from the "
            "passed 4LXZ redocking gate after receptor alignment; no 4BKX-specific "
            "candidate-informed box selection."
        ),
        "panel": [dict(value) for value in PANEL],
        "seeds": seeds,
        "exhaustiveness": exhaustiveness,
        "num_modes": num_modes,
        "cpu_per_run": cpu,
        "consensus_top_k": CONSENSUS_TOP_K,
        "cluster_rmsd_a": CLUSTER_RMSD_A,
        "stable_all_seed_consensus_compounds_4bkx": summary_4bkx.loc[
            summary_4bkx["stable_all_seed_consensus"].astype(bool), "compound"
        ].tolist(),
        "unstable_compounds_4bkx": summary_4bkx.loc[
            ~summary_4bkx["stable_all_seed_consensus"].astype(bool), "compound"
        ].tolist(),
        "source_sha256": {
            "4LXZ.pdb": sha256_file(structure_4lxz),
            "4BKX.pdb": sha256_file(structure_4bkx),
            "redocking_gate_manifest.json": sha256_file(gate_path),
            "controlled_docking_manifest.json": sha256_file(panel_manifest_path),
            "controlled_docking_summary.csv": sha256_file(panel_summary_path),
            "AD4Zn.dat": sha256_file(parameter_file),
            "vina": sha256_file(vina),
            "autogrid4": sha256_file(autogrid),
        },
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "rdkit_version": rdBase.rdkitVersion,
        "platform": platform.platform(),
        "gpu_required": False,
        "interpretation_guardrail": (
            "Cross-receptor docking measures sensitivity of a computational pose protocol "
            "to two homologous receptor structures. It does not establish affinity, "
            "isoform selectivity, efficacy, or direct target engagement. A reproducible "
            "pose is not necessarily a mechanistically correct pose."
        ),
        "outputs": [
            "4BKX_chainB_aligned_to_4LXZ_chainA.pdb",
            "docking_summary_4bkx.csv",
            "docking_clusters_4bkx.csv",
            "docking_pose_metrics_4bkx.csv",
            "docking_input_manifest_4bkx.csv",
            "cross_receptor_comparison.csv",
            "consensus_medoids/*.sdf",
            "docking_receptor_sensitivity_4bkx.png",
            "docking_receptor_sensitivity_4bkx.pdf",
            "receptor_sensitivity_4bkx_manifest.json",
            "audit.log",
        ],
    }
    manifest_path = args.output_dir / "receptor_sensitivity_4bkx_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("===== 4BKX-to-4LXZ ALIGNMENT =====")
    print(json.dumps(alignment, indent=2))
    print("\n===== 4BKX / HDAC1 FROZEN PANEL =====")
    display_columns = [
        "compound",
        "stable_all_seed_consensus",
        "selected_cluster_pose_count",
        "selected_cluster_median_score_kcal_mol",
        "zbg_min_zn_a",
        "zbg_max_zn_a",
        "nearest_heteroatom_zn_a",
    ]
    print(summary_4bkx[display_columns].to_string(index=False))
    print("\n===== 4LXZ / 4BKX RECEPTOR SENSITIVITY =====")
    print(comparison.to_string(index=False))
    final_line = "4BKX RECEPTOR SENSITIVITY: PASSED"
    print("\n" + final_line)
    audit = (
        json.dumps(alignment, indent=2)
        + "\n\n"
        + summary_4bkx[display_columns].to_string(index=False)
        + "\n\n"
        + comparison.to_string(index=False)
        + "\n\n"
        + final_line
        + "\n"
        + f"manifest_sha256={sha256_file(manifest_path)}\n"
    )
    (args.output_dir / "audit.log").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
