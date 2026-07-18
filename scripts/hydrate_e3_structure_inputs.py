#!/usr/bin/env python3
"""Hydrate and audit the frozen E3 HDAC docking structure bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUEST = Path("revision/requests/e3_structure_inputs.json")
DEFAULT_OUTPUT = Path("results/revision/docking_controls/source_structures")
USER_AGENT = (
    "PanCancer-MultiModal-HDAC-major-revision/1.0 "
    "(+https://github.com/DCarchimonde/PanCancer-MultiModal-HDAC)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, timeout: int, retries: int) -> bytes:
    if not url.startswith("https://"):
        raise RuntimeError(f"Refusing non-HTTPS structure source: {url}")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream,*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            if not raw:
                raise RuntimeError(f"Empty response from {url}")
            return raw
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(
        f"Structure download failed after {retries} attempts: {url}; error={last_error}"
    ) from last_error


def verify_sha256(raw: bytes, expected: str, label: str) -> None:
    observed = sha256_bytes(raw)
    if observed != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {label}: expected={expected}, observed={observed}"
        )


def parse_pdb(raw: bytes) -> dict[str, Any]:
    text = raw.decode("ascii", errors="strict")
    lines = text.splitlines()
    resolution: float | None = None
    protein_chains: set[str] = set()
    hetero_counts: dict[str, int] = {}
    hetero_atoms: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION."):
            match = re.search(r"RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line)
            if match:
                resolution = float(match.group(1))
        if line.startswith("ATOM  "):
            protein_chains.add(line[21].strip())
        if not line.startswith("HETATM"):
            continue
        residue = line[17:20].strip()
        hetero_counts[residue] = hetero_counts.get(residue, 0) + 1
        hetero_atoms.append(
            {
                "atom_name": line[12:16].strip(),
                "residue": residue,
                "chain": line[21].strip(),
                "residue_number": line[22:26].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": line[76:78].strip(),
            }
        )
    if resolution is None:
        raise RuntimeError("PDB file does not contain a parsed X-ray resolution")
    return {
        "text": text,
        "resolution_angstrom": resolution,
        "protein_chains": sorted(protein_chains),
        "hetero_counts": dict(sorted(hetero_counts.items())),
        "hetero_atoms": hetero_atoms,
    }


def normalize_bound_sdf(raw: bytes) -> tuple[bytes, list[tuple[str, tuple[float, float, float]]]]:
    text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
    lines = text.splitlines()
    if len(lines) < 5 or lines[0].strip() != "SHH":
        raise RuntimeError("Unexpected 4LXZ bound-ligand SDF header")
    try:
        atom_count = int(lines[3][0:3])
        bond_count = int(lines[3][3:6])
    except (ValueError, IndexError) as error:
        raise RuntimeError("Could not parse bound-ligand SDF counts line") from error
    if (atom_count, bond_count) != (19, 19):
        raise RuntimeError(
            f"Unexpected bound SHH graph size: atoms={atom_count}, bonds={bond_count}"
        )
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in lines[4 : 4 + atom_count]:
        atoms.append(
            (
                line[31:34].strip(),
                (float(line[0:10]), float(line[10:20]), float(line[20:30])),
            )
        )
    try:
        end_index = lines.index("M  END")
    except ValueError as error:
        raise RuntimeError("Bound-ligand SDF has no M  END record") from error
    normalized_lines = ["SHH", "  RCSB 4LXZ bound ligand chain A", "", *lines[3 : end_index + 1], "$$$$"]
    return ("\n".join(normalized_lines) + "\n").encode("utf-8"), atoms


def select_atoms(
    structure: dict[str, Any],
    *,
    residue: str,
    chain: str,
    residue_number: str | None = None,
) -> list[dict[str, Any]]:
    return [
        atom
        for atom in structure["hetero_atoms"]
        if atom["residue"] == residue
        and atom["chain"] == chain
        and (residue_number is None or atom["residue_number"] == residue_number)
    ]


def validate_4lxz(
    structure: dict[str, Any],
    bound_atoms: list[tuple[str, tuple[float, float, float]]],
) -> dict[str, Any]:
    if structure["resolution_angstrom"] != 1.85:
        raise RuntimeError(f"Unexpected 4LXZ resolution: {structure['resolution_angstrom']}")
    if structure["protein_chains"] != ["A", "B", "C"]:
        raise RuntimeError(f"Unexpected 4LXZ protein chains: {structure['protein_chains']}")
    crystal = select_atoms(structure, residue="SHH", chain="A", residue_number="407")
    zinc = select_atoms(structure, residue="ZN", chain="A")
    if len(crystal) != 19 or len(zinc) != 1:
        raise RuntimeError(
            f"4LXZ chain A expected 19 SHH atoms and one Zn; got {len(crystal)}, {len(zinc)}"
        )
    squared = 0.0
    for sdf_atom, pdb_atom in zip(bound_atoms, crystal, strict=True):
        element, xyz = sdf_atom
        if element.upper() != pdb_atom["element"].upper():
            raise RuntimeError(
                f"Bound-SDF/PDB element mismatch: {element} vs {pdb_atom['element']}"
            )
        squared += sum((left - pdb_atom[key]) ** 2 for left, key in zip(xyz, ("x", "y", "z"), strict=True))
    coordinate_rmsd = math.sqrt(squared / len(crystal))
    if coordinate_rmsd > 1e-6:
        raise RuntimeError(
            f"Bound-SDF/PDB coordinate mismatch for 4LXZ chain A: RMSD={coordinate_rmsd}"
        )
    zn_xyz = (zinc[0]["x"], zinc[0]["y"], zinc[0]["z"])
    heteroatom_distances = {
        atom["atom_name"]: math.dist(zn_xyz, (atom["x"], atom["y"], atom["z"]))
        for atom in crystal
        if atom["element"].upper() in {"O", "N"}
    }
    return {
        "resolution_angstrom": structure["resolution_angstrom"],
        "protein_chains": structure["protein_chains"],
        "bound_ligand": "SHH (SAHA/vorinostat)",
        "bound_ligand_chain": "A",
        "bound_ligand_residue_number": "407",
        "bound_ligand_heavy_atoms": len(crystal),
        "catalytic_zinc_atoms_chain_a": len(zinc),
        "bound_sdf_to_pdb_coordinate_rmsd_angstrom": coordinate_rmsd,
        "crystal_heteroatom_to_zinc_distances_angstrom": heteroatom_distances,
    }


def validate_4bkx(structure: dict[str, Any]) -> dict[str, Any]:
    if structure["resolution_angstrom"] != 3.0:
        raise RuntimeError(f"Unexpected 4BKX resolution: {structure['resolution_angstrom']}")
    if "B" not in structure["protein_chains"]:
        raise RuntimeError(f"4BKX is missing HDAC1 chain B: {structure['protein_chains']}")
    zinc = select_atoms(structure, residue="ZN", chain="B")
    if len(zinc) != 1:
        raise RuntimeError(f"4BKX chain B expected one catalytic Zn; got {len(zinc)}")
    observed = set(structure["hetero_counts"])
    allowed = {"ACT", "HOH", "K", "SO4", "ZN"}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected 4BKX hetero residues: {unexpected}")
    return {
        "resolution_angstrom": structure["resolution_angstrom"],
        "protein_chains": structure["protein_chains"],
        "hdac1_chain": "B",
        "catalytic_zinc_atoms_chain_b": len(zinc),
        "hetero_residue_atom_counts": structure["hetero_counts"],
        "has_small_molecule_hdac_inhibitor_cocrystal": False,
        "role": (
            "HDAC1 receptor-context sensitivity analysis only; it cannot serve as "
            "small-molecule inhibitor redocking validation."
        ),
    }


def main() -> None:
    args = parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    if request.get("schema_version") != 1:
        raise RuntimeError("Unsupported or missing request schema_version")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    structures: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="e3-structures-") as temporary:
        temporary_dir = Path(temporary)
        for entry in request["stable_sources"]:
            raw = download(entry["url"], args.timeout, args.retries)
            verify_sha256(raw, entry["sha256"], entry["output_name"])
            path = temporary_dir / entry["output_name"]
            path.write_bytes(raw)
            if path.suffix.lower() == ".pdb":
                structures[entry["pdb_id"]] = parse_pdb(raw)
            destination = args.output_dir / entry["output_name"]
            shutil.copyfile(path, destination)
            source_records.append(
                {
                    "url": entry["url"],
                    "output": str(destination),
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )

        bound_entry = request["bound_ligand_source"]
        bound_raw = download(bound_entry["url"], args.timeout, args.retries)
        normalized_sdf, bound_atoms = normalize_bound_sdf(bound_raw)
        bound_destination = args.output_dir / bound_entry["output_name"]
        bound_destination.write_bytes(normalized_sdf)
        source_records.append(
            {
                "url": bound_entry["url"],
                "output": str(bound_destination),
                "bytes": len(normalized_sdf),
                "sha256": sha256_bytes(normalized_sdf),
                "normalization": "Removed dynamic ModelServer metadata after M  END",
            }
        )

    validation_4lxz = validate_4lxz(structures["4LXZ"], bound_atoms)
    validation_4bkx = validate_4bkx(structures["4BKX"])
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_path": str(args.request),
        "request_sha256": sha256_bytes(request_raw),
        "source_records": source_records,
        "validation": {"4LXZ": validation_4lxz, "4BKX": validation_4bkx},
        "frozen_structural_design": {
            "primary_protocol_validation": (
                "Redock crystallographic SHH/SAHA (vorinostat) into human HDAC2 4LXZ "
                "and report symmetry-aware heavy-atom RMSD across fixed seeds."
            ),
            "hdac1_sensitivity": (
                "Use human HDAC1 4BKX only as a receptor-context sensitivity analysis; "
                "4BKX has no co-crystallized small-molecule HDAC inhibitor."
            ),
            "candidate_scope": ["Mocetinostat", "TC-H-106"],
            "positive_controls": ["crystallographic SHH/SAHA", "Vorinostat"],
            "negative_controls": "Frozen property-matched non-HDAC decoys selected before docking",
            "interpretation_guardrail": (
                "Docking is structural plausibility support, not measured binding affinity, "
                "target engagement, efficacy, or viability evidence."
            ),
        },
    }
    manifest_path = args.output_dir / "e3_structure_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path = args.output_dir / "audit.txt"
    audit_lines = [
        "E3 STRUCTURE INPUT HYDRATION: PASSED",
        "primary_redocking_structure=4LXZ",
        "primary_cocrystal_ligand=SHH/SAHA/vorinostat",
        "hdac1_sensitivity_structure=4BKX",
        "4BKX_small_molecule_inhibitor_cocrystal=false",
        f"4LXZ_bound_sdf_pdb_rmsd={validation_4lxz['bound_sdf_to_pdb_coordinate_rmsd_angstrom']:.8f}",
        f"manifest_sha256={sha256_file(manifest_path)}",
    ]
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print("===== E3 FROZEN STRUCTURAL DESIGN =====")
    print(json.dumps(manifest["frozen_structural_design"], indent=2))
    print("===== E3 STRUCTURE VALIDATION =====")
    print(json.dumps(manifest["validation"], indent=2))
    print("\n".join(audit_lines))


if __name__ == "__main__":
    main()
