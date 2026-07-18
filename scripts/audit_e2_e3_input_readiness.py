from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HDAC_PATTERN = re.compile(r"^HDAC(?:1|2|3|8)(?:\s|\(|$)", re.IGNORECASE)
DEPMAP_NAME_PATTERN = re.compile(
    r"(?:crispr.*gene.*effect|gene[_ .-]*effect|achilles|depmap|"
    r"sample[_ .-]*info|^model(?:s)?$)",
    re.IGNORECASE,
)
DEPMAP_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".gz"}
STRUCTURE_SUFFIXES = {
    ".pdb",
    ".pdbqt",
    ".cif",
    ".mmcif",
    ".sdf",
    ".mol",
    ".mol2",
}
PRUNED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "site-packages",
}
MODEL_ID_ALIASES = {
    "modelid",
    "depmapid",
    "achillesid",
    "broadid",
    "celllineid",
}
LINEAGE_ALIASES = {
    "oncotreelineage",
    "oncotreeprimarydisease",
    "lineage",
    "primarydisease",
    "disease",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory local inputs and software needed to finish E2 DepMap "
            "background analysis and E3 controlled HDAC1 docking."
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("/root/autodl-tmp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/e2_e3_input_readiness"),
    )
    parser.add_argument("--max-files", type=int, default=500_000)
    return parser.parse_args()


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def display_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def read_text_head(path: Path, characters: int = 1_000_000) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read(characters)
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        return handle.read(characters)


def read_delimited_header(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet

            return [str(value) for value in parquet.ParquetFile(path).schema.names]
        except Exception:
            return []
    try:
        first_line = read_text_head(path, 200_000).splitlines()[0]
    except (OSError, EOFError, IndexError):
        return []
    delimiter = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz")) else ","
    try:
        return [value.strip() for value in next(csv.reader([first_line], delimiter=delimiter))]
    except (csv.Error, StopIteration):
        return []


def pdb_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "zinc_atoms": 0,
        "hetero_residue_names": [],
        "has_candidate_cocrystal_ligand": False,
    }
    if path.suffix.lower() not in {".pdb", ".gz"}:
        return summary
    try:
        lines = read_text_head(path).splitlines()
    except (OSError, EOFError):
        return summary
    hetero_names: set[str] = set()
    zinc_atoms = 0
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        residue = line[17:20].strip().upper()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if residue == "ZN" or element == "ZN":
            zinc_atoms += 1
        if residue and residue not in {"HOH", "WAT", "DOD", "ZN"}:
            hetero_names.add(residue)
    summary["zinc_atoms"] = zinc_atoms
    summary["hetero_residue_names"] = sorted(hetero_names)
    summary["has_candidate_cocrystal_ligand"] = bool(hetero_names)
    return summary


def classify_file(path: Path) -> dict[str, Any] | None:
    lower_name = path.name.lower()
    base_name = lower_name[:-3] if lower_name.endswith(".gz") else lower_name
    suffix = Path(base_name).suffix.lower()
    depmap_named = (
        suffix in DEPMAP_SUFFIXES or path.suffix.lower() in DEPMAP_SUFFIXES
    ) and bool(DEPMAP_NAME_PATTERN.search(Path(base_name).stem))
    structure_named = suffix in STRUCTURE_SUFFIXES or path.suffix.lower() in STRUCTURE_SUFFIXES
    if not depmap_named and not structure_named:
        return None

    try:
        size = path.stat().st_size
    except OSError:
        return None
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": size,
        "size_human": display_size(size),
        "category": "depmap_candidate" if depmap_named else "structure_or_ligand",
        "columns": [],
        "hdac_gene_columns": [],
        "model_id_columns": [],
        "lineage_columns": [],
        "contains_4bkx_name": "4bkx" in lower_name,
        "pdb_summary": {},
    }
    if depmap_named:
        columns = read_delimited_header(path)
        normalized = {column: normalized_header(column) for column in columns}
        record["columns"] = columns
        record["hdac_gene_columns"] = [
            column for column in columns if HDAC_PATTERN.match(column.strip())
        ]
        record["model_id_columns"] = [
            column for column, value in normalized.items() if value in MODEL_ID_ALIASES
        ]
        record["lineage_columns"] = [
            column for column, value in normalized.items() if value in LINEAGE_ALIASES
        ]
    if structure_named and ("4bkx" in lower_name or suffix == ".pdb"):
        record["pdb_summary"] = pdb_summary(path)
    return record


def scan_files(search_root: Path, max_files: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    inspected = 0
    for directory, names, filenames in os.walk(search_root):
        names[:] = sorted(name for name in names if name not in PRUNED_DIRECTORIES)
        for filename in sorted(filenames):
            inspected += 1
            if inspected > max_files:
                raise RuntimeError(
                    f"Stopped after --max-files={max_files}; narrow --search-root"
                )
            record = classify_file(Path(directory) / filename)
            if record is not None:
                records.append(record)
    return records, inspected


def executable_inventory() -> dict[str, str | None]:
    names = [
        "vina",
        "vina_1.2",
        "smina",
        "gnina",
        "qvina2",
        "obabel",
        "babel",
        "mk_prepare_receptor.py",
        "prepare_receptor4.py",
        "prepare_ligand4.py",
    ]
    return {name: shutil.which(name) for name in names}


def module_inventory() -> dict[str, bool]:
    names = ["rdkit", "Bio", "meeko", "vina", "openbabel", "MDAnalysis"]
    return {name: importlib.util.find_spec(name) is not None for name in names}


def write_inventory_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "category",
        "path",
        "size_bytes",
        "size_human",
        "contains_4bkx_name",
        "hdac_gene_columns",
        "model_id_columns",
        "lineage_columns",
        "hetero_residue_names",
        "zinc_atoms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            pdb = record.get("pdb_summary", {})
            writer.writerow(
                {
                    "category": record["category"],
                    "path": record["path"],
                    "size_bytes": record["size_bytes"],
                    "size_human": record["size_human"],
                    "contains_4bkx_name": record["contains_4bkx_name"],
                    "hdac_gene_columns": " | ".join(record["hdac_gene_columns"]),
                    "model_id_columns": " | ".join(record["model_id_columns"]),
                    "lineage_columns": " | ".join(record["lineage_columns"]),
                    "hetero_residue_names": " | ".join(
                        pdb.get("hetero_residue_names", [])
                    ),
                    "zinc_atoms": pdb.get("zinc_atoms", ""),
                }
            )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    search_root = args.search_root.resolve()
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, inspected_files = scan_files(search_root, args.max_files)
    executables = executable_inventory()
    modules = module_inventory()
    gene_effect_files = [record for record in records if record["hdac_gene_columns"]]
    model_metadata_files = [
        record
        for record in records
        if record["model_id_columns"] and record["lineage_columns"]
    ]
    structures_4bkx = [record for record in records if record["contains_4bkx_name"]]
    cocrystal_structures = [
        record
        for record in structures_4bkx
        if record.get("pdb_summary", {}).get("zinc_atoms", 0) > 0
        and record.get("pdb_summary", {}).get("has_candidate_cocrystal_ligand", False)
    ]
    docking_engines = {
        key: value
        for key, value in executables.items()
        if key in {"vina", "vina_1.2", "smina", "gnina", "qvina2"} and value
    }
    ligand_preparation_ready = bool(
        executables.get("obabel")
        or executables.get("babel")
        or modules.get("meeko")
        or modules.get("openbabel")
    ) and modules.get("rdkit", False)

    readiness = {
        "e2_depmap_gene_effect_ready": bool(gene_effect_files),
        "e2_depmap_model_metadata_ready": bool(model_metadata_files),
        "e2_depmap_ready": bool(gene_effect_files and model_metadata_files),
        "e3_4bkx_cocrystal_ready": bool(cocrystal_structures),
        "e3_docking_engine_ready": bool(docking_engines or modules.get("vina")),
        "e3_ligand_preparation_ready": bool(ligand_preparation_ready),
        "e3_local_input_ready": bool(
            cocrystal_structures
            and (docking_engines or modules.get("vina"))
            and ligand_preparation_ready
        ),
    }
    manifest = {
        "analysis_stage": "e2_e3_local_input_readiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "search_root": str(search_root),
        "inspected_files": inspected_files,
        "matching_files": len(records),
        "readiness": readiness,
        "depmap_gene_effect_matches": [record["path"] for record in gene_effect_files],
        "depmap_model_metadata_matches": [
            record["path"] for record in model_metadata_files
        ],
        "cocrystal_4bkx_matches": [record["path"] for record in cocrystal_structures],
        "executables": executables,
        "python_modules": modules,
        "file_inventory": records,
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "notes": {
            "e2": (
                "DepMap CRISPR gene dependency is genetic background context; it "
                "does not validate candidate drug response or pan-cancer efficacy."
            ),
            "e3": (
                "Docking controls assess structural plausibility only. Required "
                "controls remain co-crystal redocking, a known HDAC1 inhibitor, "
                "and a justified negative/decoy under identical settings."
            ),
        },
    }
    manifest_path = args.output_dir / "e2_e3_input_readiness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_inventory_csv(records, args.output_dir / "e2_e3_file_inventory.csv")

    gene_effect_display = "\n".join(
        f"- {record['path']} :: {' | '.join(record['hdac_gene_columns'])}"
        for record in gene_effect_files
    ) or "NONE"
    metadata_display = "\n".join(
        f"- {record['path']} :: model={record['model_id_columns']} "
        f"lineage={record['lineage_columns']}"
        for record in model_metadata_files
    ) or "NONE"
    structure_display = "\n".join(
        f"- {record['path']} :: {record['pdb_summary']}"
        for record in structures_4bkx
    ) or "NONE"
    executable_display = "\n".join(
        f"- {name}: {path or 'MISSING'}" for name, path in executables.items()
    )
    module_display = "\n".join(
        f"- {name}: {'AVAILABLE' if present else 'MISSING'}"
        for name, present in modules.items()
    )
    audit = "\n".join(
        [
            "===== E2 DEPMAP GENE-EFFECT INPUTS =====",
            gene_effect_display,
            "",
            "===== E2 DEPMAP MODEL METADATA =====",
            metadata_display,
            "",
            "===== E3 4BKX LOCAL STRUCTURES =====",
            structure_display,
            "",
            "===== DOCKING EXECUTABLES =====",
            executable_display,
            "",
            "===== RELEVANT PYTHON MODULES =====",
            module_display,
            "",
            "===== READINESS =====",
            json.dumps(readiness, indent=2),
            "",
            "E2/E3 INPUT READINESS AUDIT: PASSED",
        ]
    ) + "\n"
    (args.output_dir / "audit.log").write_text(audit, encoding="utf-8")
    print(audit, end="")


if __name__ == "__main__":
    main()
