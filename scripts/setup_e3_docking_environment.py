#!/usr/bin/env python3
"""Create a pinned local E3 docking environment and rerun readiness checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENV_DIR = Path(".venv/e3_docking")
DEFAULT_SOURCE_DIR = Path("results/revision/docking_controls/source_structures")
DEFAULT_OUTPUT_DIR = Path("results/revision/docking_controls/environment")
DEFAULT_READINESS_DIR = Path("results/revision/e2_e3_input_readiness")
VINA_URL = (
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
    "v1.2.7/vina_1.2.7_linux_x86_64"
)
VINA_SHA256 = "f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644"
PACKAGE_SPECS = [
    "rdkit==2025.3.6",
    "meeko==0.7.1",
    "gemmi==0.7.5",
    "vina==1.2.7",
    "spyrmsd==0.9.0",
    "rustworkx==0.18.0",
]
REQUIRED_STRUCTURE_FILES = [
    "4LXZ.pdb",
    "4BKX.pdb",
    "SHH_ideal.sdf",
    "4LXZ_chainA_SHH_bound.sdf",
    "e3_structure_source_manifest.json",
]
USER_AGENT = (
    "PanCancer-MultiModal-HDAC-major-revision/1.0 "
    "(+https://github.com/DCarchimonde/PanCancer-MultiModal-HDAC)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--search-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--skip-readiness-audit", action="store_true")
    parser.add_argument("--refresh-vina", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def validate_structure_bundle(source_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    missing = [name for name in REQUIRED_STRUCTURE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing E3 source structure bundle files in {source_dir}: {missing}. "
            "Run git pull origin major-revision-2026 first."
        )
    manifest_path = source_dir / "e3_structure_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError("E3 source structure manifest is not marked passed")
    observed: dict[str, str] = {}
    for record in manifest.get("source_records", []):
        name = Path(record["output"]).name
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest source output is missing: {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(
                f"Structure-bundle SHA256 mismatch for {name}: "
                f"expected={record['sha256']}, observed={digest}"
            )
        observed[name] = digest
    if not manifest.get("validation", {}).get("4LXZ", {}).get("bound_ligand"):
        raise RuntimeError("E3 source manifest lacks validated 4LXZ bound ligand")
    if manifest.get("validation", {}).get("4BKX", {}).get(
        "has_small_molecule_hdac_inhibitor_cocrystal"
    ) is not False:
        raise RuntimeError("E3 source manifest has an unsafe 4BKX co-crystal designation")
    observed[manifest_path.name] = sha256_file(manifest_path)
    return manifest, observed


def ensure_environment(env_dir: Path) -> Path:
    python_path = env_dir / "bin/python"
    if not python_path.is_file():
        if env_dir.exists() and any(env_dir.iterdir()):
            raise RuntimeError(
                f"Docking environment exists but has no usable Python: {env_dir}. "
                "Remove only this incomplete environment directory and rerun."
            )
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-m",
                "venv",
                "--copies",
                "--system-site-packages",
                str(env_dir),
            ]
        )
    if not python_path.is_file():
        raise RuntimeError(f"Virtual environment Python was not created: {python_path}")
    python_path = python_path.resolve(strict=True)
    run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *PACKAGE_SPECS,
        ]
    )
    return python_path


def download_vina(destination: Path, refresh: bool) -> None:
    if destination.is_file() and not refresh:
        if sha256_file(destination) == VINA_SHA256:
            destination.chmod(0o755)
            return
        raise RuntimeError(
            f"Existing Vina binary has an unexpected SHA256: {destination}. "
            "Use --refresh-vina to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        VINA_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
            observed = hashlib.sha256(raw).hexdigest()
            if observed != VINA_SHA256:
                raise RuntimeError(
                    f"AutoDock Vina SHA256 mismatch: expected={VINA_SHA256}, observed={observed}"
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix="vina-", delete=False
            ) as handle:
                handle.write(raw)
                temporary = Path(handle.name)
            temporary.chmod(0o755)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(
        f"Could not install pinned AutoDock Vina executable; error={last_error}"
    ) from last_error


def environment_versions(python_path: Path) -> dict[str, str]:
    code = """
import importlib
import importlib.metadata as metadata
import json
import platform
packages = ['meeko', 'gemmi', 'vina', 'spyrmsd', 'rustworkx', 'rdkit', 'numpy', 'scipy']
result = {'python': platform.python_version()}
for name in packages:
    try:
        result[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, '__version__', 'AVAILABLE'))
        except ImportError:
            result[name] = 'MISSING'
print(json.dumps(result, sort_keys=True))
"""
    completed = run([str(python_path), "-c", code], capture=True)
    versions = json.loads(completed.stdout.strip().splitlines()[-1])
    missing = [name for name, value in versions.items() if value == "MISSING"]
    if missing:
        raise RuntimeError(f"Pinned docking environment has missing packages: {missing}")
    return versions


def command_version(command: list[str]) -> str:
    completed = run(command, capture=True)
    return completed.stdout.strip().splitlines()[0]


def main() -> None:
    args = parse_args()
    args.env_dir = args.env_dir.expanduser().resolve()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.readiness_dir = args.readiness_dir.expanduser().resolve()
    args.search_root = args.search_root.expanduser().resolve()
    started = time.perf_counter()
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            f"Pinned Vina executable supports Linux x86_64; observed={platform.system()} {machine}"
        )
    structure_manifest, structure_hashes = validate_structure_bundle(args.source_dir)
    python_path = ensure_environment(args.env_dir)
    bin_dir = args.env_dir / "bin"
    vina_path = bin_dir / "vina"
    download_vina(vina_path, args.refresh_vina)

    execution_env = os.environ.copy()
    execution_env["PATH"] = str(bin_dir.resolve()) + os.pathsep + execution_env.get("PATH", "")
    versions = environment_versions(python_path)
    vina_version = command_version([str(vina_path), "--version"])
    command_version([str(bin_dir / "mk_prepare_ligand.py"), "--help"])
    command_version([str(bin_dir / "mk_prepare_receptor.py"), "--help"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": machine,
        "base_python": sys.executable,
        "environment_dir": str(args.env_dir.resolve()),
        "environment_python": str(python_path.resolve()),
        "packages": versions,
        "vina": {
            "executable": str(vina_path.resolve()),
            "version": vina_version,
            "source_url": VINA_URL,
            "sha256": sha256_file(vina_path),
        },
        "meeko_commands": {
            "mk_prepare_ligand.py": str((bin_dir / "mk_prepare_ligand.py").resolve()),
            "mk_prepare_receptor.py": str((bin_dir / "mk_prepare_receptor.py").resolve()),
            "python_3_12_policy": (
                "Meeko is used without ProDy; receptor preparation must pass --read_pdb, "
                "as documented for Python 3.12."
            ),
        },
        "structure_bundle_manifest_sha256": structure_hashes[
            "e3_structure_source_manifest.json"
        ],
        "structure_hashes": structure_hashes,
        "structural_design": structure_manifest["frozen_structural_design"],
        "runtime_seconds": time.perf_counter() - started,
        "gpu_required": False,
        "interpretation_guardrail": (
            "Environment readiness does not validate a docking protocol. The protocol "
            "must still pass 4LXZ-SHH redocking before candidate poses are interpreted."
        ),
    }
    manifest_path = args.output_dir / "docking_environment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_lines = [
        "E3 DOCKING ENVIRONMENT SETUP: PASSED",
        f"python={versions['python']}",
        f"vina={vina_version}",
        f"vina_sha256={manifest['vina']['sha256']}",
        f"meeko={versions['meeko']}",
        f"gemmi={versions['gemmi']}",
        f"spyrmsd={versions['spyrmsd']}",
        "primary_redocking_structure=4LXZ",
        "hdac1_sensitivity_structure=4BKX",
        "gpu_required=false",
        f"manifest_sha256={sha256_file(manifest_path)}",
    ]
    (args.output_dir / "audit.txt").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(audit_lines))

    if not args.skip_readiness_audit:
        if not args.search_root.is_dir():
            raise FileNotFoundError(f"Readiness search root does not exist: {args.search_root}")
        run(
            [
                str(python_path),
                "scripts/audit_e2_e3_input_readiness.py",
                "--search-root",
                str(args.search_root),
                "--output-dir",
                str(args.readiness_dir),
            ],
            env=execution_env,
        )


if __name__ == "__main__":
    main()
