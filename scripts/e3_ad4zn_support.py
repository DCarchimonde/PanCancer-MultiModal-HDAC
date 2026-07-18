#!/usr/bin/env python3
"""Pinned AutoDock4Zn resources and AutoGrid build helpers for E3."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


AUTOGRID_COMMIT = "6d2847beaeac8ff43ca99094707fd74e3ca1ff37"
AUTOGRID_URL = (
    "https://github.com/ccsb-scripps/AutoGrid/archive/"
    f"{AUTOGRID_COMMIT}.tar.gz"
)
AUTOGRID_ARCHIVE_SHA256 = (
    "4d0bd83a446fd81577f4fc492299e22f131245589e1782e0532aecf3435e772a"
)
VINA_RESOURCE_COMMIT = "3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645"
ZINC_PSEUDO_URL = (
    "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/"
    f"{VINA_RESOURCE_COMMIT}/example/autodock_scripts/zinc_pseudo.py"
)
ZINC_PSEUDO_SHA256 = (
    "ccfa97e10614b30d32839935d3fc72a3038d43206caf5c1e7f9e0da96693e88d"
)
AD4ZN_URL = (
    "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/"
    f"{VINA_RESOURCE_COMMIT}/data/AD4Zn.dat"
)
AD4ZN_SHA256 = (
    "12b45d377f081c9f3dc25fba2d4585bb01b8367023f309769e441b8457fb1c00"
)
USER_AGENT = (
    "PanCancer-MultiModal-HDAC-major-revision/1.0 "
    "(+https://github.com/DCarchimonde/PanCancer-MultiModal-HDAC)"
)

AUTOGRID_SOURCES = [
    "main.cpp",
    "bondmanager.cpp",
    "bhtree.cpp",
    "check_size.cpp",
    "setflags.cpp",
    "ad4_shared/timesys.cc",
    "ad4_shared/timesyshms.cc",
    "ad4_shared/printhms.cc",
    "prHMSfixed.cpp",
    "ad4_shared/printdate.cc",
    "strindex.cpp",
    "banner.cpp",
    "gpfparser.cpp",
    "parsetypes.cpp",
    "atom_parameter_manager.cpp",
    "ad4_shared/read_parameter_library.cc",
    "ad4_shared/parse_param_line.cc",
    "distdepdiel.cpp",
    "ad4_shared/memalloc.cc",
    "calc_vina_potential.cpp",
    "ad4_shared/threadlog.cc",
    "ad4_shared/targetfile.cc",
    "ad4_shared/stop.cc",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, expected_sha256: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
            observed = hashlib.sha256(raw).hexdigest()
            if observed != expected_sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {url}: expected={expected_sha256}, "
                    f"observed={observed}"
                )
            return raw
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"Could not download pinned resource {url}: {last_error}") from last_error


def _atomic_write(path: Path, raw: bytes, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f"{path.name}-", delete=False
    ) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    if executable:
        temporary.chmod(0o755)
    temporary.replace(path)


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in AutoGrid archive: {member.name}")
        handle.extractall(destination, filter="data")
    directories = [path for path in destination.iterdir() if path.is_dir()]
    if len(directories) != 1:
        raise RuntimeError("Pinned AutoGrid archive has an unexpected top-level layout")
    return directories[0]


def _parameter_rows(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def _write_default_parameters(source_dir: Path) -> None:
    output = source_dir / "default_parameters.h"
    blocks = [
        ("param_string_4_0", source_dir / "ad4_shared/AD4_parameters.dat"),
        ("param_string_4_1", source_dir / "ad4_shared/AD4.1_bound.dat"),
    ]
    lines: list[str] = []
    for variable, source in blocks:
        lines.append(f"const char *{variable}[MAX_LINES] = {{")
        for row in _parameter_rows(source):
            escaped = row.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'"{escaped}\\n",')
        lines.append("};")
    lines.append("// EOF")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _build_autogrid(destination: Path) -> dict[str, Any]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError(
            "AutoDock4Zn requires g++ to compile the pinned official AutoGrid source. "
            "Install the system C++ compiler and rerun this setup."
        )
    raw = _download(AUTOGRID_URL, AUTOGRID_ARCHIVE_SHA256)
    with tempfile.TemporaryDirectory(prefix="e3-autogrid-build-") as temporary_name:
        temporary = Path(temporary_name)
        archive = temporary / "autogrid.tar.gz"
        archive.write_bytes(raw)
        source_dir = _safe_extract(archive, temporary / "source")
        _write_default_parameters(source_dir)
        built = temporary / "autogrid4"
        command = [
            compiler,
            "-std=c++11",
            "-O3",
            "-Wno-write-strings",
            "-DUSE_8A_NBCUTOFF",
            "-DUSE_DOUBLE",
            '-DPACKAGE_BUGREPORT="autodock@scripps.edu"',
            "-fopenmp",
            "-I.",
            "-Iad4_shared",
            *AUTOGRID_SOURCES,
            "-lm",
            "-o",
            str(built),
        ]
        completed = _run(command, cwd=source_dir)
        if not built.is_file():
            raise RuntimeError("AutoGrid compilation completed without an executable")
        _atomic_write(destination, built.read_bytes(), executable=True)
    version_output = _run([str(destination), "--version"]).stdout
    first_line = next(
        (line.strip() for line in version_output.splitlines() if line.strip()), "UNKNOWN"
    )
    compiler_version = _run([compiler, "--version"]).stdout.splitlines()[0]
    return {
        "version": first_line,
        "sha256": sha256_file(destination),
        "compiler": compiler,
        "compiler_version": compiler_version,
        "compile_stdout_tail": completed.stdout.splitlines()[-20:],
    }


def install_ad4zn(env_dir: Path, refresh: bool = False) -> dict[str, Any]:
    """Install pinned AutoGrid4 and official AutoDock4Zn resources into an E3 venv."""
    env_dir = env_dir.resolve()
    bin_dir = env_dir / "bin"
    share_dir = env_dir / "share/ad4zn"
    autogrid = bin_dir / "autogrid4"
    zinc_pseudo = share_dir / "zinc_pseudo.py"
    parameter_file = share_dir / "AD4Zn.dat"
    resource_manifest = share_dir / "resource_manifest.json"

    expected_resource_state = (
        zinc_pseudo.is_file()
        and sha256_file(zinc_pseudo) == ZINC_PSEUDO_SHA256
        and parameter_file.is_file()
        and sha256_file(parameter_file) == AD4ZN_SHA256
    )
    if refresh or not expected_resource_state:
        _atomic_write(
            zinc_pseudo,
            _download(ZINC_PSEUDO_URL, ZINC_PSEUDO_SHA256),
            executable=True,
        )
        _atomic_write(parameter_file, _download(AD4ZN_URL, AD4ZN_SHA256))

    previous: dict[str, Any] = {}
    if resource_manifest.is_file():
        try:
            previous = json.loads(resource_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    source_matches = previous.get("autogrid_source_commit") == AUTOGRID_COMMIT
    if refresh or not autogrid.is_file() or not source_matches:
        build = _build_autogrid(autogrid)
    else:
        version_output = _run([str(autogrid), "--version"]).stdout
        build = {
            "version": next(
                (line.strip() for line in version_output.splitlines() if line.strip()),
                "UNKNOWN",
            ),
            "sha256": sha256_file(autogrid),
            "compiler": previous.get("build", {}).get("compiler", "recorded previously"),
            "compiler_version": previous.get("build", {}).get(
                "compiler_version", "recorded previously"
            ),
        }

    manifest = {
        "schema_version": 1,
        "platform": platform.platform(),
        "autogrid_source_commit": AUTOGRID_COMMIT,
        "autogrid_source_url": AUTOGRID_URL,
        "autogrid_archive_sha256": AUTOGRID_ARCHIVE_SHA256,
        "vina_resource_commit": VINA_RESOURCE_COMMIT,
        "zinc_pseudo_url": ZINC_PSEUDO_URL,
        "zinc_pseudo_sha256": sha256_file(zinc_pseudo),
        "ad4zn_url": AD4ZN_URL,
        "ad4zn_sha256": sha256_file(parameter_file),
        "autogrid_executable": str(autogrid),
        "zinc_pseudo_script": str(zinc_pseudo),
        "parameter_file": str(parameter_file),
        "build": build,
    }
    share_dir.mkdir(parents=True, exist_ok=True)
    resource_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["resource_manifest"] = str(resource_manifest)
    manifest["resource_manifest_sha256"] = sha256_file(resource_manifest)
    return manifest


def pdbqt_atom_types(path: Path) -> list[str]:
    types = {
        line.split()[-1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    }
    if not types:
        raise RuntimeError(f"No PDBQT atom types found in {path}")
    return sorted(types)


def write_ad4zn_gpf(
    path: Path,
    receptor: Path,
    ligand: Path,
    parameter_file_name: str,
    center: tuple[float, float, float],
    npts: tuple[int, int, int],
    spacing: float = 0.375,
) -> str:
    """Write the official AutoDock4Zn GPF semantics without legacy Python-2 ADT."""
    prefix = receptor.stem
    receptor_types = pdbqt_atom_types(receptor)
    ligand_types = pdbqt_atom_types(ligand)
    lines = [
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"parameter_file {parameter_file_name}",
        f"gridfld {prefix}.maps.fld",
        f"spacing {spacing}",
        f"receptor_types {' '.join(receptor_types)}",
        f"ligand_types {' '.join(ligand_types)}",
        f"receptor {receptor.name}",
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}",
        "smooth 0.5",
        *[f"map {prefix}.{atom_type}.map" for atom_type in ligand_types],
        f"elecmap {prefix}.e.map",
        f"dsolvmap {prefix}.d.map",
        "dielectric -0.1465",
        "nbp_r_eps 0.25 23.2135 12 6 NA TZ",
        "nbp_r_eps 2.1   3.8453 12 6 OA Zn",
        "nbp_r_eps 2.25  7.5914 12 6 SA Zn",
        "nbp_r_eps 1.0   0.0    12 6 HD Zn",
        "nbp_r_eps 2.0   0.0060 12 6 NA Zn",
        "nbp_r_eps 2.0   0.2966 12 6  N Zn",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return prefix
