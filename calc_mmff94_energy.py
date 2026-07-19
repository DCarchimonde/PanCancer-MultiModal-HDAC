#!/usr/bin/env python3
"""Deprecated MMFF94 validation entry point.

The submitted utility used a structure inconsistent with the frozen TC-H-106
identity and described one absolute minimized force-field energy as
``thermodynamic validation`` and an indication of biological quality.  An
isolated MMFF94 conformer energy is neither an affinity, efficacy, stability,
nor thermodynamic validation endpoint.  The analysis and all associated claims
are removed from the major revision.

Ligand conformer generation used for controlled docking is documented in the
versioned docking manifests and must not be interpreted as independent
biological evidence.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "DEPRECATED: absolute MMFF94 conformer energy is not validation and is "
        "excluded from the revised manuscript."
    )


if __name__ == "__main__":
    main()
