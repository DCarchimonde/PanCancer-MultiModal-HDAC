#!/usr/bin/env python3
"""Deprecated legacy proxy benchmark entry point.

This file previously labelled two short, local surrogate implementations as
``DeepCE_Proxy`` and ``ChemCPA_Proxy``.  Those implementations did not reproduce
the native DeepCE or ChemCPA objectives, cell-context inputs, training
protocols, or published evaluation settings and must not be reported as direct
comparisons with those methods.

The major-revision benchmark compares only the frozen dual-stream
atom-token/fingerprint model with an honestly named fingerprint MLP trained on
the identical target matrix and split membership.  Run-level outputs for the
five leakage-aware settings are reported under ``results/revision`` and in the
revision supplementary tables.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "DEPRECATED: this script contained non-equivalent DeepCE/ChemCPA proxy "
        "models and is excluded from the revised manuscript. Use the frozen "
        "dual_stream versus fingerprint_mlp benchmark scripts and manifests "
        "under scripts/ and results/revision/."
    )


if __name__ == "__main__":
    main()
