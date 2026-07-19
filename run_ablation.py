#!/usr/bin/env python3
"""Deprecated submitted ablation entry point.

The legacy file trained two five-epoch exploratory variants and then printed a
hard-coded ``Full Model (Cross-Attention): Pearson R = 0.2841`` value.  That
number was not calculated by the script, and the implemented model has no
cross-attention block.  The output is therefore excluded from the major
revision and must not be cited as an ablation result.

The revised paper reports repeated, leakage-aware comparisons of the frozen
dual-stream atom-token/fingerprint model and fingerprint MLP.  No replacement
ablation claim is made.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "DEPRECATED: the legacy ablation included a hard-coded, mislabelled "
        "cross-attention result and is excluded from the revised evidence set."
    )


if __name__ == "__main__":
    main()
