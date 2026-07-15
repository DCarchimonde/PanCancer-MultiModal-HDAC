from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.screen_revision_models import (
    ScreeningDataset,
    collate_screening,
    load_checkpoint_model,
)


def main() -> None:
    root = Path("data/screening_input")
    run_dir = Path(
        "results/revision/screening_spearman/"
        "pair_split/dual_stream/seed_1"
    )
    checkpoint_path = Path(
        "results/revision/benchmarks/"
        "pair_split/dual_stream/seed_1/best_model.pt"
    )

    library = pd.read_csv(root / "screening_library.csv", low_memory=False)
    disease = np.load(root / "disease_matrix.npy")
    observed_mask = np.load(root / "disease_observed_mask.npy").astype(bool)
    cancers = pd.read_csv(root / "disease_manifest.csv")["cancer"].astype(str).tolist()
    gene_ids = pd.read_csv(root / "gene_ids.csv", dtype=str)
    stored_scores = np.load(run_dir / "spearman_reversal_scores.npy")

    name_text = (
        library["display_name"].fillna("").astype(str)
        + " "
        + library["cache_name"].fillna("").astype(str)
        + " "
        + library["compoundinfo_name"].fillna("").astype(str)
    )

    patterns = {
        "Belinostat": r"belinostat",
        "TC-H-106": r"TC-H-106",
        "RG2833": r"RG2833|RG-2833",
        "Tianeptinaline_BG-1010": r"Tianeptinaline|BG-1010",
    }

    selected: list[dict[str, object]] = []
    for label, pattern in patterns.items():
        matches = library[
            name_text.str.contains(pattern, case=False, regex=True, na=False)
        ]
        if matches.empty:
            raise ValueError(f"Could not find compound: {label}")
        row = matches.iloc[0]
        selected.append(
            {
                "label": label,
                "compound_index": int(row["compound_index"]),
                "display_name": str(row["display_name"]),
            }
        )

    dataset = ScreeningDataset(library)
    samples = [dataset[int(item["compound_index"])] for item in selected]
    _, atoms, atom_mask, fingerprints = collate_screening(samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, hidden_dim = load_checkpoint_model(
        checkpoint_path,
        "dual_stream",
        len(gene_ids),
        device,
    )

    atoms = atoms.to(device)
    atom_mask = atom_mask.to(device)
    fingerprints = fingerprints.to(device)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            predictions = model(atoms, atom_mask, fingerprints)

    prediction_array = predictions.float().cpu().numpy()
    cancers_to_check = ["BLCA", "STAD", "UCEC"]
    rows: list[dict[str, object]] = []

    for local_index, item in enumerate(selected):
        compound_index = int(item["compound_index"])
        for cancer in cancers_to_check:
            cancer_index = cancers.index(cancer)
            mask = observed_mask[cancer_index]

            prediction_series = pd.Series(
                prediction_array[local_index, mask],
                dtype="float64",
            )
            disease_series = pd.Series(
                disease[cancer_index, mask],
                dtype="float64",
            )

            pandas_rho = prediction_series.corr(
                disease_series,
                method="spearman",
            )
            pandas_reversal = -float(pandas_rho)
            stored_reversal = float(stored_scores[compound_index, cancer_index])
            absolute_difference = abs(pandas_reversal - stored_reversal)

            rows.append(
                {
                    "compound": item["label"],
                    "compound_index": compound_index,
                    "cancer": cancer,
                    "pandas_rho": pandas_rho,
                    "pandas_reversal": pandas_reversal,
                    "stored_reversal": stored_reversal,
                    "absolute_difference": absolute_difference,
                }
            )

    result = pd.DataFrame(rows)
    print("\n===== PANDAS VS SCREENING IMPLEMENTATION =====")
    print(result.to_string(index=False))

    maximum_difference = float(result["absolute_difference"].max())
    print("\nMaximum absolute difference:", maximum_difference)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Hidden dimension:", hidden_dim)

    values = result[
        ["pandas_reversal", "stored_reversal", "absolute_difference"]
    ].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value found during equivalence validation")
    if maximum_difference >= 1e-6:
        raise AssertionError(
            "Spearman implementation differs from pandas by "
            f"{maximum_difference}"
        )

    print("\nSPEARMAN PANDAS EQUIVALENCE: PASSED")


if __name__ == "__main__":
    main()
