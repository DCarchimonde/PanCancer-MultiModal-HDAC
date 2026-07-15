from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.screen_revision_models import (  # noqa: E402
    ScreeningDataset,
    collate_screening,
    load_checkpoint_model,
)
from scripts.screen_spearman_sensitivity import (  # noqa: E402
    prepare_disease_rank_groups,
    spearman_reversal_scores,
)


def find_first_index(library: pd.DataFrame, pattern: str) -> int:
    text = (
        library["display_name"].fillna("").astype(str)
        + " "
        + library["cache_name"].fillna("").astype(str)
        + " "
        + library["compoundinfo_name"].fillna("").astype(str)
    )
    matches = library[text.str.contains(pattern, case=False, regex=True, na=False)]
    if matches.empty:
        raise ValueError(f"Could not find compound pattern: {pattern}")
    return int(matches.iloc[0]["compound_index"])


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

    selected_patterns = {
        "Belinostat": r"belinostat",
        "TC-H-106": r"tc[- ]?h[- ]?106",
        "RG2833": r"rg[- ]?2833",
        "Tianeptinaline_BG-1010": r"tianeptinaline|bg[- ]?1010",
    }
    selected = {
        label: find_first_index(library, pattern)
        for label, pattern in selected_patterns.items()
    }

    dataset = ScreeningDataset(library)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, hidden_dim = load_checkpoint_model(
        checkpoint_path,
        "dual_stream",
        len(gene_ids),
        device,
    )
    prepared_groups = prepare_disease_rank_groups(disease, observed_mask)

    validation_rows: list[dict[str, object]] = []
    cancers_to_check = ["BLCA", "STAD", "UCEC"]
    batch_size = 128

    grouped_indices: dict[int, list[tuple[str, int]]] = {}
    for label, compound_index in selected.items():
        batch_start = (compound_index // batch_size) * batch_size
        grouped_indices.setdefault(batch_start, []).append((label, compound_index))

    for batch_start, members in grouped_indices.items():
        batch_end = min(batch_start + batch_size, len(dataset))
        samples = [dataset[index] for index in range(batch_start, batch_end)]
        indices, atoms, atom_mask, fingerprints = collate_screening(samples)

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
        vectorized_scores = spearman_reversal_scores(
            prediction_array,
            prepared_groups,
            len(cancers),
        )

        position_lookup = {
            int(compound_index): local_position
            for local_position, compound_index in enumerate(indices.tolist())
        }

        for label, compound_index in members:
            local_position = position_lookup[compound_index]
            prediction = prediction_array[local_position]

            for cancer in cancers_to_check:
                cancer_index = cancers.index(cancer)
                mask = observed_mask[cancer_index]

                pandas_rho = pd.Series(
                    prediction[mask],
                    dtype="float64",
                ).corr(
                    pd.Series(disease[cancer_index, mask], dtype="float64"),
                    method="spearman",
                )
                pandas_reversal = -float(pandas_rho)
                vectorized_reversal = float(
                    vectorized_scores[local_position, cancer_index]
                )
                stored_reversal = float(
                    stored_scores[compound_index, cancer_index]
                )

                validation_rows.append({
                    "compound": label,
                    "compound_index": compound_index,
                    "cancer": cancer,
                    "pandas_reversal": pandas_reversal,
                    "vectorized_reversal": vectorized_reversal,
                    "stored_reversal": stored_reversal,
                    "abs_diff_pandas_vs_vectorized": abs(
                        pandas_reversal - vectorized_reversal
                    ),
                    "abs_diff_vectorized_vs_stored": abs(
                        vectorized_reversal - stored_reversal
                    ),
                })

    validation = pd.DataFrame(validation_rows)
    print("\n===== PANDAS / VECTORIZED / STORED SPEARMAN CHECK =====")
    print(validation.to_string(index=False))

    max_formula_difference = float(
        validation["abs_diff_pandas_vs_vectorized"].max()
    )
    max_stored_difference = float(
        validation["abs_diff_vectorized_vs_stored"].max()
    )

    print("\nMaximum pandas-vs-vectorized difference:", max_formula_difference)
    print("Maximum vectorized-vs-stored difference:", max_stored_difference)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Hidden dimension:", hidden_dim)

    assert max_formula_difference < 1e-6, (
        "Vectorized Spearman implementation differs from pandas by "
        f"{max_formula_difference}"
    )
    assert max_stored_difference < 1e-5, (
        "Stored screening scores differ from exact-batch recomputation by "
        f"{max_stored_difference}"
    )
    print("\nSPEARMAN PANDAS EQUIVALENCE: PASSED")

    ranks = rankdata(-stored_scores, method="average", axis=0)
    candidate_patterns = {
        "TC-H-106": r"tc[- ]?h[- ]?106",
        "RG2833": r"rg[- ]?2833",
        "Tianeptinaline_or_BG-1010": r"tianeptinaline|bg[- ]?1010",
        "Belinostat": r"belinostat",
        "PCI-24781": r"pci[- ]?24781|abexinostat",
        "Panobinostat": r"panobinostat|lbh[- ]?589",
        "Mocetinostat": r"mocetinostat|mgcd[- ]?0103",
        "Vorinostat": r"vorinostat|saha",
    }

    text = (
        library["display_name"].fillna("").astype(str)
        + " "
        + library["cache_name"].fillna("").astype(str)
        + " "
        + library["compoundinfo_name"].fillna("").astype(str)
    )

    rank_rows: list[dict[str, object]] = []
    for label, pattern in candidate_patterns.items():
        matches = library[text.str.contains(pattern, case=False, regex=True, na=False)]
        for row in matches.itertuples(index=False):
            compound_index = int(row.compound_index)
            candidate_ranks = ranks[compound_index]
            rank_rows.append({
                "candidate_label": label,
                "display_name": row.display_name,
                "compoundinfo_name": row.compoundinfo_name,
                "is_hdac_annotated": row.is_hdac_annotated,
                "minimum_rank": int(candidate_ranks.min()),
                "median_rank": float(np.median(candidate_ranks)),
                "mean_rank": float(candidate_ranks.mean()),
                "top100_cancers": int((candidate_ranks <= 100).sum()),
                "top500_cancers": int((candidate_ranks <= 500).sum()),
            })

    rank_summary = pd.DataFrame(rank_rows).sort_values(
        ["median_rank", "mean_rank"],
        ascending=[True, True],
    )
    print("\n===== SEED-1 SPEARMAN CANDIDATE RANKS =====")
    print(rank_summary.to_string(index=False))


if __name__ == "__main__":
    main()
