from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import Dataset

ATOM_TYPES = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
)
ATOM_INDEX = {symbol: index for index, symbol in enumerate(ATOM_TYPES)}
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


@dataclass(frozen=True)
class MoleculeFeatures:
    atom_features: torch.Tensor
    fingerprint: torch.Tensor


@lru_cache(maxsize=50_000)
def featurize_smiles(smiles: str) -> MoleculeFeatures:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    atoms = np.zeros((mol.GetNumAtoms(), len(ATOM_TYPES)), dtype=np.float32)
    unknown_index = ATOM_INDEX["Unknown"]
    for atom_index, atom in enumerate(mol.GetAtoms()):
        feature_index = ATOM_INDEX.get(atom.GetSymbol(), unknown_index)
        atoms[atom_index, feature_index] = 1.0

    fingerprint = MORGAN_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    return MoleculeFeatures(
        atom_features=torch.from_numpy(atoms),
        fingerprint=torch.from_numpy(fingerprint),
    )


class CachedLINCSExpressionDataset(Dataset):
    """Join a split CSV to the expression cache by sig_id."""

    def __init__(
        self,
        split_csv: Path,
        cache_metadata_csv: Path,
        expression_npy: Path,
        max_samples: int | None = None,
    ) -> None:
        split = pd.read_csv(split_csv, low_memory=False)
        cache_metadata = pd.read_csv(
            cache_metadata_csv,
            usecols=["cache_row", "sig_id"],
            low_memory=False,
        )
        required = {"sig_id", "canonical_smiles"}
        missing = required - set(split.columns)
        if missing:
            raise ValueError(f"Split file is missing columns: {sorted(missing)}")

        frame = split.merge(cache_metadata, on="sig_id", how="inner", validate="one_to_one")
        if len(frame) != len(split):
            missing_count = len(split) - len(frame)
            raise ValueError(f"{missing_count} split signatures are absent from the expression cache")
        if max_samples is not None:
            frame = frame.head(max_samples).copy()

        self.frame = frame.reset_index(drop=True)
        self.expression = np.load(expression_npy, mmap_mode="r")
        self.sig_ids = self.frame["sig_id"].astype(str).tolist()
        self.smiles = self.frame["canonical_smiles"].astype(str).tolist()
        self.cache_rows = self.frame["cache_row"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = featurize_smiles(self.smiles[index])
        target = np.asarray(self.expression[self.cache_rows[index]], dtype=np.float32)
        return (
            self.sig_ids[index],
            features.atom_features,
            features.fingerprint,
            torch.from_numpy(target.copy()),
        )


def warm_molecule_cache(smiles_values: Iterable[str]) -> None:
    for smiles in dict.fromkeys(str(value) for value in smiles_values):
        featurize_smiles(smiles)


def collate_cached_profiles(
    batch: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sig_ids, atom_arrays, fingerprints, targets = zip(*batch)
    max_atoms = max(array.shape[0] for array in atom_arrays)
    feature_dim = atom_arrays[0].shape[1]

    atom_batch = torch.zeros((len(batch), max_atoms, feature_dim), dtype=torch.float32)
    atom_mask = torch.zeros((len(batch), max_atoms), dtype=torch.bool)
    for row, atom_features in enumerate(atom_arrays):
        count = atom_features.shape[0]
        atom_batch[row, :count] = atom_features
        atom_mask[row, :count] = True

    return (
        list(sig_ids),
        atom_batch,
        atom_mask,
        torch.stack(fingerprints),
        torch.stack(targets),
    )
