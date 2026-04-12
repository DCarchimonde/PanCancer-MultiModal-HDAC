import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from cmapPy.pandasGEXpress.parse import parse
from drug_to_graph import smiles_to_graph


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal drug perturbation data.

    This class integrates 2D topological molecular graphs and 1D functional
    fingerprints, mapped to systemic transcriptomic signatures (LINCS L1000).
    """

    def __init__(self, csv_file, gctx_file, metrics_file=None, debug_mode=False):
        """
        Args:
            csv_file (str): Path to the primary metadata CSV file.
            gctx_file (str): Path to the LINCS L1000 Level 5 .gctx file.
            metrics_file (str, optional): Path to sig_metrics.txt for QC filtering.
            debug_mode (bool): If True, loads a small subset for debugging purposes.
        """
        self.gctx_file = gctx_file

        print(f"Loading metadata from: {csv_file}")
        self.meta_data = pd.read_csv(csv_file)

        # ==========================================
        # Quality Control (QC) and Signature Filtering
        # ==========================================
        if metrics_file:
            print(f"Applying quality control filters using: {metrics_file} ...")
            metrics_df = pd.read_csv(metrics_file, sep='\t', usecols=['sig_id', 'tas', 'distil_cc_q75'])

            original_len = len(self.meta_data)
            self.meta_data = pd.merge(self.meta_data, metrics_df, on='sig_id', how='inner')

            # Retain high-fidelity signatures based on Transcriptional Activity Score (TAS >= 0.2)
            self.meta_data = self.meta_data[self.meta_data['tas'] >= 0.2]
            print(f"QC applied. Samples retained: {len(self.meta_data)} (Original: {original_len})")
        else:
            print("Warning: metrics_file not provided. Bypassing quality control filters.")

        # ==========================================
        # Data Sanitization
        # ==========================================
        if 'smiles' in self.meta_data.columns:
            before_clean = len(self.meta_data)
            # Remove proprietary/restricted structural data
            self.meta_data = self.meta_data[self.meta_data['smiles'] != 'restricted']
            if before_clean != len(self.meta_data):
                print(f"Restricted structures removed. Valid samples: {len(self.meta_data)}")

        # --- Debug Mode ---
        if debug_mode:
            print("DEBUG MODE active: Reducing dataset to 200 samples.")
            self.meta_data = self.meta_data.head(200)

        # Cache lists for faster __getitem__ retrieval
        self.all_sig_ids = self.meta_data['sig_id'].tolist()
        self.all_smiles = self.meta_data['smiles'].tolist()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        smiles = self.all_smiles[idx]
        sig_id = self.all_sig_ids[idx]

        # 1. Modality 1: Topological Graph Extraction
        features, adj = smiles_to_graph(smiles)
        if features is None or adj is None:
            return None

        # 2. Modality 2: Functional Substructure Motif Extraction (Morgan Fingerprint)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 1024-bit ECFP4 fingerprint
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp_array = morgan_gen.GetFingerprintAsNumPy(mol)

        # 3. Ground Truth: Gene Expression Profile Extraction
        try:
            data_gctx = parse(self.gctx_file, cid=[sig_id])
            gene_expression = data_gctx.data_df.values.flatten()
        except Exception:
            # Handle rare index mismatches or read errors gracefully
            return None

        # 4. Return Multi-Modal Data Tuple
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(fp_array, dtype=torch.float32),
            torch.tensor(gene_expression, dtype=torch.float32)
        )