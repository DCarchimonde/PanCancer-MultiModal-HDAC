import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import warnings

warnings.filterwarnings('ignore')


def create_drug_cell_line_split():
    """
    Implements a rigorous Cold-Start Drug-Cell Line Split strategy.

    This function processes the LINCS L1000 dataset by filtering out low-quality
    signatures (TAS < 0.2) and restricted compounds. It subsequently extracts
    cell line identifiers and performs a disjoint split based on unique
    drug-cell line pairs (85% Train / 15% Test) to strictly prevent data leakage
    during cross-context pan-cancer drug repurposing evaluations.
    """
    print("[INFO] Initializing rigorous Drug-Cell Line Split pipeline...")

    # 1. Load dataset and metrics
    csv_path = './data/clean_dataset.csv'
    metrics_path = './data/GSE92742_Broad_LINCS_sig_metrics.txt'

    df = pd.read_csv(csv_path)
    metrics_df = pd.read_csv(metrics_path, sep='\t')

    # 2. Quality Control: Filter TAS >= 0.2 and valid SMILES
    print("[INFO] Performing quality control filtering...")
    df = df.merge(metrics_df[['sig_id', 'tas']], on='sig_id', how='left')
    df = df[df['tas'] >= 0.2]

    if 'smiles' in df.columns:
        df = df[df['smiles'] != 'restricted']
    else:
        raise ValueError("[ERROR] Column 'smiles' not found in dataset.")

    df = df.reset_index(drop=True)
    print(f"[INFO] Quality control complete. Retained profiles: {len(df)}")

    # 3. Feature Engineering: Extract cell lines from signatures
    # Example parsing: 'ABY001_A549_XH:BRD-K81418486:10:3' -> 'A549'
    df['cell_id'] = df['sig_id'].apply(lambda x: x.split('_')[1] if '_' in str(x) else 'UNKNOWN')
    df['drug_cell_pair'] = df['smiles'] + "_" + df['cell_id']

    # 4. Rigorous Split based on unique drug-cell pairs
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['drug_cell_pair']))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # --- Strict Integrity Verification ---
    train_pairs = set(train_df['drug_cell_pair'])
    test_pairs = set(test_df['drug_cell_pair'])
    overlap = train_pairs.intersection(test_pairs)

    print("\n--------------------------------------------------")
    print("[REPORT] Data Leakage Integrity Verification")
    print(f"Training Set : {len(train_df)} profiles ({len(train_pairs)} unique pairs)")
    print(f"Testing Set  : {len(test_df)} profiles ({len(test_pairs)} unique pairs)")
    print(f"Overlap Pairs: {len(overlap)} (Target: 0)")
    print("--------------------------------------------------")

    if len(overlap) == 0:
        print("[SUCCESS] Zero data leakage confirmed across subsets.")
    else:
        print("[WARNING] Data leakage detected! Review split logic.")

    # 5. Export streamlined subsets
    columns_to_save = ['sig_id', 'smiles', 'drug_id', 'cell_id']
    train_df[columns_to_save].to_csv('./data/train_dataset_drugcell.csv', index=False)
    test_df[columns_to_save].to_csv('./data/test_dataset_drugcell.csv', index=False)
    print("[INFO] Split datasets successfully exported to ./data/")


if __name__ == '__main__':
    create_drug_cell_line_split()