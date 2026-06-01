# Pan-Cancer Multi-Modal HDAC Inhibitor Repurposing

This repository contains the source code, computational workflow, and supplementary material associated with the manuscript:

**Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal**

The study implements a hypothesis-generating computational drug-repurposing workflow that integrates molecular representation learning, LINCS L1000 perturbational transcriptomics, TCGA pan-cancer disease signatures, weighted transcriptomic reversal scoring, network pharmacology, public dependency/sensitivity resources, and structural docking support.

## Scope and evidence statement

This repository supports a **computational prioritization** study. The analyses nominate Class I HDAC inhibitors, including TC-H-106, RG2833, and Tianeptinaline, as candidates for future experimental testing. The repository does **not** provide wet-lab validation and should not be interpreted as proving therapeutic efficacy.

Transcriptomic reversal, network enrichment, survival stratification, DepMap dependency, GDSC surrogate comparison, docking, and conformer-energy analysis are supportive computational evidence layers rather than direct pharmacological validation.

## Current implementation

The reported model uses two complementary molecular representations:

1. **Graph-derived atom-token representation** parsed from SMILES using RDKit.
2. **Morgan fingerprint representation** using ECFP4 fingerprints with radius 2 and 1024 bits.

In the current implementation, molecules are encoded using an atom-token Transformer encoder and a fingerprint MLP branch, followed by a regularized nonlinear fusion head to predict LINCS Level 5 transcriptomic perturbation vectors. Explicit edge-conditioned graph message passing is not claimed as the basis of the reported results.

## Repository contents

| File | Purpose |
|---|---|
| `dataset.py` | PyTorch dataset for metadata, SMILES parsing, fingerprint generation, and LINCS expression retrieval. |
| `drug_to_graph.py` | RDKit-based conversion of SMILES strings into atom-feature and adjacency representations. |
| `multimodal_model.py` | Dual-stream molecular representation model. |
| `strict_data_split.py` | Drug-cell line pair split used to reduce duplicate-pair leakage. |
| `train.py` | Main training script for the multi-modal model. |
| `run_baselines.py` | Reimplemented baseline models for comparison. |
| `run_ablation.py` | Ablation experiments for molecular representation components. |
| `calc_mmff94_energy.py` | RDKit MMFF94 conformer energy calculation for TC-H-106. |
| `Supplementary_Material.pdf` | Supplementary analyses and extended pan-cancer contextualization figures. |

## Data requirements

Large public source datasets are not redistributed in this repository. Please download them from their original sources:

- **TCGA RNA-seq and clinical metadata:** Genomic Data Commons (GDC) portal.
- **LINCS L1000 Level 5 perturbation profiles:** GEO accession `GSE92742`.
- **LINCS signature metrics:** Broad LINCS signature metadata and quality metrics.
- **HDAC1 protein structure:** RCSB PDB ID `4BKX`.
- **TC-H-106 ligand conformer:** PubChem CID `16070100`.
- **DepMap and GDSC data:** Official DepMap and GDSC public download portals.

Expected local data paths used by the scripts:

```text
data/
  clean_dataset.csv
  GSE92742_Broad_LINCS_sig_metrics.txt
  level5_beta_trt_cp_n720216x12328.gctx
```

`clean_dataset.csv` should contain at least the following columns:

```text
sig_id, smiles, drug_id
```

The `sig_id` field must match the column identifiers in the LINCS Level 5 `.gctx` file.

## Environment

The code was developed in Python with PyTorch and RDKit. A minimal environment should include:

```bash
pip install torch numpy pandas scipy scikit-learn tqdm cmapPy
```

RDKit installation is often more stable through conda:

```bash
conda install -c conda-forge rdkit
pip install torch numpy pandas scipy scikit-learn tqdm cmapPy
```

A minimal dependency file is provided as `requirements.txt`.

## Reproducing the main computational workflow

### 1. Prepare the data

Place the LINCS metadata, LINCS Level 5 `.gctx` file, and signature metrics under `data/` using the filenames listed above. Confirm that `sig_id` values in `clean_dataset.csv` match the `.gctx` column identifiers.

### 2. Generate the drug-cell line pair split

```bash
python strict_data_split.py
```

This creates:

```text
data/train_dataset_drugcell.csv
data/test_dataset_drugcell.csv
```

This split prevents exact drug-cell line perturbation pairs from appearing in both training and testing subsets. It is not a strict zero-shot drug, zero-shot cell-line, or scaffold split.

### 3. Train the multi-modal model

```bash
python train.py
```

Default training settings:

- batch size: 128
- training epochs: 60
- optimizer: AdamW
- learning rate: 5e-4
- output dimension: 12,328 genes

Expected output:

```text
multimodal_model_epoch_*.pth
```

### 4. Run baseline comparisons

```bash
python run_baselines.py
```

This script trains and evaluates reimplemented single-modality baselines under the same drug-cell line pair split.

### 5. Run ablation experiments

```bash
python run_ablation.py
```

This script evaluates the contribution of molecular representation components.

### 6. Compute MMFF94 conformer support

```bash
python calc_mmff94_energy.py
```

This script computes an MMFF94 minimized conformer energy for TC-H-106 as structural support. It does not prove biological activity.

## Evaluation metrics

The manuscript reports:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- profile-wise Pearson correlation
- profile-wise Spearman correlation
- weighted transcriptomic reversal score (wTRS)

Correlation metrics are interpreted as directional transcriptomic concordance for downstream compound prioritization, not as direct evidence of drug efficacy.

## Reproducibility notes and limitations

- The current train/test split is a drug-cell line pair split.
- The manuscript explicitly acknowledges that stronger future benchmarks should include leave-drug-out, leave-cell-line-out, scaffold-based, and external pharmacogenomic validation settings.
- Large `.gctx`, checkpoint, and intermediate prediction files are intentionally not stored in the repository.
- Users should verify local paths before running the scripts.
- The GDSC analysis uses Entinostat as a pharmacological surrogate because TC-H-106 is not available in GDSC. This result is reported as contextual and non-significant, not as positive validation.
- Molecular docking and MMFF94 calculations provide structural plausibility, not evidence of cellular efficacy.

## Supplementary material

The supplementary material contains extended pan-cancer network analyses and additional supporting figures. For the journal submission, use the conservative version aligned with the manuscript title and hypothesis-generating framing:

```text
Supplementary_Material.pdf
```

If you keep multiple drafts locally, make sure the submitted GitHub-facing version corresponds to `Supplementary_Material_csbj_safe_v4_flow_fixed.pdf` or the latest caption/flow-fixed version.

## Citation

If you use this code or processed outputs, please cite the associated preprint/manuscript:

```bibtex
@article{tong2026multimodal,
  title={Multi-Modal Molecular Representation Learning Prioritizes Class I HDAC Inhibitors for Pan-Cancer Transcriptomic Reversal},
  author={Tong, Siyuan and Zhang, Wen and Ji, Shiliang},
  year={2026},
  doi={10.64898/2026.04.22.720196}
}
```

## License

This repository is intended for non-commercial academic research use. Please check the licenses and terms of use of the original public data sources before redistributing derived files.
