# Pan-Cancer Multi-Modal HDAC Inhibitor Repurposing
This repository contains the official PyTorch implementation and the Supplementary Material for our paper:
**"Multi-Modal Deep Learning Integrates Spatial Topologies and Sequential Motifs to Identify Class I HDAC Inhibitors as Pan-Cancer Therapeutics"**

## 📄 Supplementary Material
The complete supplementary appendices, including extended pan-cancer network analyses (Figures S1-S5) and evaluations against metastatic hallmarks (Tumor Mutational Burden & Treatment-Enriched Drivers), can be found in `Supplementary_Material.pdf`.

## ⚙️ Core Modules
* `dataset.py`: Drug-cell line interaction data processing.
* `multimodal_model.py`: The GNN + self-attention Transformer dual-stream architecture.
* `strict_data_split.py`: Implementation of the Cold-Start Scaffold-Split for rigorous evaluation.
* `train.py`: Main training loop.
* `run_ablation.py` & `run_baselines.py`: Scripts for ablation studies and SOTA baseline comparisons.
* `calc_mmff94_energy.py`: Thermodynamic conformational stability calculation (RDKit).