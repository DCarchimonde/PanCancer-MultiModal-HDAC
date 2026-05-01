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

## 📝 Citation

If you find our work, data, or code useful for your research, please cite our bioRxiv preprint:

> **Multi-Modal Deep Learning Integrates Spatial Topologies and Sequential Motifs to Identify Class I HDAC Inhibitors as Pan-Cancer Therapeutics**
> Siyuan Tong, Wen Zhang, and Shiliang Ji.
> *bioRxiv* (2026). DOI: [10.64898/2026.04.22.720196](https://doi.org/10.64898/2026.04.22.720196)
```bibtex
@article{tong2026multimodal,
  title={Multi-Modal Deep Learning Integrates Spatial Topologies and Sequential Motifs to Identify Class I HDAC Inhibitors as Pan-Cancer Therapeutics},
  author={Tong, Siyuan and Zhang, Wen and Ji, Shiliang},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.64898/2026.04.22.720196}
}
