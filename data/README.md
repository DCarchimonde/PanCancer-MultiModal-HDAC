# Data layout for the major revision

Large source datasets are intentionally excluded from GitHub. Keep them in a local or AutoDL data directory and pass that directory to the revision scripts.

Required large files:

```text
siginfo_beta.txt
GSE92742_Broad_LINCS_sig_metrics.txt
level5_beta_trt_cp_n720216x12328.gctx
clean_dataset.csv
```

The original pair-level split and small public annotation tables are versioned in this repository:

```text
data/splits/train_dataset_drugcell.csv
data/splits/test_dataset_drugcell.csv
data/metadata/compoundinfo_beta.txt
data/metadata/repurposing_drugs_20200324.txt
data/metadata/repurposing_samples_20200324.txt
```

Windows example:

```powershell
python scripts/audit_lincs_dataset.py `
  --raw-data-dir "E:\DrugDiscovery\Data"
```

Linux/AutoDL example:

```bash
python scripts/audit_lincs_dataset.py \
  --raw-data-dir /root/autodl-tmp/pancancer_data
```

Generated reports are written to `results/revision/audit/`. Large expression caches, prediction arrays, and checkpoints must remain outside Git or in ignored directories.
