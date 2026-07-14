#!/usr/bin/env bash
set -euo pipefail

# Reproducible benchmark grid for the BMC Bioinformatics major revision.
# Usage:
#   bash scripts/run_revision_grid.sh core
#   bash scripts/run_revision_grid.sh extended
#
# core:     five splits x two models x seeds 1,2,3 = 30 runs
# extended: core plus pair-split seeds 4,5 for both models = 34 runs total

MODE="${1:-core}"
if [[ "$MODE" != "core" && "$MODE" != "extended" ]]; then
  echo "Usage: bash scripts/run_revision_grid.sh [core|extended]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

OUTPUT_ROOT="results/revision/benchmarks"
LOG_ROOT="results/revision/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

SPLITS=(
  pair_split
  leave_drug_out
  leave_cell_line_out
  scaffold_split
  hdac_class_holdout
)
MODELS=(dual_stream fingerprint_mlp)
CORE_SEEDS=(1 2 3)

run_one() {
  local split="$1"
  local model="$2"
  local seed="$3"
  local run_dir="$OUTPUT_ROOT/$split/$model/seed_$seed"
  local summary="$run_dir/summary.json"
  local log="$LOG_ROOT/${split}__${model}__seed_${seed}.log"

  if [[ -s "$summary" ]]; then
    echo "[SKIP] Completed: $split / $model / seed $seed"
    return 0
  fi

  echo "[START] $split / $model / seed $seed"
  echo "Log: $log"
  python scripts/train_revision.py \
    --split "$split" \
    --model "$model" \
    --seed "$seed" \
    --epochs 30 \
    --patience 5 \
    --batch-size 128 \
    --hidden-dim 512 \
    --num-workers 4 \
    --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$log"
  echo "[DONE] $split / $model / seed $seed"
}

started_at="$(date --iso-8601=seconds)"
printf '{\n  "mode": "%s",\n  "started_at": "%s"\n}\n' "$MODE" "$started_at" > "$OUTPUT_ROOT/grid_manifest.json"

for split in "${SPLITS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${CORE_SEEDS[@]}"; do
      run_one "$split" "$model" "$seed"
    done
  done
done

if [[ "$MODE" == "extended" ]]; then
  for model in "${MODELS[@]}"; do
    for seed in 4 5; do
      run_one pair_split "$model" "$seed"
    done
  done
fi

finished_at="$(date --iso-8601=seconds)"
printf '{\n  "mode": "%s",\n  "started_at": "%s",\n  "finished_at": "%s"\n}\n' \
  "$MODE" "$started_at" "$finished_at" > "$OUTPUT_ROOT/grid_manifest.json"

echo "All requested benchmark runs completed."
