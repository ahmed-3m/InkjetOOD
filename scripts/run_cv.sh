#!/usr/bin/env bash
# =============================================================================
# scripts/run_cv.sh
# Run a single 5-fold cross-validation experiment.
#
# Usage:
#   bash scripts/run_cv.sh                     # λ=0 baseline (thesis main result)
#   LAM=0.01 bash scripts/run_cv.sh            # with separation loss
#   GPU=3 bash scripts/run_cv.sh               # override GPU
#
# Before running:
#   export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
#
# Thesis main result (λ=0, 5-fold CV):
#   AUROC = 0.8673 ± 0.023   FPR@95 = 0.563
# =============================================================================
set -e

GPU="${GPU:-0}"
LAM="${LAM:-0.0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  5-Fold Cross-Validation — CDM"
echo "  GPU: $GPU   λ: $LAM   Epochs: 100   K: 50"
echo "============================================================"

# Use batch=64 when sep loss is active to avoid OOM with 3 forward passes
if python -c "import sys; sys.exit(0 if float('$LAM') > 0 else 1)" 2>/dev/null; then
    BATCH=64
else
    BATCH=128
fi

python run_cv.py \
    --epochs          100 \
    --batch_size      "$BATCH" \
    --lr              2e-4 \
    --sep_loss_weight "$LAM" \
    --img_size        128 \
    --n_folds         5 \
    --num_trials      50 \
    --seed            42 \
    --out_dir         "results/cv_lambda${LAM}"

echo ""
echo "5-Fold CV complete. Results in: results/cv_lambda${LAM}/"
echo "Summary: results/cv_lambda${LAM}/cv_summary.json"
