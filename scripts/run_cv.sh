#!/usr/bin/env bash
# =============================================================================
# scripts/run_cv.sh
# Run the full 5-fold cross-validation experiment.
# =============================================================================
set -e

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  5-Fold Cross-Validation — CDM with Separation Loss"
echo "============================================================"

python run_cv.py \
    --epochs          100 \
    --batch_size      128 \
    --lr              2e-4 \
    --sep_loss_weight 0.01 \
    --img_size        128 \
    --n_folds         5 \
    --num_trials      50 \
    --out_dir         results/cv

echo "Cross-validation complete."
