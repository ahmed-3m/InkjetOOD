#!/usr/bin/env bash
# =============================================================================
# scripts/run_train.sh
# Quick-start training script.
# =============================================================================
set -e

# Choose a free GPU — edit CUDA_VISIBLE_DEVICES as needed
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  CDM Training — Inkjet Print Quality Control"
echo "  Batch size: 128  (32 GB VRAM)"
echo "  Sep-loss λ: 0.01"
echo "============================================================"

python train.py \
    --epochs          100 \
    --batch_size      128 \
    --lr              2e-4 \
    --sep_loss_weight 0.01 \
    --img_size        128 \
    --schedule        cosine \
    --out_dir         results/train \
    --eval_after

echo "Training complete."
