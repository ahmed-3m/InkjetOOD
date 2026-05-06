#!/usr/bin/env bash
# =============================================================================
# scripts/run_train.sh
# Train the CDM for a single split.
#
# Usage:
#   bash scripts/run_train.sh              # uses default GPU 0
#   GPU=3 bash scripts/run_train.sh        # override GPU
#
# Before running:
#   export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
#   python download_weights.py             # fetches YOLO weights from HF
# =============================================================================
set -e

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# Go to repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  CDM Training — Inkjet Print Quality Control"
echo "  GPU: $GPU"
echo "  Sep-loss λ: 0.01   Batch: 64   Epochs: 100"
echo "============================================================"

# λ=0.01 (proposed) — batch=64 because sep-loss runs 3 forward passes
python train.py \
    --epochs          100 \
    --batch_size      64 \
    --lr              2e-4 \
    --sep_loss_weight 0.01 \
    --img_size        128 \
    --schedule        cosine \
    --eval_after \
    --out_dir         results/train_proposed

echo ""
echo "Training complete. Results in: results/train_proposed/"
echo "To also train the baseline (λ=0), run:"
echo "  python train.py --sep_loss_weight 0.0 --batch_size 128 --eval_after --out_dir results/train_baseline"
