#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablation.sh
# Run the separation-loss ablation study (λ ∈ {0.0, 0.001, 0.01, 0.05, 0.1}).
# =============================================================================
set -e

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Separation Loss Ablation Study"
echo "  λ values: 0.0  0.001  0.01  0.05  0.1"
echo "============================================================"

python run_ablation.py \
    --weights     0.0 0.001 0.01 0.05 0.1 \
    --epochs      100 \
    --batch_size  128 \
    --num_trials  50 \
    --out_dir     results/ablation

echo "Ablation study complete."
