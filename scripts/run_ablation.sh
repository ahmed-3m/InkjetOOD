#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablation.sh
# Run the single-split separation-loss ablation study.
# λ ∈ {0.0, 0.001, 0.01, 0.02, 0.05, 0.1}
#
# Usage:
#   bash scripts/run_ablation.sh
#   GPU=3 bash scripts/run_ablation.sh
#
# Before running:
#   export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
#
# Note: This is the SINGLE-SPLIT ablation (not CV). Results:
#   λ=0.0  → AUROC 0.8325
#   λ=0.01 → AUROC 0.8603  ← best single-split
#   λ=0.02 → AUROC 0.8541
#   λ=0.05 → AUROC 0.8553
# =============================================================================
set -e

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Single-Split Separation Loss Ablation"
echo "  λ values: 0.0  0.001  0.01  0.02  0.05  0.1"
echo "  GPU: $GPU"
echo "============================================================"

python run_ablation.py \
    --weights    0.0 0.001 0.01 0.02 0.05 0.1 \
    --epochs     100 \
    --batch_size 128 \
    --num_trials 100 \
    --out_dir    results/ablation

echo ""
echo "Ablation complete. Summary: results/ablation/ablation_results.json"
