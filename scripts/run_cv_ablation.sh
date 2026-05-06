#!/usr/bin/env bash
# =============================================================================
# scripts/run_cv_ablation.sh
# Run 5-fold CV for all λ values to reproduce the full CV ablation table.
#
# Usage:
#   bash scripts/run_cv_ablation.sh
#   GPU=3 bash scripts/run_cv_ablation.sh
#
# Before running:
#   export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
#
# Expected results (from thesis):
#   λ=0.0   → AUROC 0.8673 ± 0.023  ← THESIS MAIN RESULT
#   λ=0.01  → AUROC 0.8628 ± 0.029
#   λ=0.02  → AUROC 0.8510 ± 0.033
#   λ=0.05  → AUROC 0.8670 ± 0.026
#
# Total estimated runtime: ~15 hours on Quadro GV100 (32 GB).
# =============================================================================
set -e

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Full CV Ablation — λ ∈ {0.0, 0.01, 0.02, 0.05}"
echo "  GPU: $GPU"
echo "============================================================"

for LAM in 0.0 0.01 0.02 0.05; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Running λ = $LAM"
    echo "------------------------------------------------------------"

    if python -c "import sys; sys.exit(0 if float('$LAM') > 0 else 1)" 2>/dev/null; then
        BATCH=64    # separation loss: 3 forward passes
    else
        BATCH=128   # baseline: 1 forward pass
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

    echo "  → Done: results/cv_lambda${LAM}/cv_summary.json"
done

echo ""
echo "============================================================"
echo "  All λ values complete."
echo "  Plot results:"
echo "    python scripts/plot_inkjet_results_cv.py --out_dir results/figures_cv"
echo "============================================================"
