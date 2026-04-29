#!/bin/bash
# =============================================================================
#  Inkjet CDM — Separation Loss Ablation (5-Fold CV)
# =============================================================================
#
#  Runs 4 λ values with 5-fold stratified cross-validation.
#  All runs use identical settings except λ, for a clean ablation.
#
#  Expected output (per run):
#    results/cv_lambda<X>/cv_summary.json   — aggregated mean ± std
#    results/cv_lambda<X>/fold<N>/          — per-fold scores + metrics
#
#  Estimated GPU time: ~24h total (~6h per λ value)
#
#  Usage:
#    chmod +x run_inkjet_ablation.sh
#    CUDA_VISIBLE_DEVICES=<GPU_ID> ./run_inkjet_ablation.sh
#
#    Or with nohup for long-running jobs:
#    CUDA_VISIBLE_DEVICES=<GPU_ID> nohup ./run_inkjet_ablation.sh > ablation.log 2>&1 &
#
# =============================================================================

set -e  # Exit on error

# ── Configuration (same across all runs) ─────────────────────────────────────
EPOCHS=100
BATCH_SIZE=64          # Must be 64 for sep loss runs (3 fwd passes → OOM at 128)
SEED=42                # Same primary seed as CIFAR-10 experiments
N_FOLDS=5
NUM_TRIALS=100         # K=100 MC trials for stable evaluation (matching single-seed runs)
BASE_DIR="results"

# ── Activate conda environment ───────────────────────────────────────────────
# Option A: conda activate by name
#   conda activate sdm
# Option B: conda activate by path (uncomment and edit):
#   conda activate /path/to/your/env
# Option C: just ensure the right python is on PATH
echo "Using python: $(which python 2>/dev/null || where python 2>/dev/null)"

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Start time: $(date)"
echo ""

# ── λ values to sweep ───────────────────────────────────────────────────────
LAMBDAS=(0.0 0.01 0.02 0.05)

for LAMBDA in "${LAMBDAS[@]}"; do
    OUT_DIR="${BASE_DIR}/cv_lambda${LAMBDA}"

    echo "============================================================="
    echo "  λ = ${LAMBDA}  |  Output: ${OUT_DIR}"
    echo "  Started: $(date)"
    echo "============================================================="

    # Skip if cv_summary.json already exists (resume support)
    if [ -f "${OUT_DIR}/cv_summary.json" ]; then
        echo "  [SKIP] ${OUT_DIR}/cv_summary.json already exists. Delete to re-run."
        echo ""
        continue
    fi

    python run_cv.py \
        --sep_loss_weight "${LAMBDA}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --num_trials "${NUM_TRIALS}" \
        --n_folds "${N_FOLDS}" \
        --out_dir "${OUT_DIR}" \
        2>&1 | tee "${BASE_DIR}/cv_lambda${LAMBDA}.log"

    # Clean up checkpoint files to save disk space (~800MB per run)
    # We only need the scores CSVs and metrics JSONs, not the model weights
    if [ -f "${OUT_DIR}/cv_summary.json" ]; then
        find "${OUT_DIR}" -name "fold*_best.pt" -delete
        echo "  [CLEANUP] Deleted checkpoint files to save disk space"
    fi

    echo ""
    echo "  Finished λ=${LAMBDA} at $(date)"
    echo ""
done

# ── Print summary of all results ─────────────────────────────────────────────
echo ""
echo "============================================================="
echo "  ALL ABLATION RUNS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================="
echo ""
echo "Results summary:"
for LAMBDA in "${LAMBDAS[@]}"; do
    SUMMARY="${BASE_DIR}/cv_lambda${LAMBDA}/cv_summary.json"
    if [ -f "${SUMMARY}" ]; then
        AUROC=$(python -c "import json; d=json.load(open('${SUMMARY}')); print(f\"AUROC={d['mean_auroc']:.4f}±{d['std_auroc']:.4f}\")")
        FPR=$(python -c "import json; d=json.load(open('${SUMMARY}')); f=d.get('mean_fpr95'); print(f'FPR@95={f:.4f}±{d[\"std_fpr95\"]:.4f}' if f else 'FPR@95=N/A')")
        echo "  λ=${LAMBDA}:  ${AUROC}  ${FPR}"
    else
        echo "  λ=${LAMBDA}:  NOT FOUND"
    fi
done
echo ""
echo "Full results in: $(pwd)/${BASE_DIR}/cv_lambda*/"
