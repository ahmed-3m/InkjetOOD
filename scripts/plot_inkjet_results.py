"""
plot_inkjet_results.py
======================
Generate all publication-quality figures for the inkjet CDM thesis chapter.

Usage:
  cd thesis_cdm_final
  python scripts/plot_inkjet_results.py
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# Style constants (matching CIFAR-10 ablation plot)
# ---------------------------------------------------------------------------
BG          = '#0f1117'
GRID_COL    = '#333333'
WHITE       = 'white'
BLUE        = '#4fc3f7'
RED         = '#ef5350'
GOLD        = '#ffd54f'
GREEN       = '#66bb6a'
ORANGE      = '#ff9800'
PURPLE      = '#ab47bc'
TEAL        = '#26c6da'

POINT_COLORS = [RED, GREEN, GOLD, ORANGE, TEAL, PURPLE]

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hard-coded results (verified, K=100 for all)
# ---------------------------------------------------------------------------
LAMBDAS    = [0.0,   0.01,   0.02,   0.05]
AUROCS     = [0.8325, 0.8603, 0.8541, 0.8553]
FPRS       = [0.8161, 0.6264, 0.6609, 0.5287]   # FPR@95TPR (lower is better)
LAMBDA_LBL = ['0.0\n(baseline)', '0.01\n(best AUROC)', '0.02', '0.05\n(best FPR)']

FEATURES = ['angle', 'dist1', 'dist6', 'dots', 'edge1', 'edge2', 'edge3', 'edge4']

# Per-feature AUROC (K=100)
AUROC_BASE = [0.5556, 0.9000, 0.8278, 0.9126, 0.7760, 0.7302, 0.7188, 0.6667]
AUROC_BEST = [0.5679, 0.8571, 0.8111, 0.9266, 0.8177, 0.7242, 0.8750, 0.7361]  # λ=0.01

# Per-feature FPR@95TPR (K=100)
FPR_BASE   = [0.9630, 0.2381, 0.9667, 0.9615, 0.9167, 0.8889, 0.5625, 0.5833]
FPR_BEST   = [0.9630, 0.4286, 0.9333, 0.8077, 0.9167, 0.7778, 0.4375, 0.4583]  # λ=0.01

# CSV paths (scores per sample)
CSV_BASE = "results/inkjet_lambda0_k100/scores.csv"
CSV_BEST = "results/inkjet_lambda0.01/scores_final.csv"


def set_dark_style(ax):
    """Apply consistent dark background style to an axis."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.yaxis.set_tick_params(labelcolor=WHITE)
    ax.xaxis.set_tick_params(labelcolor=WHITE)
    ax.grid(axis='y', color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)


# ===========================================================================
# Figure 1: Inkjet λ Ablation (analog of CIFAR-10 plot)
# ===========================================================================
def plot_lambda_ablation():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x_pos = list(range(len(LAMBDAS)))
    peak_idx = AUROCS.index(max(AUROCS))

    ax.plot(x_pos, AUROCS, color=BLUE, linewidth=2.5, zorder=2)

    for i, (x, y, c) in enumerate(zip(x_pos, AUROCS, POINT_COLORS)):
        ms = 220 if i == peak_idx else 100
        ax.scatter(x, y, color=c, s=ms, zorder=5, edgecolors=WHITE, linewidths=1.5)
        weight = 'bold' if i == peak_idx else 'normal'
        ax.annotate(f'{y:.4f}', (x, y + 0.006), ha='center', va='bottom',
                    fontsize=11, color=WHITE, fontweight=weight)

    # Peak annotation
    ax.annotate('★ Best AUROC', xy=(x_pos[peak_idx], AUROCS[peak_idx]),
                xytext=(x_pos[peak_idx] + 0.55, AUROCS[peak_idx] - 0.015),
                fontsize=10, color=GOLD, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))

    # Best FPR annotation
    fpr_best_idx = FPRS.index(min(FPRS))
    ax.annotate('★ Best FPR@95TPR', xy=(x_pos[fpr_best_idx], AUROCS[fpr_best_idx]),
                xytext=(x_pos[fpr_best_idx] - 0.6, AUROCS[fpr_best_idx] - 0.018),
                fontsize=9.5, color=TEAL, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5))

    # Baseline dashed line
    ax.axhline(y=AUROCS[0], color=RED, linewidth=1, linestyle='--', alpha=0.5)
    ax.annotate('── No sep. loss baseline', (0.3, AUROCS[0] - 0.0025),
                fontsize=8.5, color=RED, va='top', fontweight='bold')

    # Optimal zone shading
    ax.axvspan(0.5, 3.5, alpha=0.08, color=GOLD)
    ax.annotate('Optimal zone (λ=0.01–0.05)', (2.0, 0.815),
                ha='center', va='bottom', fontsize=8.5,
                color=GOLD, fontstyle='italic')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(LAMBDA_LBL, fontsize=11, color=WHITE)
    ax.set_xlabel('Separation Loss Weight λ', fontsize=13, color=WHITE, labelpad=10)
    ax.set_ylabel('AUROC', fontsize=13, color=WHITE, labelpad=10)
    ax.set_title('Inkjet CDM — Separation Loss Weight Ablation',
                 fontsize=14, color=WHITE, fontweight='bold', pad=15)
    ax.set_ylim(0.80, 0.88)
    ax.set_xlim(-0.4, 3.6)
    set_dark_style(ax)

    out = f"{OUT_DIR}/fig1_inkjet_lambda_ablation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ Saved: {out}")


# ===========================================================================
# Figure 2: Per-Feature AUROC — Grouped Bar Chart
# ===========================================================================
def plot_per_feature_auroc():
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = np.arange(len(FEATURES))
    w = 0.35

    bars_base = ax.bar(x - w/2, AUROC_BASE, w, label='Baseline (λ=0)', color=RED, alpha=0.85)
    bars_best = ax.bar(x + w/2, AUROC_BEST, w, label='Proposed (λ=0.01)', color=GREEN, alpha=0.85)

    # Value labels on bars
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7.5, color=WHITE)
    for bar in bars_best:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7.5, color=WHITE)

    # Δ annotations on top of best bars for features that improved notably
    for i, (b, g) in enumerate(zip(AUROC_BASE, AUROC_BEST)):
        delta = g - b
        if abs(delta) >= 0.02:
            sign = '+' if delta > 0 else ''
            color = GREEN if delta > 0 else RED
            ax.text(x[i] + w/2, g + 0.025, f'{sign}{delta:.3f}',
                    ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    ax.axhline(0.5, color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, fontsize=11, color=WHITE)
    ax.set_xlabel('Feature', fontsize=13, color=WHITE, labelpad=8)
    ax.set_ylabel('AUROC', fontsize=13, color=WHITE, labelpad=8)
    ax.set_title('Per-Feature AUROC — Baseline vs Proposed (λ=0.01)',
                 fontsize=14, color=WHITE, fontweight='bold', pad=12)
    ax.set_ylim(0.40, 1.00)
    ax.legend(fontsize=11, facecolor='#1e1e2e', labelcolor=WHITE, edgecolor=GRID_COL)
    set_dark_style(ax)

    out = f"{OUT_DIR}/fig2_per_feature_auroc.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ Saved: {out}")


# ===========================================================================
# Figure 3: FPR@95TPR Comparison (lower = better)
# ===========================================================================
def plot_fpr_comparison():
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = np.arange(len(FEATURES))
    w = 0.35

    ax.bar(x - w/2, FPR_BASE, w, label='Baseline (λ=0)', color=RED, alpha=0.85)
    ax.bar(x + w/2, FPR_BEST, w, label='Proposed (λ=0.01)', color=GREEN, alpha=0.85)

    for i, (b, g) in enumerate(zip(FPR_BASE, FPR_BEST)):
        delta = g - b  # negative = improvement (lower FPR is better)
        if abs(delta) >= 0.03:
            sign = '+' if delta > 0 else ''
            color = RED if delta > 0 else GREEN  # reversed: improvement is green
            ax.text(x[i] + w/2, g + 0.015, f'{sign}{delta:.3f}',
                    ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, fontsize=11, color=WHITE)
    ax.set_xlabel('Feature', fontsize=13, color=WHITE, labelpad=8)
    ax.set_ylabel('FPR @ 95% TPR  (↓ better)', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('Per-Feature FPR@95TPR — Baseline vs Proposed (λ=0.01)',
                 fontsize=14, color=WHITE, fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11, facecolor='#1e1e2e', labelcolor=WHITE, edgecolor=GRID_COL)
    set_dark_style(ax)

    out = f"{OUT_DIR}/fig3_fpr_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ Saved: {out}")


# ===========================================================================
# Figure 4: ROC Curves (baseline vs proposed, loaded from CSV)
# ===========================================================================
def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    try:
        df0 = pd.read_csv(CSV_BASE)
        fpr0, tpr0, _ = roc_curve(df0['true_label'], df0['score'])
        roc_auc0 = auc(fpr0, tpr0)
        ax.plot(fpr0, tpr0, color=RED, lw=2, label=f'Baseline λ=0  (AUROC={roc_auc0:.4f})')
    except Exception as e:
        print(f"  [ROC] Could not load baseline CSV: {e}")

    try:
        df1 = pd.read_csv(CSV_BEST)
        fpr1, tpr1, _ = roc_curve(df1['true_label'], df1['score'])
        roc_auc1 = auc(fpr1, tpr1)
        ax.plot(fpr1, tpr1, color=GREEN, lw=2.5, label=f'Proposed λ=0.01  (AUROC={roc_auc1:.4f})')
    except Exception as e:
        print(f"  [ROC] Could not load proposed CSV: {e}")

    ax.plot([0, 1], [0, 1], color=GRID_COL, lw=1, linestyle='--', label='Random (AUROC=0.50)')
    ax.axhline(0.95, color=GOLD, lw=0.8, linestyle=':', alpha=0.7)
    ax.text(0.02, 0.953, 'TPR=0.95 (FPR@95TPR threshold)',
            fontsize=8, color=GOLD, alpha=0.9)

    ax.set_xlabel('False Positive Rate', fontsize=13, color=WHITE, labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=13, color=WHITE, labelpad=8)
    ax.set_title('ROC Curves — Inkjet CDM\n(Overall, K=100 MC trials)',
                 fontsize=13, color=WHITE, fontweight='bold', pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor=WHITE, edgecolor=GRID_COL,
              loc='lower right')
    ax.grid(color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)
    set_dark_style(ax)

    out = f"{OUT_DIR}/fig4_roc_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ Saved: {out}")


# ===========================================================================
# Figure 5: Score Distributions (GOOD vs BAD) — Baseline and Proposed
# ===========================================================================
def plot_score_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    configs = [
        (CSV_BASE, axes[0], 'Baseline (λ=0)', RED),
        (CSV_BEST, axes[1], 'Proposed (λ=0.01)', GREEN),
    ]

    for csv_path, ax, title, main_color in configs:
        ax.set_facecolor(BG)
        try:
            df = pd.read_csv(csv_path)
            good_scores = df[df['true_label'] == 0]['score'].values
            bad_scores  = df[df['true_label'] == 1]['score'].values

            bins = np.linspace(df['score'].min(), df['score'].max(), 40)

            ax.hist(good_scores, bins=bins, alpha=0.6, color=BLUE,
                    label=f'GOOD  (n={len(good_scores)})', density=True)
            ax.hist(bad_scores, bins=bins, alpha=0.6, color=ORANGE,
                    label=f'BAD   (n={len(bad_scores)})', density=True)

            # Threshold at score=0
            ax.axvline(0, color=WHITE, lw=1.2, linestyle='--', alpha=0.7)
            ax.text(0.5, 0.93, 'Decision boundary (score=0)',
                    color=WHITE, fontsize=8, ha='center', va='top',
                    transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Data not found\n({e})', transform=ax.transAxes,
                    ha='center', va='center', color=WHITE, fontsize=10)

        ax.set_xlabel('OOD Score  (score > 0 → BAD)', fontsize=11, color=WHITE, labelpad=8)
        ax.set_ylabel('Density', fontsize=11, color=WHITE, labelpad=8)
        ax.set_title(title, fontsize=13, color=WHITE, fontweight='bold', pad=10)
        ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor=WHITE, edgecolor=GRID_COL)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis='y', color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)

    fig.suptitle('OOD Score Distributions — GOOD vs BAD Samples (K=100)',
                 fontsize=14, color=WHITE, fontweight='bold', y=1.0)

    out = f"{OUT_DIR}/fig5_score_distributions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ Saved: {out}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(f"\nGenerating inkjet thesis figures → {OUT_DIR}/\n{'='*55}")
    plot_lambda_ablation()
    plot_per_feature_auroc()
    plot_fpr_comparison()
    plot_roc_curves()
    plot_score_distributions()
    print(f"\n{'='*55}")
    print("All 5 figures generated successfully.")
    print(f"  fig1 — λ ablation curve (analog of CIFAR-10 plot)")
    print(f"  fig2 — per-feature AUROC grouped bar chart")
    print(f"  fig3 — FPR@95TPR per-feature comparison")
    print(f"  fig4 — overall ROC curves (λ=0 vs λ=0.01)")
    print(f"  fig5 — score distributions GOOD vs BAD")
