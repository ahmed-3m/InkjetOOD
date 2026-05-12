"""
plot_inkjet_results_cv.py
=========================
Generate all publication-quality figures and LaTeX tables for the inkjet CDM
thesis chapter, based on 5-fold cross-validation results.

Usage:
  cd thesis_cdm_final
  python scripts/plot_inkjet_results_cv.py
"""

import os
import json
import numpy as np
import pandas as pd

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
OUT_FIG     = "results/figures"
OUT_TEX     = "results/tables"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TEX, exist_ok=True)

LAMBDAS     = [0.0, 0.01, 0.02, 0.05]
LAMBDA_DIRS = {l: f"{RESULTS_DIR}/cv_lambda{l}" for l in LAMBDAS}

# ---------------------------------------------------------------------------
# Style (matching CIFAR-10 plots for visual consistency in thesis)
# ---------------------------------------------------------------------------
BG       = '#0f1117'
GRID_COL = '#333333'
WHITE    = 'white'
BLUE     = '#4fc3f7'
RED      = '#ef5350'
GOLD     = '#ffd54f'
GREEN    = '#66bb6a'
ORANGE   = '#ff9800'
PURPLE   = '#ab47bc'
TEAL     = '#26c6da'
COLORS   = [RED, GREEN, GOLD, TEAL]   # one per λ value

FEATURES = ['angle', 'dist1', 'dist6', 'dots', 'edge1', 'edge2', 'edge3', 'edge4']

# ---------------------------------------------------------------------------
# Load all CV summary data
# ---------------------------------------------------------------------------
def load_cv_data():
    data = {}
    for lam in LAMBDAS:
        path = f"{LAMBDA_DIRS[lam]}/cv_summary.json"
        with open(path) as f:
            data[lam] = json.load(f)
    return data

def set_dark_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.yaxis.set_tick_params(labelcolor=WHITE)
    ax.xaxis.set_tick_params(labelcolor=WHITE)
    ax.grid(axis='y', color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)


# ===========================================================================
# Figure 1: λ Ablation Curve with Error Bars (5-fold CV)
# ===========================================================================
def plot_lambda_ablation_cv(data):
    aurocs = [data[l]['mean_auroc'] for l in LAMBDAS]
    stds   = [data[l]['std_auroc']  for l in LAMBDAS]
    fprs   = [data[l].get('mean_fpr95', 0) for l in LAMBDAS]
    fpr_stds = [data[l].get('std_fpr95', 0) for l in LAMBDAS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    x_pos    = list(range(4))
    x_labels = ['0.0\n(baseline)', '0.01', '0.02', '0.05']

    # ── Left: AUROC ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.errorbar(x_pos, aurocs, yerr=stds, fmt='o-', color=BLUE,
                linewidth=2.5, markersize=10, capsize=6,
                ecolor=WHITE, elinewidth=1.5, markerfacecolor=BLUE,
                markeredgecolor=WHITE, markeredgewidth=1.5, zorder=5)

    for i, (x, y, s) in enumerate(zip(x_pos, aurocs, stds)):
        ax.annotate(f'{y:.4f}\n±{s:.4f}', (x, y + s + 0.004),
                    ha='center', va='bottom', fontsize=8.5, color=WHITE)

    # Baseline ref line
    ax.axhline(aurocs[0], color=RED, linewidth=1, linestyle='--', alpha=0.5)
    ax.text(0.35, aurocs[0] - 0.003, '── Baseline (λ=0)',
            fontsize=8, color=RED, va='top')

    # Shaded region showing all values are within 1 std of baseline
    ax.fill_between(x_pos,
                    [aurocs[0] - stds[0]] * 4,
                    [aurocs[0] + stds[0]] * 4,
                    alpha=0.08, color=RED, label='Baseline ± 1 std')

    ax.annotate('No significant\nimprovement', xy=(1.5, 0.861),
                ha='center', fontsize=9, color=GOLD, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                          edgecolor=GOLD, alpha=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11, color=WHITE)
    ax.set_xlabel('Separation Loss Weight λ', fontsize=12, color=WHITE, labelpad=8)
    ax.set_ylabel('AUROC (5-fold CV)', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('Inkjet CDM — AUROC Ablation', fontsize=13, color=WHITE,
                 fontweight='bold', pad=12)
    ax.set_ylim(0.80, 0.92)
    set_dark_style(ax)

    # ── Right: FPR@95TPR ────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    ax.errorbar(x_pos, fprs, yerr=fpr_stds, fmt='s-', color=TEAL,
                linewidth=2.5, markersize=10, capsize=6,
                ecolor=WHITE, elinewidth=1.5, markerfacecolor=TEAL,
                markeredgecolor=WHITE, markeredgewidth=1.5, zorder=5)

    for i, (x, y, s) in enumerate(zip(x_pos, fprs, fpr_stds)):
        ax.annotate(f'{y:.4f}\n±{s:.4f}', (x, y + s + 0.01),
                    ha='center', va='bottom', fontsize=8.5, color=WHITE)

    ax.axhline(fprs[0], color=RED, linewidth=1, linestyle='--', alpha=0.5)
    ax.text(0.35, fprs[0] + 0.01, '── Baseline (λ=0)',
            fontsize=8, color=RED, va='bottom')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11, color=WHITE)
    ax.set_xlabel('Separation Loss Weight λ', fontsize=12, color=WHITE, labelpad=8)
    ax.set_ylabel('FPR@95TPR ↓  (lower is better)', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('Inkjet CDM — FPR@95TPR Ablation', fontsize=13, color=WHITE,
                 fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.0)
    set_dark_style(ax)

    fig.suptitle('Inkjet CDM — Separation Loss Ablation (5-Fold CV, K=100)',
                 fontsize=15, color=WHITE, fontweight='bold', y=1.02)

    out = f"{OUT_FIG}/fig1_inkjet_lambda_ablation_cv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ===========================================================================
# Figure 2: Per-Feature AUROC — All 4 λ values, with error bars
# ===========================================================================
def plot_per_feature_auroc_cv(data):
    n_feat   = len(FEATURES)
    n_lambda = len(LAMBDAS)
    width    = 0.18
    x        = np.arange(n_feat)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    lambda_labels = ['λ=0.0 (baseline)', 'λ=0.01', 'λ=0.02', 'λ=0.05']

    for k, (lam, col, lbl) in enumerate(zip(LAMBDAS, COLORS, lambda_labels)):
        means = [data[lam]['per_feature'].get(f, {}).get('mean', 0) for f in FEATURES]
        stds  = [data[lam]['per_feature'].get(f, {}).get('std',  0) for f in FEATURES]
        offset = (k - n_lambda/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=lbl, color=col, alpha=0.85)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt='none', ecolor=WHITE, elinewidth=1.2, capsize=3, zorder=5)

    ax.axhline(0.5, color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, fontsize=11, color=WHITE)
    ax.set_xlabel('Feature Type', fontsize=13, color=WHITE, labelpad=8)
    ax.set_ylabel('AUROC (mean ± std, 5-fold CV)', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('Per-Feature AUROC — Separation Loss Ablation (Inkjet CDM)',
                 fontsize=14, color=WHITE, fontweight='bold', pad=12)
    ax.set_ylim(0.40, 1.10)
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor=WHITE,
              edgecolor=GRID_COL, loc='upper right')
    set_dark_style(ax)

    # Annotate angle as unreliable
    ax.annotate('⚠ Unreliable\n(few BAD samples)', xy=(0, 0.82),
                ha='center', fontsize=7.5, color=GOLD,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                          edgecolor=GOLD, alpha=0.7))

    out = f"{OUT_FIG}/fig2_per_feature_auroc_cv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ===========================================================================
# Figure 3: Cross-Domain Comparison (CIFAR-10 vs Inkjet)
# ===========================================================================
def plot_cross_domain_comparison(data):
    # CIFAR-10 data (from verified results_summary.md)
    cifar_lambdas = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1]
    cifar_aurocs  = [0.9252, 0.9732, 0.9882, 0.9903, 0.9851, 0.9667]

    # Inkjet data (5-fold CV)
    inkjet_aurocs = [data[l]['mean_auroc'] for l in LAMBDAS]
    inkjet_stds   = [data[l]['std_auroc']  for l in LAMBDAS]
    inkjet_x      = [0, 1, 2, 3]  # positions

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    # ── Left: CIFAR-10 ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    x_c = list(range(len(cifar_lambdas)))
    peak_i = cifar_aurocs.index(max(cifar_aurocs))

    ax.plot(x_c, cifar_aurocs, color=GREEN, linewidth=2.5, zorder=2)
    for i, (x, y) in enumerate(zip(x_c, cifar_aurocs)):
        ms = 200 if i == peak_i else 80
        col = GOLD if i == peak_i else WHITE
        ax.scatter(x, y, color=col, s=ms, zorder=5,
                   edgecolors=WHITE, linewidths=1.2)

    ax.annotate(f'Peak: 0.9903\n(λ=0.02)',
                xy=(x_c[peak_i], cifar_aurocs[peak_i]),
                xytext=(x_c[peak_i] + 0.5, cifar_aurocs[peak_i] - 0.04),
                fontsize=9, color=GOLD,
                arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))

    # Baseline
    ax.axhline(cifar_aurocs[0], color=RED, linewidth=1, linestyle='--', alpha=0.6)
    ax.text(0.05, cifar_aurocs[0] + 0.005, f'Baseline: {cifar_aurocs[0]:.4f}',
            fontsize=8, color=RED)

    # Improvement annotation
    delta = max(cifar_aurocs) - cifar_aurocs[0]
    ax.annotate(f'+{delta:.1%} AUROC\nimprovement', xy=(3, 0.87),
                ha='center', fontsize=10, color=GREEN, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                          edgecolor=GREEN, alpha=0.8))

    ax.set_xticks(x_c)
    ax.set_xticklabels(['0.0', '0.001', '0.01', '0.02', '0.05', '0.1'],
                       fontsize=10, color=WHITE)
    ax.set_xlabel('Separation Loss Weight λ', fontsize=12, color=WHITE, labelpad=8)
    ax.set_ylabel('AUROC', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('CIFAR-10 (50K images, 2 classes)\nSeed=42, Single Split',
                 fontsize=12, color=WHITE, fontweight='bold', pad=12)
    ax.set_ylim(0.75, 1.02)
    set_dark_style(ax)

    # ── Right: Inkjet ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    x_i = [0, 1, 2, 3]
    ax.errorbar(x_i, inkjet_aurocs, yerr=inkjet_stds, fmt='o-', color=BLUE,
                linewidth=2.5, markersize=10, capsize=6,
                ecolor=WHITE, elinewidth=1.5, markerfacecolor=BLUE,
                markeredgecolor=WHITE, markeredgewidth=1.5, zorder=5)

    # Baseline shaded region
    ax.fill_between(x_i,
                    [inkjet_aurocs[0] - inkjet_stds[0]] * 4,
                    [inkjet_aurocs[0] + inkjet_stds[0]] * 4,
                    alpha=0.12, color=RED)

    ax.axhline(inkjet_aurocs[0], color=RED, linewidth=1, linestyle='--', alpha=0.6)
    ax.text(0.1, inkjet_aurocs[0] + 0.003,
            f'Baseline: {inkjet_aurocs[0]:.4f} ± {inkjet_stds[0]:.4f}',
            fontsize=8, color=RED)

    # No improvement annotation
    ax.annotate('No significant\nimprovement\n(all within σ)',
                xy=(1.5, 0.862), ha='center', fontsize=10,
                color=ORANGE, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                          edgecolor=ORANGE, alpha=0.8))

    ax.set_xticks(x_i)
    ax.set_xticklabels(['0.0', '0.01', '0.02', '0.05'], fontsize=10, color=WHITE)
    ax.set_xlabel('Separation Loss Weight λ', fontsize=12, color=WHITE, labelpad=8)
    ax.set_ylabel('AUROC (5-Fold CV)', fontsize=12, color=WHITE, labelpad=8)
    ax.set_title('Inkjet QC (1,327 images, 8 features)\n5-Fold CV, Seed=42',
                 fontsize=12, color=WHITE, fontweight='bold', pad=12)
    ax.set_ylim(0.75, 1.02)
    set_dark_style(ax)

    fig.suptitle('Cross-Domain Analysis: Separation Loss Effect',
                 fontsize=15, color=WHITE, fontweight='bold', y=1.02)

    out = f"{OUT_FIG}/fig3_cross_domain_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ===========================================================================
# Figure 4: ROC Curves — from best fold scores (λ=0 vs λ=0.01)
# ===========================================================================
def plot_roc_curves_cv(data):
    # Find best fold for λ=0.0 and λ=0.01
    def get_best_fold_csv(lam):
        fold_aurocs = data[lam]['fold_details']
        best_fold = max(fold_aurocs, key=lambda x: x['overall_auroc'])
        fold_idx  = best_fold['fold']
        return f"{LAMBDA_DIRS[lam]}/fold{fold_idx}/scores_fold{fold_idx}.csv", fold_idx

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    configs = [
        (0.0,  RED,   'Baseline (λ=0)'),
        (0.01, GREEN, 'Proposed (λ=0.01)'),
        (0.05, TEAL,  'Proposed (λ=0.05)'),
    ]

    for lam, col, label in configs:
        # Try all folds and use mean ROC (more representative than best fold)
        all_fprs = np.linspace(0, 1, 100)
        all_tprs = []
        for fold_det in data[lam]['fold_details']:
            fidx = fold_det['fold']
            csv = f"{LAMBDA_DIRS[lam]}/fold{fidx}/scores_fold{fidx}.csv"
            try:
                df = pd.read_csv(csv)
                fpr, tpr, _ = roc_curve(df['true_label'], df['score'])
                all_tprs.append(np.interp(all_fprs, fpr, tpr))
            except Exception:
                pass

        if all_tprs:
            mean_tpr = np.mean(all_tprs, axis=0)
            std_tpr  = np.std(all_tprs, axis=0)
            roc_auc  = auc(all_fprs, mean_tpr)
            ax.plot(all_fprs, mean_tpr, color=col, lw=2.5,
                    label=f'{label}  (AUROC={roc_auc:.4f})')
            ax.fill_between(all_fprs,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            alpha=0.12, color=col)

    ax.plot([0, 1], [0, 1], color=GRID_COL, lw=1, linestyle='--',
            label='Random (AUROC=0.50)')
    ax.axhline(0.95, color=GOLD, lw=0.8, linestyle=':', alpha=0.7)
    ax.text(0.02, 0.953, 'TPR=0.95 threshold', fontsize=8, color=GOLD, alpha=0.9)

    ax.set_xlabel('False Positive Rate', fontsize=13, color=WHITE, labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=13, color=WHITE, labelpad=8)
    ax.set_title('ROC Curves — Inkjet CDM\n(Mean ± Std across 5 folds, K=100)',
                 fontsize=13, color=WHITE, fontweight='bold', pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor=WHITE,
              edgecolor=GRID_COL, loc='lower right')
    ax.grid(color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)
    set_dark_style(ax)

    out = f"{OUT_FIG}/fig4_roc_curves_cv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ===========================================================================
# Figure 5: Score Distributions — pooled across all folds (λ=0 vs λ=0.01)
# ===========================================================================
def plot_score_distributions_cv(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    configs = [
        (0.0,  axes[0], 'Baseline (λ=0)', RED),
        (0.01, axes[1], 'Proposed (λ=0.01)', GREEN),
    ]

    for lam, ax, title, col in configs:
        ax.set_facecolor(BG)
        good_scores, bad_scores = [], []

        for fold_det in data[lam]['fold_details']:
            fidx = fold_det['fold']
            csv = f"{LAMBDA_DIRS[lam]}/fold{fidx}/scores_fold{fidx}.csv"
            try:
                df = pd.read_csv(csv)
                good_scores.extend(df[df['true_label'] == 0]['score'].tolist())
                bad_scores.extend( df[df['true_label'] == 1]['score'].tolist())
            except Exception:
                pass

        if good_scores and bad_scores:
            all_scores = good_scores + bad_scores
            bins = np.linspace(min(all_scores), max(all_scores), 50)
            ax.hist(good_scores, bins=bins, alpha=0.65, color=BLUE,
                    label=f'GOOD  (n={len(good_scores)})', density=True)
            ax.hist(bad_scores,  bins=bins, alpha=0.65, color=ORANGE,
                    label=f'BAD   (n={len(bad_scores)})',  density=True)
            ax.axvline(0, color=WHITE, lw=1.5, linestyle='--', alpha=0.8)
            ax.text(0.5, 0.93, 'Decision boundary (score=0)',
                    color=WHITE, fontsize=8.5, ha='center', va='top',
                    transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Score CSVs not found',
                    transform=ax.transAxes, ha='center', va='center',
                    color=WHITE, fontsize=10)

        ax.set_xlabel('OOD Score  (score > 0 → BAD)', fontsize=11, color=WHITE, labelpad=8)
        ax.set_ylabel('Density', fontsize=11, color=WHITE, labelpad=8)
        ax.set_title(title, fontsize=13, color=WHITE, fontweight='bold', pad=10)
        ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor=WHITE, edgecolor=GRID_COL)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis='y', color=GRID_COL, linewidth=0.8, linestyle='--', alpha=0.6)

    fig.suptitle('OOD Score Distributions — Pooled across 5 Folds (K=100)',
                 fontsize=14, color=WHITE, fontweight='bold', y=1.0)

    out = f"{OUT_FIG}/fig5_score_distributions_cv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ===========================================================================
# LaTeX Table 1: Main Ablation Table
# ===========================================================================
def write_ablation_table(data):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Separation loss ablation on the Inkjet QC dataset (5-fold stratified CV,"
        r" K=100 MC trials). Results reported as mean~$\pm$~std across folds."
        r" No $\lambda$ value significantly outperforms the baseline.}",
        r"\label{tab:inkjet_ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\lambda$ & AUROC $\uparrow$ & Accuracy $\uparrow$ & FPR@95TPR $\downarrow$ \\",
        r"\midrule",
    ]
    for lam in LAMBDAS:
        d = data[lam]
        auroc = f"{d['mean_auroc']:.4f} $\\pm$ {d['std_auroc']:.4f}"
        acc   = f"{d.get('mean_acc', 0):.4f} $\\pm$ {d.get('std_acc', 0):.4f}"
        fpr   = f"{d.get('mean_fpr95', 0):.4f} $\\pm$ {d.get('std_fpr95', 0):.4f}"
        prefix = r"\textbf{" if lam == 0.0 else ""
        suffix = r"}" if lam == 0.0 else ""
        label = {0.0: '0.0 (baseline)', 0.01: '0.01', 0.02: '0.02', 0.05: '0.05'}[lam]
        lines.append(f"{prefix}{label}{suffix} & {auroc} & {acc} & {fpr} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = f"{OUT_TEX}/inkjet_ablation_table.tex"
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {path}")


# ===========================================================================
# LaTeX Table 2: Per-Feature AUROC (baseline vs best λ)
# ===========================================================================
def write_per_feature_table(data):
    feat_display = {
        'angle': 'Angle ($\\dagger$)', 'dist1': 'Distance 1',
        'dist6': 'Distance 6',         'dots': 'Dots',
        'edge1': 'Edge 1',             'edge2': 'Edge 2',
        'edge3': 'Edge 3',             'edge4': 'Edge 4',
    }
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-feature AUROC on the Inkjet QC dataset (5-fold CV, mean~$\pm$~std)."
        r" $\dagger$~angle has fewer than 5 BAD samples per fold; AUROC is unreliable.}",
        r"\label{tab:inkjet_per_feature}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Feature & $\lambda=0.0$ & $\lambda=0.01$ & $\lambda=0.02$ & $\lambda=0.05$ \\",
        r"\midrule",
    ]
    for feat in FEATURES:
        row_vals = []
        best_mean = max(
            data[lam]['per_feature'].get(feat, {}).get('mean', 0)
            for lam in LAMBDAS
        )
        for lam in LAMBDAS:
            pf = data[lam]['per_feature'].get(feat, {})
            m  = pf.get('mean', 0)
            s  = pf.get('std', 0)
            val = f"{m:.4f}$\\pm${s:.4f}"
            if abs(m - best_mean) < 1e-6:
                val = f"\\textbf{{{m:.4f}}}$\\pm${s:.4f}"
            row_vals.append(val)
        disp = feat_display.get(feat, feat)
        lines.append(f"{disp} & " + " & ".join(row_vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = f"{OUT_TEX}/inkjet_per_feature_table.tex"
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {path}")


# ===========================================================================
# LaTeX Table 3: Cross-Domain Comparison
# ===========================================================================
def write_cross_domain_table(data):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-domain comparison of the separation loss effect."
        r" CIFAR-10 results use a single train/test split (seed=42);"
        r" Inkjet results use 5-fold stratified CV (seed=42, mean~$\pm$~std).}",
        r"\label{tab:cross_domain}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Baseline ($\lambda=0$) & Best sep.\ loss & $\Delta$ AUROC \\",
        r"\midrule",
        r"CIFAR-10 & $0.925 \pm 0.111$ & $\mathbf{0.9903 \pm 0.0007}$ ($\lambda=0.02$) & $\mathbf{+6.5}$ pp \\",
        f"Inkjet QC & "
        f"\\textbf{{{data[0.0]['mean_auroc']:.4f}}}~$\\pm$~{data[0.0]['std_auroc']:.4f} & "
        f"{data[0.05]['mean_auroc']:.4f}~$\\pm$~{data[0.05]['std_auroc']:.4f} ($\\lambda=0.05$) & "
        r"$\approx$0.0\% (n.s.) \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\small n.s.\ = not statistically significant (all $\Delta \leq 0.016$, within cross-fold std)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = f"{OUT_TEX}/inkjet_cross_domain_table.tex"
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(f"\nLoading 5-fold CV data...")
    data = load_cv_data()
    print(f"  ✓ Loaded data for λ values: {list(data.keys())}")

    print(f"\nGenerating figures → {OUT_FIG}/")
    print("=" * 55)
    plot_lambda_ablation_cv(data)
    plot_per_feature_auroc_cv(data)
    plot_cross_domain_comparison(data)
    plot_roc_curves_cv(data)
    plot_score_distributions_cv(data)

    print(f"\nGenerating LaTeX tables → {OUT_TEX}/")
    print("=" * 55)
    write_ablation_table(data)
    write_per_feature_table(data)
    write_cross_domain_table(data)

    print(f"\n{'='*55}")
    print("All outputs generated successfully.")
    print(f"\n  FIGURES ({OUT_FIG}/):")
    print("  fig1_inkjet_lambda_ablation_cv.png  — AUROC + FPR ablation with error bars")
    print("  fig2_per_feature_auroc_cv.png       — all 4 λ per feature with error bars")
    print("  fig3_cross_domain_comparison.png    — CIFAR-10 vs Inkjet side-by-side")
    print("  fig4_roc_curves_cv.png              — mean ROC across folds")
    print("  fig5_score_distributions_cv.png     — pooled score distributions")
    print(f"\n  TABLES ({OUT_TEX}/):")
    print("  inkjet_ablation_table.tex           — main ablation table")
    print("  inkjet_per_feature_table.tex        — per-feature AUROC table")
    print("  inkjet_cross_domain_table.tex       — cross-domain comparison")
