from __future__ import annotations
"""
evaluate.py
===========
Evaluation utilities implementing the CDM classification algorithm.

Algorithm 1 – Diffusion Classifier Score
-----------------------------------------
For a test image x and K random timesteps t_1 … t_K:

    score(x) = (1/K) Σ_k [ ||ε_k − ε_θ(x_{t_k}, t_k, c=GOOD)||²
                          − ||ε_k − ε_θ(x_{t_k}, t_k, c=BAD )||² ]

Interpretation in the OOD framework:
  - If the model has learned a sharp distribution under c=GOOD, then
    in-distribution (GOOD) images will have low reconstruction error
    under c=GOOD.
  - Defective (BAD/OOD) images will have higher error under c=GOOD
    and lower error under c=BAD  →  score > 0  →  classified as defect.

The number of trials K trades off speed vs. estimate variance.
K=50 was found optimal in the ablation study (Section 5.x, Fig. x).
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

# Readable names for feature IDs (kept in sync with dataset.py)
FEATURE_NAMES = ['angle', 'dist1', 'dist6', 'dots', 'edge1', 'edge2', 'edge3', 'edge4']
TEMPLATE_NAMES = {0: 'A', 1: 'B', 2: 'C'}


# ---------------------------------------------------------------------------
# Core scoring function (Algorithm 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sample(
    model,
    schedule,
    img: torch.Tensor,
    template_id: torch.Tensor,
    feature_id: torch.Tensor,
    bbox: torch.Tensor,
    num_trials: int = 50,
    device: str = 'cuda',
) -> float:
    """
    Compute the OOD detection score for a single sample.

    Returns a scalar where:
        score > 0  →  BAD (OOD / defective)
        score < 0  →  GOOD (in-distribution)
    """
    errors_good: list[float] = []
    errors_bad:  list[float] = []

    for _ in range(num_trials):
        t = torch.randint(0, schedule.num_timesteps, (1,), device=device)
        x_t, noise = schedule.q_sample(img, t)

        q_good = torch.tensor([0], device=device)
        q_bad  = torch.tensor([1], device=device)

        pred_good = model(x_t, t, template_id, feature_id, q_good, bbox)
        pred_bad  = model(x_t, t, template_id, feature_id, q_bad,  bbox)

        errors_good.append(F.mse_loss(pred_good, noise).item())
        errors_bad.append( F.mse_loss(pred_bad,  noise).item())

    return float(np.mean(errors_good) - np.mean(errors_bad))


# ---------------------------------------------------------------------------
# Full evaluation over a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cdm(
    model,
    schedule,
    loader,
    num_trials: int = 50,
    device: str = 'cuda',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the CDM classifier over an entire test set and return a DataFrame
    with per-sample scores and labels.

    The DataFrame contains columns:
        score, true_label, pred_label, template_id, feature_id

    Metrics (AUROC, accuracy, confusion matrix) are printed per-feature
    and per-template, plus overall.
    """
    model.eval()
    records: list[dict] = []

    for batch in tqdm(loader, desc="Evaluating", disable=not verbose):
        images      = batch['image'].to(device)
        template_ids = batch['template_id'].to(device)
        feature_ids  = batch['feature_id'].to(device)
        true_labels  = batch['quality'].cpu().numpy()
        bboxes       = batch['bbox'].to(device)

        for i in range(images.size(0)):
            sc = score_sample(
                model, schedule,
                images[i:i+1],
                template_ids[i:i+1],
                feature_ids[i:i+1],
                bboxes[i:i+1],
                num_trials=num_trials,
                device=device,
            )
            records.append({
                'score'      : sc,
                'true_label' : int(true_labels[i]),
                'pred_label' : 1 if sc > 0 else 0,
                'template_id': template_ids[i].item(),
                'feature_id' : feature_ids[i].item(),
            })

    df = pd.DataFrame(records)

    if verbose:
        _print_metrics(df)

    return df


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------

def _fpr_at_95_tpr(y_true, scores):
    """FPR when TPR=0.95 (standard OOD benchmark metric)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr) - 1)])


def _print_metrics(df: pd.DataFrame):
    SEP = "=" * 70
    n_good_total = int((df['true_label'] == 0).sum())
    n_bad_total  = int((df['true_label'] == 1).sum())

    # Dataset sanity check — common pitfall with small datasets
    if n_good_total == 0 or n_bad_total == 0:
        print(f"\n  WARNING: test set has only one class "
              f"(GOOD={n_good_total}, BAD={n_bad_total}). "
              f"AUROC is undefined.")
        print("  Tip: use a larger dataset split or a different seed.")
        return

    # Overall
    print(f"\n{SEP}")
    print("  CDM EVALUATION RESULTS  (Diffusion Classifier — difference scoring)")
    print(SEP)
    try:
        auroc = roc_auc_score(df['true_label'], df['score'])
        acc   = accuracy_score(df['true_label'], df['pred_label'])
        fpr95 = _fpr_at_95_tpr(df['true_label'].values, df['score'].values)
        print(f"  Overall  AUROC={auroc:.4f}  Acc={acc:.4f}  "
              f"FPR@95TPR={fpr95:.4f}  "
              f"N={len(df)} (GOOD={n_good_total}, BAD={n_bad_total})")
    except Exception as e:
        print(f"  Overall metrics error: {e}")

    # Per-feature
    print(f"\n  {'Feature':<12}  {'AUROC':>6}  {'Acc':>6}  {'FPR@95':>7}  "
          f"{'N_GOOD':>6}  {'N_BAD':>5}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*5}")
    for fid in sorted(df['feature_id'].unique()):
        sub  = df[df['feature_id'] == fid]
        name = FEATURE_NAMES[fid] if fid < len(FEATURE_NAMES) else str(fid)
        ng   = int((sub['true_label'] == 0).sum())
        nb   = int((sub['true_label'] == 1).sum())
        if ng == 0 or nb == 0:
            print(f"  {name:<12}  {'--':>6}  {'--':>6}  {'--':>7}  "
                  f"{ng:>6}  {nb:>5}  [single class — AUROC undefined]")
            continue
        try:
            au  = roc_auc_score(sub['true_label'], sub['score'])
            ac  = accuracy_score(sub['true_label'], sub['pred_label'])
            fpr = _fpr_at_95_tpr(sub['true_label'].values, sub['score'].values)
            print(f"  {name:<12}  {au:>6.4f}  {ac:>6.4f}  {fpr:>7.4f}  {ng:>6}  {nb:>5}")
        except Exception as e:
            print(f"  {name:<12}  ERR({e})")

    # Per-template
    print(f"\n  {'Template':<10}  {'AUROC':>6}  {'Acc':>6}  {'FPR@95':>7}  {'N':>5}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}")
    for tid in sorted(df['template_id'].unique()):
        sub  = df[df['template_id'] == tid]
        name = TEMPLATE_NAMES.get(tid, str(tid))
        ng   = int((sub['true_label'] == 0).sum())
        nb   = int((sub['true_label'] == 1).sum())
        if ng == 0 or nb == 0:
            print(f"  {name:<10}  {'--':>6}  {'--':>6}  {'--':>7}  "
                  f"{len(sub):>5}  [single class]")
            continue
        try:
            au  = roc_auc_score(sub['true_label'], sub['score'])
            ac  = accuracy_score(sub['true_label'], sub['pred_label'])
            fpr = _fpr_at_95_tpr(sub['true_label'].values, sub['score'].values)
            print(f"  {name:<10}  {au:>6.4f}  {ac:>6.4f}  {fpr:>7.4f}  {len(sub):>5}")
        except Exception as e:
            print(f"  {name:<10}  ERR({e})")

    print(SEP)


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, out_dir: Union[str, Path], tag: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix  = f"_{tag}" if tag else ""

    # Per-sample CSV
    df.to_csv(out_dir / f"scores{suffix}.csv", index=False)

    # Aggregated JSON
    summary: dict = {}
    try:
        summary['overall_auroc'] = float(roc_auc_score(df['true_label'], df['score']))
        summary['overall_acc']   = float(accuracy_score(df['true_label'], df['pred_label']))
        summary['overall_fpr95'] = float(_fpr_at_95_tpr(df['true_label'].values, df['score'].values))
    except Exception:
        pass

    per_feature: dict = {}
    for fid in sorted(df['feature_id'].unique()):
        sub  = df[df['feature_id'] == fid]
        name = FEATURE_NAMES[fid] if fid < len(FEATURE_NAMES) else str(fid)
        try:
            per_feature[name] = {
                'auroc'   : float(roc_auc_score(sub['true_label'], sub['score'])),
                'accuracy': float(accuracy_score(sub['true_label'], sub['pred_label'])),
                'fpr95'   : float(_fpr_at_95_tpr(sub['true_label'].values, sub['score'].values)),
                'n_good'  : int((sub['true_label'] == 0).sum()),
                'n_bad'   : int((sub['true_label'] == 1).sum()),
            }
        except Exception:
            per_feature[name] = {'auroc': None}
    summary['per_feature'] = per_feature

    with open(out_dir / f"metrics{suffix}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[Results] Saved to {out_dir}")
    return summary
