from __future__ import annotations
"""
cross_validation.py
===================
5-Fold stratified cross-validation wrapper for the CDM pipeline.

Each fold:
  1. Splits metadata into train / test with stratification on `label`.
  2. Trains a fresh CDM with the class separation loss.
  3. Evaluates using the diffusion classifier scoring algorithm.
  4. Records per-fold AUROC (overall + per-feature).

Final summary: mean ± std over the 5 folds.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model     import NoisePredictorV3
from src.diffusion import DiffusionSchedule
from src.dataset   import InkjetCDMDataset
from src.trainer   import train_cdm
from src.evaluate  import evaluate_cdm, save_results


# ---------------------------------------------------------------------------
# Shared transform
# ---------------------------------------------------------------------------

def get_transform(img_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Main CV function
# ---------------------------------------------------------------------------

def run_cross_validation(
    metadata_path: str | Path,
    img_dir: str | Path,
    out_dir: str | Path,
    yolo_model=None,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 2e-4,
    sep_loss_weight: float = 0.01,
    img_size: int = 128,
    num_trials: int = 50,
    base_channels: int = 64,
    device: str = 'cuda',
    seed: int = 42,
) -> dict:
    """
    Run n-fold cross-validation and return the aggregated results dict.

    Parameters
    ----------
    metadata_path   : path to metadata.csv
    img_dir         : root directory containing the images
    out_dir         : directory to write per-fold checkpoints and results
    yolo_model      : pre-loaded YOLO model (or None)
    n_splits        : number of CV folds
    epochs          : training epochs per fold
    batch_size      : training batch size
    lr              : learning rate
    sep_loss_weight : λ for the class separation auxiliary loss
    img_size        : crop resolution fed to the CDM
    num_trials      : K scoring trials per test sample
    base_channels   : UNet width multiplier
    device          : 'cuda' or 'cpu'
    seed            : random seed for reproducible splits

    Returns
    -------
    results : dict with fold and aggregate metrics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"[CV] Loaded {len(df)} samples from {metadata_path}")

    transform = get_transform(img_size)

    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels   = df['label'].values
    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, labels), start=1):
        print(f"\n{'='*65}")
        print(f"  FOLD {fold_idx} / {n_splits}")
        print(f"{'='*65}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # ── Datasets ────────────────────────────────────────────────
        train_ds = InkjetCDMDataset(
            train_df, img_dir, yolo_model=yolo_model,
            transform=transform, crop_size=img_size,
            oversample_minority=True, max_ratio=3.0, augment=True,
        )
        test_ds = InkjetCDMDataset(
            test_df, img_dir, yolo_model=yolo_model,
            transform=transform, crop_size=img_size,
            oversample_minority=False, augment=False,
        )

        _nw = min(4, os.cpu_count() or 1)
        _pw = (_nw > 0) and (sys.platform != 'win32')

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=_nw, pin_memory=True,
                                  persistent_workers=_pw)
        test_loader  = DataLoader(test_ds,  batch_size=4,          shuffle=False,
                                  num_workers=min(2, _nw), pin_memory=True)

        # ── Model + schedule ─────────────────────────────────────────
        model    = NoisePredictorV3(base_channels=base_channels).to(device)
        schedule = DiffusionSchedule(schedule='cosine', device=device)
        print(f"[CV] Model parameters: {model.count_parameters():,}")

        # ── Training ─────────────────────────────────────────────────
        ckpt_path = out_dir / f"fold{fold_idx}_best.pt"
        train_cdm(
            model, schedule, train_loader,
            epochs=epochs, lr=lr,
            sep_loss_weight=sep_loss_weight,
            device=device,
            save_path=ckpt_path,
        )

        # Load best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[CV] Loaded best checkpoint (epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f})")

        # ── Evaluation ────────────────────────────────────────────────
        fold_out = out_dir / f"fold{fold_idx}"
        scores_df = evaluate_cdm(model, schedule, test_loader, num_trials=num_trials, device=device)
        summary   = save_results(scores_df, fold_out, tag=f"fold{fold_idx}")

        fold_results.append({
            'fold'         : fold_idx,
            'overall_auroc': summary.get('overall_auroc'),
            'overall_acc'  : summary.get('overall_acc'),
            'overall_fpr95': summary.get('overall_fpr95'),
            'per_feature'  : summary.get('per_feature', {}),
        })

    # ── Aggregate over folds ─────────────────────────────────────────
    overall_aurocs = [r['overall_auroc'] for r in fold_results if r['overall_auroc'] is not None]
    overall_accs   = [r['overall_acc']   for r in fold_results if r.get('overall_acc') is not None]
    overall_fpr95s = [r['overall_fpr95'] for r in fold_results if r.get('overall_fpr95') is not None]
    agg: dict = {
        'n_folds'           : n_splits,
        'mean_auroc'        : float(np.mean(overall_aurocs)),
        'std_auroc'         : float(np.std(overall_aurocs)),
        'mean_acc'          : float(np.mean(overall_accs))   if overall_accs   else None,
        'std_acc'           : float(np.std(overall_accs))    if overall_accs   else None,
        'mean_fpr95'        : float(np.mean(overall_fpr95s)) if overall_fpr95s else None,
        'std_fpr95'         : float(np.std(overall_fpr95s))  if overall_fpr95s else None,
        'fold_aurocs'       : overall_aurocs,
        'fold_accs'         : overall_accs,
        'fold_fpr95s'       : overall_fpr95s,
        'sep_loss_weight'   : sep_loss_weight,
        'per_feature'       : _aggregate_per_feature(fold_results),
        'fold_details'      : fold_results,
    }

    # Save aggregated results
    with open(out_dir / 'cv_summary.json', 'w') as f:
        json.dump(agg, f, indent=2)

    # Print final table
    print(f"\n{'='*65}")
    print(f"  5-FOLD CROSS-VALIDATION SUMMARY  (λ={sep_loss_weight})")
    print(f"{'='*65}")
    print(f"  Overall AUROC    = {agg['mean_auroc']:.4f} ± {agg['std_auroc']:.4f}")
    if agg.get('mean_acc') is not None:
        print(f"  Overall Acc      = {agg['mean_acc']:.4f} ± {agg['std_acc']:.4f}")
    if agg.get('mean_fpr95') is not None:
        print(f"  Overall FPR@95   = {agg['mean_fpr95']:.4f} ± {agg['std_fpr95']:.4f}")
    print(f"\n  {'Feature':<12}  {'Mean AUROC':>10}  {'Std':>6}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}")
    for feat, stats in agg['per_feature'].items():
        print(f"  {feat:<12}  {stats['mean']:>10.4f}  {stats['std']:>6.4f}")
    print(f"{'='*65}\n")

    return agg


def _aggregate_per_feature(fold_results: list[dict]) -> dict:
    """Compute mean/std AUROC per feature across folds."""
    from collections import defaultdict
    per_feat: dict[str, list[float]] = defaultdict(list)
    for fold in fold_results:
        for feat, stats in fold.get('per_feature', {}).items():
            if stats.get('auroc') is not None:
                per_feat[feat].append(stats['auroc'])
    return {
        feat: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        for feat, vals in per_feat.items()
    }
