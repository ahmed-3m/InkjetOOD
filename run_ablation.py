"""
run_ablation.py
===============
Separation-loss ablation study (thesis Section 6.3 / Table 6.2).

Trains the CDM with different values of the class separation loss weight λ
and records the AUROC for each.  Results are saved to results/ablation/.

Usage
-----
    python run_ablation.py

    # To run only specific λ values:
    python run_ablation.py --weights 0.0 0.01 0.1

    # Fewer epochs for a quick sanity check:
    python run_ablation.py --epochs 30 --num_trials 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.default import (
    METADATA_CSV, IMG_DIR, RESULTS_DIR,
    BASE_CHANNELS, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    IMG_SIZE, CONF_THRESHOLD, NUM_TRIALS, SEED,
    SEP_ABLATION_WEIGHTS,
)
from src.model          import NoisePredictorV3
from src.diffusion      import DiffusionSchedule
from src.dataset        import InkjetCDMDataset
from src.trainer        import train_cdm
from src.evaluate       import evaluate_cdm, FEATURE_NAMES


def parse_args():
    p = argparse.ArgumentParser(description='Separation-loss ablation study')
    p.add_argument('--metadata',    type=str,   default=str(METADATA_CSV))
    p.add_argument('--img_dir',     type=str,   default=str(IMG_DIR))
    p.add_argument('--out_dir',     type=str,   default=str(RESULTS_DIR / 'ablation'))
    p.add_argument('--weights',     type=float, nargs='+',
                   default=SEP_ABLATION_WEIGHTS,
                   help='List of λ values to ablate.')
    p.add_argument('--epochs',      type=int,   default=EPOCHS)
    p.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    p.add_argument('--num_trials',  type=int,   default=NUM_TRIALS)
    p.add_argument('--seed',        type=int,   default=SEED)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Separation Loss Ablation Study")
    print(f"  λ values: {args.weights}")
    print(f"{'='*65}\n")

    # ── Shared data split ────────────────────────────────────────────────
    df = pd.read_csv(args.metadata)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['label']
    )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Build datasets once (no YOLO needed for ablation)
    train_ds = InkjetCDMDataset(
        train_df, args.img_dir, yolo_model=None,
        transform=transform, crop_size=IMG_SIZE,
        conf_threshold=CONF_THRESHOLD,
        oversample_minority=True, max_ratio=3.0, augment=True,
    )
    test_ds = InkjetCDMDataset(
        test_df, args.img_dir, yolo_model=None,
        transform=transform, crop_size=IMG_SIZE,
        conf_threshold=CONF_THRESHOLD,
        oversample_minority=False, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=8, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=8,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Run ablation ─────────────────────────────────────────────────────
    ablation_results: list[dict] = []

    for lam in args.weights:
        print(f"\n[Ablation] λ = {lam}")
        print("-" * 40)

        # Fresh model for each λ
        model    = NoisePredictorV3(base_channels=BASE_CHANNELS).to(device)
        schedule = DiffusionSchedule(schedule='cosine', device=device)

        ckpt_path = out_dir / f"cdm_sep{lam}.pt"
        train_cdm(
            model, schedule, train_loader,
            epochs=args.epochs, lr=LEARNING_RATE,
            sep_loss_weight=lam, device=device,
            save_path=ckpt_path,
        )

        # Load best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        # Evaluate
        scores_df = evaluate_cdm(model, schedule, test_loader,
                                 num_trials=args.num_trials, device=device)

        try:
            overall_auroc = roc_auc_score(scores_df['true_label'], scores_df['score'])
        except Exception:
            overall_auroc = None

        per_feat: dict = {}
        for fid in sorted(scores_df['feature_id'].unique()):
            sub  = scores_df[scores_df['feature_id'] == fid]
            name = FEATURE_NAMES[fid] if fid < len(FEATURE_NAMES) else str(fid)
            try:
                per_feat[name] = float(roc_auc_score(sub['true_label'], sub['score']))
            except Exception:
                per_feat[name] = None

        ablation_results.append({
            'sep_loss_weight': lam,
            'overall_auroc'  : overall_auroc,
            'per_feature'    : per_feat,
        })

        print(f"  → Overall AUROC = {overall_auroc:.4f}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ABLATION SUMMARY")
    print(f"{'='*65}")

    header = f"  {'λ':>6}  {'AUROC':>7}"
    for name in (FEATURE_NAMES or []):
        header += f"  {name[:8]:>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in ablation_results:
        row = f"  {r['sep_loss_weight']:>6.3f}  {r['overall_auroc'] or 0:>7.4f}"
        for name in (FEATURE_NAMES or []):
            val = r['per_feature'].get(name)
            row += f"  {val:>8.4f}" if val is not None else f"  {'N/A':>8}"
        print(row)

    # Save
    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[Ablation] Results saved to {out_dir / 'ablation_results.json'}")


if __name__ == '__main__':
    main()
