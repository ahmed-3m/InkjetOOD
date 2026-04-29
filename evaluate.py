"""
evaluate.py  (entry point)
==========================
Load a trained CDM checkpoint and evaluate it on the test set.

Usage
-----
# Evaluate a trained model:
    python evaluate.py --checkpoint results/train/cdm_best.pt

# Evaluate with more scoring trials for a stable AUROC estimate:
    python evaluate.py --checkpoint results/train/cdm_best.pt --num_trials 100

# Evaluate on a single feature only:
    python evaluate.py --checkpoint results/train/cdm_best.pt --feature angle
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.default import (
    METADATA_CSV, IMG_DIR, RESULTS_DIR,
    BASE_CHANNELS, NUM_TIMESTEPS, SCHEDULE,
    IMG_SIZE, CONF_THRESHOLD, NUM_TRIALS, SEED,
)
from src.model     import NoisePredictorV3
from src.diffusion import DiffusionSchedule
from src.dataset   import InkjetCDMDataset, META_TO_YOLO
from src.evaluate  import evaluate_cdm, save_results


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a trained CDM checkpoint')
    p.add_argument('--checkpoint',    type=str, required=True,
                   help='Path to the .pt checkpoint.')
    p.add_argument('--metadata',      type=str, default=str(METADATA_CSV))
    p.add_argument('--img_dir',       type=str, default=str(IMG_DIR))
    p.add_argument('--out_dir',       type=str, default=str(RESULTS_DIR / 'eval'))
    p.add_argument('--num_trials',    type=int, default=NUM_TRIALS)
    p.add_argument('--img_size',      type=int, default=IMG_SIZE)
    p.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    p.add_argument('--schedule',      type=str, default=SCHEDULE,
                   choices=['cosine', 'linear'])
    p.add_argument('--feature',       type=str, default=None,
                   help='Evaluate only on this feature (e.g. angle, e.rought2).')
    p.add_argument('--batch_size',    type=int, default=8)
    p.add_argument('--seed',          type=int, default=SEED)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata and reproduce train/test split ─────────────────────
    df = pd.read_csv(args.metadata)
    if args.feature:
        yname = META_TO_YOLO.get(args.feature, args.feature)
        df    = df[df['feature'].isin([args.feature, yname])].reset_index(drop=True)
        print(f"[Eval] Feature filter '{args.feature}': {len(df)} samples.")

    _, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['label']
    )

    # ── Dataset / loader ─────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    test_ds = InkjetCDMDataset(
        test_df, args.img_dir, yolo_model=None,
        transform=transform, crop_size=args.img_size,
        conf_threshold=CONF_THRESHOLD,
        oversample_minority=False, augment=False,
    )
    _nw = min(2, os.cpu_count() or 1)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=_nw, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model    = NoisePredictorV3(base_channels=args.base_channels).to(device)
    schedule = DiffusionSchedule(num_timesteps=NUM_TIMESTEPS,
                                 schedule=args.schedule, device=device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
    if 'config' in ckpt:
        print(f"[Eval] Training config: {ckpt['config']}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    scores_df = evaluate_cdm(
        model, schedule, test_loader,
        num_trials=args.num_trials, device=device
    )
    save_results(scores_df, out_dir)
    print(f"\n[Done] Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
