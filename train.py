"""
train.py
========
Entry point for training the CDM with class separation loss.

Usage examples
--------------
# Standard training (all defaults from configs/default.py):
    python train.py

# Override specific hyperparameters:
    python train.py --epochs 150 --batch_size 128 --sep_loss_weight 0.01

# Use YOLO for dynamic bbox extraction:
    python train.py --use_yolo

# Train on a single feature for quick testing:
    python train.py --feature angle --epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

# Allow relative imports when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.default import (
    METADATA_CSV, IMG_DIR, YOLO_WEIGHTS, RESULTS_DIR,
    BASE_CHANNELS, NUM_TIMESTEPS, SCHEDULE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SEP_LOSS_WEIGHT,
    IMG_SIZE, CONF_THRESHOLD, OVERSAMPLE_MINORITY, MAX_RATIO, AUGMENT,
    NUM_TRIALS, SEED,
)
from src.model     import NoisePredictorV3
from src.diffusion import DiffusionSchedule
from src.dataset   import InkjetCDMDataset, META_TO_YOLO
from src.trainer   import train_cdm
from src.evaluate  import evaluate_cdm, save_results


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train the Conditional Diffusion Model for Inkjet QC'
    )
    p.add_argument('--metadata',        type=str,   default=str(METADATA_CSV))
    p.add_argument('--img_dir',         type=str,   default=str(IMG_DIR))
    p.add_argument('--out_dir',         type=str,   default=str(RESULTS_DIR / 'train'))
    p.add_argument('--epochs',          type=int,   default=EPOCHS)
    p.add_argument('--batch_size',      type=int,   default=BATCH_SIZE,
                   help='Training batch size. Default=128 (tuned for 32 GB VRAM).')
    p.add_argument('--lr',              type=float, default=LEARNING_RATE)
    p.add_argument('--sep_loss_weight', type=float, default=SEP_LOSS_WEIGHT,
                   help='λ for the class separation loss. 0.0 = disabled (DDPM baseline).')
    p.add_argument('--img_size',        type=int,   default=IMG_SIZE)
    p.add_argument('--num_trials',      type=int,   default=NUM_TRIALS)
    p.add_argument('--base_channels',   type=int,   default=BASE_CHANNELS)
    p.add_argument('--schedule',        type=str,   default=SCHEDULE,
                   choices=['cosine', 'linear'])
    p.add_argument('--feature',         type=str,   default=None,
                   help='If set, train only on this feature (e.g. angle, e.rought2).')
    p.add_argument('--use_yolo',        action='store_true',
                   help='Use YOLOv8 for dynamic bbox extraction.')
    p.add_argument('--eval_after',      action='store_true',
                   help='Run evaluation on test set after training.')
    p.add_argument('--seed',            type=int,   default=SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  CDM for Inkjet Print Quality Control — Training")
    print(f"{'='*65}")
    print(f"  Device           : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    if device == 'cuda':
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM             : {vram_gb:.1f} GB")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Sep loss (λ)     : {args.sep_loss_weight}")
    print(f"  Diffusion sched. : {args.schedule}")
    print(f"  Image size       : {args.img_size}×{args.img_size}")
    print(f"  Output dir       : {out_dir}")
    print(f"{'='*65}\n")

    # ── YOLO model ──────────────────────────────────────────────────────
    yolo_model = None
    if args.use_yolo:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(YOLO_WEIGHTS))
            print(f"[YOLO] Loaded weights from {YOLO_WEIGHTS}")
        except Exception as exc:
            print(f"[YOLO] Warning: could not load model — {exc}")
            print("[YOLO] Falling back to fixed bounding boxes.")

    # ── Metadata ────────────────────────────────────────────────────────
    df = pd.read_csv(args.metadata)
    print(f"[Data] Loaded {len(df)} samples.")

    if args.feature:
        # Filter to a single feature for targeted experiments
        raw_name = args.feature                       # e.g. 'e.rought2'
        yolo_name = META_TO_YOLO.get(raw_name, raw_name)
        # Accept both 'e.rought2' and 'edge2' style
        df = df[df['feature'].isin([raw_name, yolo_name])].reset_index(drop=True)
        print(f"[Data] Filtered to feature '{args.feature}': {len(df)} samples.")

    # Stratified 80/20 split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['label']
    )
    print(f"[Data] Train: {len(train_df)}   Test: {len(test_df)}")

    # ── Transforms ──────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = InkjetCDMDataset(
        train_df, args.img_dir, yolo_model=yolo_model,
        transform=transform, crop_size=args.img_size,
        conf_threshold=CONF_THRESHOLD,
        oversample_minority=OVERSAMPLE_MINORITY,
        max_ratio=MAX_RATIO, augment=AUGMENT,
    )
    test_ds = InkjetCDMDataset(
        test_df, args.img_dir, yolo_model=yolo_model,
        transform=transform, crop_size=args.img_size,
        conf_threshold=CONF_THRESHOLD,
        oversample_minority=False, augment=False,
    )

    # Adapt num_workers to the host platform
    _nw = min(4, os.cpu_count() or 1)
    _pw = (_nw > 0) and (sys.platform != 'win32')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=_nw, pin_memory=True,
        persistent_workers=_pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=8,
        shuffle=False, num_workers=min(2, _nw), pin_memory=True,
    )

    # ── Model + schedule ─────────────────────────────────────────────────
    model    = NoisePredictorV3(base_channels=args.base_channels).to(device)
    schedule = DiffusionSchedule(num_timesteps=NUM_TIMESTEPS,
                                 schedule=args.schedule, device=device)
    print(f"[Model] Parameters: {model.count_parameters():,}")

    # ── Training ─────────────────────────────────────────────────────────
    ckpt_path = out_dir / 'cdm_best.pt'
    train_cdm(
        model, schedule, train_loader,
        epochs=args.epochs,
        lr=args.lr,
        sep_loss_weight=args.sep_loss_weight,
        device=device,
        save_path=ckpt_path,
    )

    # ── Optional post-training evaluation ────────────────────────────────
    if args.eval_after:
        print("\n[Eval] Loading best checkpoint for evaluation …")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        scores_df = evaluate_cdm(
            model, schedule, test_loader,
            num_trials=args.num_trials, device=device
        )
        save_results(scores_df, out_dir, tag='final')

    print(f"\n[Done] All outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
