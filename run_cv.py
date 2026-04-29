"""
run_cv.py
=========
Run the full 5-fold cross-validation experiment described in the thesis.

Usage
-----
    python run_cv.py

    python run_cv.py --epochs 80 --sep_loss_weight 0.01 --out_dir results/cv_run1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from configs.default import (
    METADATA_CSV, IMG_DIR, YOLO_WEIGHTS, RESULTS_DIR,
    BASE_CHANNELS, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    SEP_LOSS_WEIGHT, IMG_SIZE, NUM_TRIALS, N_FOLDS, SEED,
)
from src.cross_validation import run_cross_validation


def parse_args():
    p = argparse.ArgumentParser(
        description='5-Fold Cross-Validation for the Inkjet CDM'
    )
    p.add_argument('--metadata',        type=str,   default=str(METADATA_CSV))
    p.add_argument('--img_dir',         type=str,   default=str(IMG_DIR))
    p.add_argument('--out_dir',         type=str,   default=str(RESULTS_DIR / 'cv'))
    p.add_argument('--epochs',          type=int,   default=EPOCHS)
    p.add_argument('--batch_size',      type=int,   default=BATCH_SIZE,
                   help='Batch size. Default=128 (tuned for 32 GB VRAM).')
    p.add_argument('--lr',              type=float, default=LEARNING_RATE)
    p.add_argument('--sep_loss_weight', type=float, default=SEP_LOSS_WEIGHT)
    p.add_argument('--img_size',        type=int,   default=IMG_SIZE)
    p.add_argument('--num_trials',      type=int,   default=NUM_TRIALS)
    p.add_argument('--n_folds',         type=int,   default=N_FOLDS)
    p.add_argument('--base_channels',   type=int,   default=BASE_CHANNELS)
    p.add_argument('--use_yolo',        action='store_true')
    p.add_argument('--seed',            type=int,   default=SEED)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo_model = None
    if args.use_yolo:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(YOLO_WEIGHTS))
            print(f"[YOLO] Loaded: {YOLO_WEIGHTS}")
        except Exception as exc:
            print(f"[YOLO] Warning: {exc} — falling back to fixed bboxes.")

    run_cross_validation(
        metadata_path    = args.metadata,
        img_dir          = args.img_dir,
        out_dir          = args.out_dir,
        yolo_model       = yolo_model,
        n_splits         = args.n_folds,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        sep_loss_weight  = args.sep_loss_weight,
        img_size         = args.img_size,
        num_trials       = args.num_trials,
        base_channels    = args.base_channels,
        device           = device,
        seed             = args.seed,
    )


if __name__ == '__main__':
    main()
