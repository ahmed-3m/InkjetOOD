"""
configs/default.py
==================
All hyperparameters for the Inkjet CDM experiments.

PATH SETUP (edit these two lines, or set environment variables):
  export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
  export INKJET_YOLO_WEIGHTS=/path/to/yolo_best.pt

After running  `python download_weights.py`  the YOLO weights default
to models/yolo_best.pt in this repo — no env var needed.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  — set via environment variables or edit the fallback defaults below
# ---------------------------------------------------------------------------

# Dataset root: directory containing metadata.csv and all print images.
# Download metadata + crop examples from https://huggingface.co/ahmed-3m/InkjetOOD
DATASET_ROOT = Path(os.environ.get(
    'INKJET_DATA_DIR',
    'data/inkjet_dataset'          # fallback: relative to repo root
))
METADATA_CSV = DATASET_ROOT / 'metadata.csv'
IMG_DIR      = DATASET_ROOT

# YOLO feature detector weights.
# Run `python download_weights.py` to fetch from HuggingFace, or set env var.
YOLO_WEIGHTS = Path(os.environ.get(
    'INKJET_YOLO_WEIGHTS',
    str(Path(__file__).resolve().parents[1] / 'models' / 'yolo_best.pt')
))

# Default output directory for all experiments
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

BASE_CHANNELS  = 64       # UNet width; 64 → 9.33 M params (proposed model)
                          #                128 → 34.2 M params (baseline, pretrained cdm_v3_baseline.pt)
                          # Override with --base_channels 128 to match cdm_v3_baseline.pt
NUM_TEMPLATES  = 3        # A / B / C template types
NUM_FEATURES   = 8        # 8 YOLO print feature classes
TIME_DIM       = 256      # sinusoidal timestep embedding dimension

# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

NUM_TIMESTEPS  = 1000
SCHEDULE       = 'cosine' # 'cosine' (thesis default) or 'linear'

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Batch size notes (32 GB VRAM GPU):
#   λ=0.0  (1 forward pass):  batch=128 fits (~10 GB)
#   λ>0    (3 forward passes): batch=64  fits (~15 GB)
# Override with --batch_size 64 when sep_loss_weight > 0.
BATCH_SIZE       = 128
EPOCHS           = 100
LEARNING_RATE    = 2e-4
WEIGHT_DECAY     = 1e-4

# Class separation loss weight (λ).
#
# Inkjet note: lambda=0.01 is retained only as an exploratory single-split
# default. The thesis headline 5-fold CV result uses lambda=0, K=100:
# AUROC = 0.8673 +/- 0.0230. Non-zero lambda settings did not significantly
# improve inkjet AUROC.
#
# CIFAR-10 optimal (DiffusionOOD repo): λ=0.02
#   Final three-seed AUROC = 0.9903 +/- 0.0007 versus 0.925 +/- 0.111 at lambda=0.
#
# Set λ=0.0 to recover the standard DDPM baseline.
SEP_LOSS_WEIGHT  = 0.01

# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------

IMG_SIZE            = 128    # crop size fed to the CDM (pixels)
CONF_THRESHOLD      = 0.3    # minimum YOLO detection confidence
OVERSAMPLE_MINORITY = True   # duplicate BAD samples for imbalanced features
MAX_RATIO           = 3.0    # maximum GOOD:BAD ratio after oversampling
AUGMENT             = True   # random flips/brightness on oversampled BAD samples

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# K in Algorithm 1 (number of Monte Carlo timestep trials per sample).
# Use K=100 for the retained thesis 5-fold CV estimates. K=50 remains a
# faster exploratory setting when reproducing earlier single-split diagnostics.
NUM_TRIALS = 50

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

N_FOLDS    = 5
SEED       = 42

# ---------------------------------------------------------------------------
# Separation-loss ablation values
# ---------------------------------------------------------------------------

SEP_ABLATION_WEIGHTS = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1]
