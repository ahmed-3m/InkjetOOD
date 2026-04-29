"""
configs/default.py
==================
All hyperparameters for the CDM thesis experiments.
Edit this file to reproduce any experiment from the thesis.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (configured via environment variables or CLI arguments)
# ---------------------------------------------------------------------------
#
# Set environment variables before running, or pass paths via CLI flags:
#   export INKJET_DATA_ROOT=./data/images
#   export INKJET_METADATA=./data/metadata.csv
#   export YOLO_WEIGHTS=./yolo_weights/best.pt
#
# Or override at the command line:
#   python train.py --metadata ./data/metadata.csv --img_dir ./data/images

DATASET_ROOT = Path(os.environ.get('INKJET_DATA_ROOT', './data/images'))
METADATA_CSV = Path(os.environ.get('INKJET_METADATA', './data/metadata.csv'))
IMG_DIR      = DATASET_ROOT

YOLO_WEIGHTS = Path(os.environ.get('YOLO_WEIGHTS', './yolo_weights/best.pt'))

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

BASE_CHANNELS  = 64       # UNet width; 64 → ~9.3 M parameters
NUM_TEMPLATES  = 3        # A / B / C
NUM_FEATURES   = 8        # 8 YOLO classes
TIME_DIM       = 256      # sinusoidal embedding dimension

# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

NUM_TIMESTEPS  = 1000
SCHEDULE       = 'cosine' # 'cosine' (thesis default) or 'linear'

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# 32 GB VRAM:
#   - Without sep loss (λ=0.0): batch=128 fits (~10 GB, 1 forward pass)
#   - With sep loss (λ>0):      batch=64  fits (~15 GB, 3 forward passes)
# The default here applies to the baseline run; override with --batch_size 64 when λ>0.
BATCH_SIZE       = 128
EPOCHS           = 100
LEARNING_RATE    = 2e-4
WEIGHT_DECAY     = 1e-4

# Class separation loss weight (λ).
# λ=0.02 is the confirmed optimal value from the CIFAR-10 ablation study
# (AUROC=0.9911, the true peak — see results/figures/separation_loss_ablation_final.png).
# The original docs cited 0.9786 at λ=0.02 due to a coarse evaluation interval;
# the re-run with proper eval confirmed 0.9911 > 0.9869 (λ=0.01).
# Setting λ=0.0 recovers the standard DDPM baseline (AUROC=0.8025 on CIFAR-10).
SEP_LOSS_WEIGHT  = 0.02

# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------

IMG_SIZE            = 128    # crop size fed to the CDM
CONF_THRESHOLD      = 0.3    # minimum YOLO detection confidence
OVERSAMPLE_MINORITY = True   # duplicate BAD samples for imbalanced features
MAX_RATIO           = 3.0    # maximum GOOD:BAD ratio after oversampling
AUGMENT             = True   # augment oversampled BAD samples

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# K in Algorithm 1 (diffusion classifier scoring).
# K=50 gives stable AUROC estimates with acceptable inference time.
NUM_TRIALS = 50

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

N_FOLDS    = 5
SEED       = 42

# ---------------------------------------------------------------------------
# Separation-loss ablation values (Section 5.x)
# ---------------------------------------------------------------------------

SEP_ABLATION_WEIGHTS = [0.0, 0.001, 0.01, 0.05, 0.1]
