# InkjetOOD — Conditional Diffusion Model for Inkjet Print Quality Control

[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-ahmed--3m%2FInkjetOOD-blue)](https://huggingface.co/ahmed-3m/InkjetOOD)
[![GitHub](https://img.shields.io/badge/GitHub-ahmed--3m%2FInkjetOOD-black)](https://github.com/ahmed-3m/InkjetOOD)

**Thesis:** *Conditional Diffusion Models as Generative Classifiers for Out-of-Distribution Detection in Inkjet Print Quality Control*
**Author:** Ahmed Mohammed — MSc AI, Johannes Kepler University Linz
**Supervisor:** Univ.-Prof. Dr. Sepp Hochreiter · Industrial Partner: PROFACTOR GmbH

---

## What This Does

This repository implements a **Conditional Diffusion Model (CDM)** for out-of-distribution (OOD) detection in inkjet print quality control. A YOLOv8 detector extracts per-feature crops (dots, edge roughness, angle, spacing) from printed samples. A 4-head UNet CDM trained only on good (in-distribution) samples then scores each crop: **high reconstruction error = OOD (defective print)**. No defect labels are needed at training time.

**Key results (full 5-fold cross-validation, thesis main result):**
- CDM AUROC: **0.8673 ± 0.023** across 8 print features
- YOLOv8 feature detector: **95.0% mAP@50**

---

## Pretrained Weights on HuggingFace

All trained model weights live at: **[https://huggingface.co/ahmed-3m/InkjetOOD](https://huggingface.co/ahmed-3m/InkjetOOD)**

| File on HF | Description | AUROC | Params |
|---|---|---|---|
| `models/cdm_v3_yolo_bbox.pt` | **CDM λ=0.01, base_ch=64 (proposed)** — single-split best | 0.8603 single-split | 9.33 M |
| `models/cdm_v3_baseline.pt` | CDM λ=0, base_ch=128 (5-fold CV result) | 0.8673 ± 0.023 CV | 34.2 M |
| `models/cdm_v3_test.pt` | CDM λ=0.01, base_ch=64 (dev checkpoint) | ≈0.85 | 9.33 M |
| `models/yolo_best.pt` | YOLOv8 feature detector (8 print features) | mAP@50=0.950 | 25.86 M |
| `models/semantic_mismatch_angle_model.pt` | Per-feature CDM: angle (~9.67:1 imbalance) | ~0.82 | 8.94 M |
| `models/semantic_mismatch_dist1_model.pt` | Per-feature CDM: dist1 | ~0.89 | 8.94 M |
| `models/semantic_mismatch_dots_model.pt` | Per-feature CDM: dots (best feature) | ~0.96 | 8.94 M |

---

## Quick Start

### Path A — Evaluate with pretrained weights (no training needed, ~5 min)

```bash
# 1. Clone and install
git clone https://github.com/ahmed-3m/InkjetOOD
cd InkjetOOD
pip install -r requirements.txt

# 2. Download pretrained weights from HuggingFace
python download_weights.py

# 3. Set your dataset path (see "Dataset" section below)
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset

# 4. Evaluate
python evaluate.py \
    --checkpoint models/cdm_proposed.pt \
    --num_trials 100 \
    --out_dir results/eval_pretrained
```

Expected output: `results/eval_pretrained/metrics.json` with AUROC ≈ 0.860.

---

### Path B — Train from scratch (single split, ~3 hours on 32 GB GPU)

```bash
# 1. Set dataset path
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset

# 2. Download YOLO weights (needed for bbox extraction during training)
python download_weights.py

# 3. Train baseline (λ=0, no separation loss)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 100 \
    --batch_size 128 \
    --sep_loss_weight 0.0 \
    --schedule cosine \
    --eval_after \
    --out_dir results/train_baseline

# 4. Train proposed model (λ=0.01)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 100 \
    --batch_size 64 \
    --sep_loss_weight 0.01 \
    --schedule cosine \
    --eval_after \
    --out_dir results/train_proposed
```

> **Note on batch size:** With separation loss (λ>0), the training loop performs 3 forward passes per step.
> Use `--batch_size 64` with λ>0 to avoid OOM on a 32 GB GPU. Use `--batch_size 128` for λ=0.

---

### Path C — 5-fold cross-validation (thesis main result, ~15 hours on 32 GB GPU)

```bash
# Reproduces the thesis 5-fold CV result: AUROC 0.8673 ± 0.023
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset

CUDA_VISIBLE_DEVICES=0 python run_cv.py \
    --epochs 100 \
    --batch_size 128 \
    --sep_loss_weight 0.0 \
    --num_trials 100 \
    --n_folds 5 \
    --seed 42 \
    --out_dir results/cv_lambda0
```

Or run all λ values to reproduce the full CV ablation:
```bash
bash scripts/run_cv_ablation.sh
```

---

## Step-by-Step Installation

### Requirements

- Python 3.9+
- CUDA GPU (tested on Quadro GV100 32 GB; minimum ~16 GB recommended)
- ~5 GB disk for model weights and dataset metadata

### Install

```bash
git clone https://github.com/ahmed-3m/InkjetOOD
cd InkjetOOD
pip install -r requirements.txt
```

All dependencies are in `requirements.txt`. Core packages:
- `torch>=2.0`, `torchvision>=0.15`
- `ultralytics>=8.0` (YOLOv8)
- `scikit-learn>=1.2`
- `huggingface-hub>=0.20`

---

## Step-by-Step: Evaluate with Pretrained Weights

**Step 1 — Download weights**
```bash
python download_weights.py
```
This downloads to `models/`:
- `models/yolo_best.pt` — YOLOv8 feature detector
- `models/cdm_baseline.pt` — CDM λ=0 (5-fold CV result: 0.8673)
- `models/cdm_proposed.pt` — CDM λ=0.01 (single-split result: 0.8603)

**Step 2 — Prepare your dataset**

The dataset is the FTI_Zer0P inkjet print dataset. Metadata and crop examples are also available on HuggingFace at [`ahmed-3m/InkjetOOD`](https://huggingface.co/ahmed-3m/InkjetOOD) under `data/`.

Set the path to the root directory containing all print images and `metadata.csv`:
```bash
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
```
Or pass it directly: `--metadata /path/to/metadata.csv --img_dir /path/to/images`

**Step 3 — Run evaluation**
```bash
python evaluate.py \
    --checkpoint models/cdm_proposed.pt \
    --num_trials 100 \
    --out_dir results/eval_pretrained
```

**Step 4 — View results**
```
results/eval_pretrained/
├── metrics.json     ← overall AUROC, FPR@95, per-feature breakdown
├── scores.csv       ← per-sample scores and true labels
└── roc_curve.png    ← ROC curve plot
```

---

## Step-by-Step: Train from Scratch

**Step 1 — Set paths and download YOLO weights**
```bash
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset
python download_weights.py
```

**Step 2 — (Optional) Verify your setup with a quick smoke run**
```bash
python train.py \
    --epochs 5 \
    --batch_size 32 \
    --sep_loss_weight 0.0 \
    --out_dir results/smoke_test
```
This takes ~10 min and confirms the data pipeline and model are working.

**Step 3 — Train the baseline (λ=0)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 100 \
    --batch_size 128 \
    --sep_loss_weight 0.0 \
    --schedule cosine \
    --eval_after \
    --out_dir results/train_baseline
```

**Step 4 — Train the proposed model (λ=0.01)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 100 \
    --batch_size 64 \
    --sep_loss_weight 0.01 \
    --schedule cosine \
    --eval_after \
    --out_dir results/train_proposed
```

**Step 5 — Compare results**
```bash
python scripts/plot_inkjet_results.py --out_dir results/figures
```

---

## Step-by-Step: 5-Fold Cross-Validation

The 5-fold CV is the **primary evaluation protocol** in the thesis (more robust than single-split on this small dataset, ~1330 samples).

**Step 1 — Run CV for each λ value**
```bash
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset

for LAM in 0.0 0.01 0.02 0.05; do
  CUDA_VISIBLE_DEVICES=0 python run_cv.py \
      --sep_loss_weight $LAM \
      --epochs 100 \
      --batch_size 128 \
      --num_trials 100 \
      --seed 42 \
      --out_dir results/cv_lambda${LAM}
done
```

Or use the provided script:
```bash
bash scripts/run_cv_ablation.sh
```

**Step 2 — View CV summaries**

Each run produces `results/cv_lambdaX.X/cv_summary.json`:
```json
{
  "mean_auroc": 0.8673,
  "std_auroc": 0.0230,
  "fold_aurocs": [0.871, 0.852, 0.834, 0.902, 0.878]
}
```

**Step 3 — Plot CV results**
```bash
python scripts/plot_inkjet_results_cv.py --out_dir results/figures_cv
```

---

## Step-by-Step: Separation Loss Ablation

```bash
export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset

python run_ablation.py \
    --weights 0.0 0.01 0.02 0.05 \
    --epochs 100 \
    --batch_size 128 \
    --num_trials 100 \
    --out_dir results/ablation

# View summary
cat results/ablation/ablation_results.json
```

---

## Results

### 5-Fold Cross-Validation (Thesis Main Result)

| λ | Mean AUROC | Std | Mean FPR@95 | Protocol |
|---|---|---|---|---|
| **0.0 (baseline)** | **0.8673** | **0.023** | 0.563 | 5-fold CV, K=100 |
| 0.01 | 0.8628 | 0.029 | 0.552 | 5-fold CV, K=100 |
| 0.02 | 0.8510 | 0.033 | 0.624 | 5-fold CV, K=100 |
| 0.05 | 0.8670 | 0.026 | 0.570 | 5-fold CV, K=100 |

> The 5-fold CV result (λ=0, 0.8673 ± 0.023) is the primary thesis result.

### Single-Split Evaluation (Seed=42, N=266 test)

| λ | AUROC | FPR@95 | K |
|---|---|---|---|
| 0.0 (baseline) | 0.8325 | 0.816 | 100 |
| **0.01 (proposed)** | **0.8603** | **0.626** | 100 |
| 0.02 | 0.8541 | 0.661 | 100 |
| 0.05 | 0.8553 | 0.529 | 100 |

> On a single fixed split, λ=0.01 is clearly best (+2.8 pp AUROC, −19 pp FPR95 vs baseline).
> The CV result is more reliable because of the small dataset size (~1330 samples).

### Per-Feature AUROC (5-Fold CV, λ=0)

| Feature | AUROC | Std |
|---|---|---|
| dots | 0.956 | 0.035 |
| dist6 | 0.936 | 0.067 |
| dist1 | 0.887 | 0.073 |
| angle | 0.817 | 0.138 |
| edge2 | 0.813 | 0.031 |
| edge1 | 0.796 | 0.138 |
| edge4 | 0.762 | 0.099 |
| edge3 | 0.744 | 0.076 |

### YOLO Feature Detector

| Metric | Value |
|---|---|
| Precision | 94.1% |
| Recall | 89.1% |
| mAP@50 | **95.0%** |
| mAP@50-95 | 84.7% |

### Comparison with Discriminative Baselines (5-Fold CV)

| Method | AUROC | Type |
|---|---|---|
| ResNet-FullImg | **0.9451** | Discriminative |
| Dual-Branch | 0.9274 | Discriminative |
| ResNet-CropOnly | 0.8471 | Discriminative |
| **CDM (ours)** | **0.8673** | Generative (OOD) |

> CDMs are weaker than supervised discriminative baselines on AUROC but offer OOD detection without defect labels.

---

## Architecture

```
Input crop (128×128×3)
        │
   ┌────▼─────────────────────────────────┐
   │  UNet encoder (4 residual blocks)    │
   │  Conditioning (4 heads fused):       │
   │  ├── Template type  (A / B / C)      │
   │  ├── Feature class  (8 YOLO classes) │
   │  ├── Quality label  (GOOD / BAD)     │
   │  └── YOLO bbox      (4 coordinates) │
   │  UNet decoder (4 residual blocks)    │
   └────┬─────────────────────────────────┘
        │  predicted noise
   ┌────▼────────────────────────────────┐
   │  Scoring (Algorithm 1)              │
   │  K=50 uniform timestep samples      │
   │  score = mean(e₀ − e₁) over K       │
   │  e₀ = error(GOOD label)             │
   │  e₁ = error(BAD label)              │
   └─────────────────────────────────────┘
```

- **Parameters:** 9.33 M (base_ch=64, proposed model) · 34.2 M (base_ch=128, baseline model)
- **Diffusion schedule:** Cosine, T=1000
- **Training:** 100 epochs, AdamW (lr=2e-4), batch=64–128
- **Separation loss:** pushes GOOD/BAD embeddings apart in latent space
- **YOLO detector:** YOLOv8-based, 25.86 M parameters, mAP@50=0.950

---

## Repository Structure

```
InkjetOOD/
├── README.md                      ← this file
├── requirements.txt               ← pip dependencies
├── download_weights.py            ← download pretrained weights from HF
│
├── train.py                       ← training entry point
├── evaluate.py                    ← evaluation entry point
├── run_cv.py                      ← 5-fold cross-validation
├── run_ablation.py                ← separation-loss ablation study
│
├── configs/
│   └── default.py                 ← all hyperparameters (edit paths here)
│
├── src/
│   ├── model.py                   ← UNet with 4-head conditioning (~9.3 M params)
│   ├── diffusion.py               ← cosine/linear diffusion schedule
│   ├── dataset.py                 ← dataset: YOLO integration + oversampling
│   ├── trainer.py                 ← training loop + separation loss
│   ├── evaluate.py                ← Algorithm 1 scoring + metrics
│   └── cross_validation.py       ← 5-fold CV wrapper
│
├── scripts/
│   ├── run_train.sh               ← quick training launcher
│   ├── run_cv.sh                  ← 5-fold CV launcher
│   ├── run_cv_ablation.sh         ← runs CV for all λ values
│   ├── run_ablation.sh            ← separation-loss ablation launcher
│   └── plot_inkjet_results_cv.py  ← generates CV figures
│
├── yolo_feature_detection/        ← YOLO training scripts and results
│   ├── src/                       ← training, inference, evaluation
│   └── docs/                      ← architecture, methodology, results docs
│
├── cv_lambda_ablation/            ← stored CV results for all λ values
│   ├── cv_lambda0.0/              ← λ=0: per-fold metrics + summary
│   ├── cv_lambda0.01/
│   ├── cv_lambda0.02/
│   └── cv_lambda0.05/
│
├── single_split_lambda_ablation/  ← single-split ablation results
├── per_feature_evaluation/        ← per-feature figures and tables
└── docs/                          ← RESULTS.md, EXPERIMENTS.md, FIGURES.md
```

---

## Dataset

The **FTI_Zer0P** inkjet print dataset consists of ~1330 labeled crop samples across 8 print features (dots, edge roughness ×4, angle, spacing ×2). Each sample has:
- A 128×128 crop extracted by the YOLO detector
- A binary GOOD/BAD quality label
- Feature class and template type metadata

Dataset metadata and crop examples are available on HuggingFace at [`ahmed-3m/InkjetOOD`](https://huggingface.co/ahmed-3m/InkjetOOD) under `data/`.

To use the dataset:
1. Download from HF: `python download_weights.py --also-data`
2. Or set `INKJET_DATA_DIR` to your local copy of the raw images

---

## Configuration

All paths and hyperparameters are in `configs/default.py`. For a new machine, set these two environment variables:

```bash
# Required: path to FTI_Zer0P dataset root (must contain metadata.csv)
export INKJET_DATA_DIR=/path/to/dataset

# Optional: path to YOLO weights (default: models/yolo_best.pt after download_weights.py)
export INKJET_YOLO_WEIGHTS=/path/to/yolo_best.pt
```

Key hyperparameters (from `configs/default.py`):

| Parameter | Default | Notes |
|---|---|---|
| `SEP_LOSS_WEIGHT` | `0.01` | λ for class separation loss (inkjet optimal) |
| `EPOCHS` | `100` | Training epochs |
| `BATCH_SIZE` | `128` | Use 64 when λ>0 (3 forward passes) |
| `NUM_TRIALS` | `50` | K in Algorithm 1 (use 100 for final evaluation) |
| `SCHEDULE` | `cosine` | Diffusion noise schedule |
| `IMG_SIZE` | `128` | Crop size fed to the CDM |
| `N_FOLDS` | `5` | Folds for cross-validation |

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{mohammed2026inkjet,
  title   = {Conditional Diffusion Models as Generative Classifiers for
             Out-of-Distribution Detection in Inkjet Print Quality Control},
  author  = {Mohammed, Ahmed},
  school  = {Johannes Kepler University Linz},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## License

Code: MIT License. Dataset (FTI_Zer0P): CC BY 4.0 — see `LICENSE` files.
