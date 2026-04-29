# Conditional Diffusion Model for Inkjet Print Quality Control

**Thesis:** *Conditional Diffusion Models for Out-of-Distribution Detection in Industrial Quality Control*

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/RESULTS.md`](docs/RESULTS.md) | **All final numbers** — AUROC, FPR@95TPR, per-feature, all λ |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Full experiment log — configs, outputs, decisions |
| [`docs/FIGURES.md`](docs/FIGURES.md) | Figure index with thesis captions, where to use each figure |

---

## Quick Summary of Final Results

| Model | AUROC | FPR@95TPR |
|-------|:---:|:---:|
| Baseline CDM (λ=0) | 0.8325 | 0.8161 |
| **Proposed CDM (λ=0.01)** | **0.8603** | **0.6264** |
| Proposed CDM (λ=0.05, best FPR) | 0.8553 | **0.5287** |

> All results: K=100 MC trials, N=266 test samples, seed=42.

---

## Dataset

Download the PROFACTOR FTI_Zer0P Dataset from Zenodo:

  https://zenodo.org/records/11444566
  DOI: 10.5281/zenodo.11444566

After downloading, organize as:

```
data/
├── images/          ← all .png image files
└── metadata.csv     ← CSV with columns: file_name, label, feature
```

**`metadata.csv` schema:**

| Column | Type | Description |
|--------|------|-------------|
| `file_name` | str | Image filename (e.g. `"0971.png"`) |
| `label` | int | `1` = GOOD sample, `0` = BAD/defect sample |
| `feature` | str | One of: `angle`, `dist.1`, `dist.6`, `dots`, `e.rought1`, `e.rought2`, `e.rought3`, `e.rought4` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train
python train.py \
    --metadata ./data/metadata.csv \
    --img_dir ./data/images \
    --epochs 100 \
    --sep_loss_weight 0.02 \
    --eval_after

# 3. Evaluate a checkpoint
python evaluate.py \
    --checkpoint results/train/cdm_best.pt \
    --metadata ./data/metadata.csv \
    --img_dir ./data/images \
    --num_trials 100
```

---

## Environment

```bash
# Option A: set paths via environment variables
export INKJET_DATA_ROOT=./data/images
export INKJET_METADATA=./data/metadata.csv
python train.py

# Option B: pass paths directly as CLI arguments
python train.py --metadata /your/path/metadata.csv --img_dir /your/path/images
```

---

## Directory Structure

```
InkjetOOD/
│
├── train.py            ← Training entry point
├── evaluate.py         ← Evaluation entry point
├── run_cv.py           ← 5-fold cross-validation
├── run_ablation.py     ← Separation-loss ablation study
├── requirements.txt    ← Python dependencies
├── .gitignore
│
├── configs/
│   └── default.py      ← All hyperparameters (paths via env vars or CLI)
│
├── src/
│   ├── model.py        ← CDM UNet with 4-head conditioning
│   ├── diffusion.py    ← Cosine / linear diffusion schedule
│   ├── dataset.py      ← Dataset: YOLO integration + oversampling
│   ├── trainer.py      ← Training loop + separation loss
│   ├── evaluate.py     ← Algorithm 1 scoring + metrics
│   └── cross_validation.py  ← 5-fold CV wrapper
│
├── scripts/
│   ├── plot_inkjet_results.py      ← Generates all 5 thesis figures
│   ├── plot_inkjet_results_cv.py   ← CV figures + LaTeX tables
│   ├── run_train.sh
│   ├── run_cv.sh
│   └── run_ablation.sh
│
├── docs/               ← Thesis documentation
│   ├── RESULTS.md      ← Complete verified results
│   ├── EXPERIMENTS.md  ← Experiment log
│   └── FIGURES.md      ← Figure index with captions
│
└── results/            ← All training/eval outputs (gitignored)
```

---

## Reproducing Results

**Train all λ values sequentially:**
```bash
# Baseline
python train.py --epochs 100 --batch_size 128 \
    --sep_loss_weight 0.0 --schedule cosine --eval_after --num_trials 100 \
    --out_dir results/inkjet_lambda0 2>&1 | tee results/inkjet_lambda0.log

# Best AUROC
python train.py --epochs 100 --batch_size 64 \
    --sep_loss_weight 0.01 --schedule cosine --eval_after --num_trials 100 \
    --out_dir results/inkjet_lambda0.01 2>&1 | tee results/inkjet_lambda0.01.log
```

**Re-evaluate any checkpoint (K=100):**
```bash
python evaluate.py \
    --checkpoint results/inkjet_lambda0.01/cdm_best.pt \
    --num_trials 100 \
    --out_dir results/inkjet_lambda0.01_eval
```

**Regenerate all thesis figures:**
```bash
python scripts/plot_inkjet_results.py
```

---

## Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Noise schedule | Cosine | Smoother gradient at early timesteps → better texture discrimination |
| Conditioning | Template + Feature + Quality + BBox (4 heads) | Minimal sufficient context for per-class fidelity |
| λ (inkjet) | 0.01 (best AUROC) | Transferred directly from CIFAR-10 optimal zone [0.01–0.05] |
| Batch size | 64 (sep loss) / 128 (baseline) | Sep loss requires 3 forward passes; 128 causes OOM |
| K=100 trials | K=100 | Stable AUROC estimates on small dataset (±0.003 variance) |
| No full λ sweep | 4 points only | 266 test samples → noise overwhelms fine λ differences |

---

## Licence
The code in this repository is released under the **MIT Licence**.  
The accompanying dataset (FTI\_Zer0P Dataset 2023) is published separately on
Zenodo under **CC BY 4.0** (DOI: 10.5281/zenodo.11444566).