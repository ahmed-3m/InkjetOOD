# Inkjet CDM — Experiment Log

Full chronological log of all inkjet CDM experiments.

---

## Model Architecture

```
CDM UNet — Multi-Head Conditioning
  Parameters:    ~9.3M (inkjet model, smaller than CIFAR-10's 35.7M)
  Input:         YOLO-cropped inkjet print region (bbox crop)
  Conditioning:  4 heads — template_id, feature_id, quality_label, bbox_coords
  Schedule:      Cosine (Nichol & Dhariwal 2021), 1000 timesteps, epsilon prediction
  Scoring:       Algorithm 1 — difference method (pred_good_err - pred_bad_err)
```

---

## Dataset Statistics

```
Total samples:   1,327 (from feature_cls.csv)
Train split:     80% = ~1,061 samples (with BAD oversampling ×3 → effective train size larger)
Test split:      20% = 266 samples (FIXED, seed=42, no oversampling)

Test set breakdown:
  Templates:  A=105, B=28, C=133
  Labels:     GOOD=174, BAD=92  (imbalance: 1.89:1)

Per-feature test counts:
  angle:  27 GOOD,  3 BAD   ← WARNING: only 3 BAD — AUROC unreliable
  dist1:  21 GOOD, 10 BAD
  dist6:  30 GOOD,  6 BAD
  dots:   26 GOOD, 11 BAD
  edge1:  12 GOOD, 16 BAD
  edge2:  18 GOOD, 28 BAD
  edge3:  16 GOOD, 12 BAD
  edge4:  24 GOOD,  6 BAD
```

---

## Experiment 1 — Baseline CDM (λ=0)

**Goal:** Establish CDM performance without separation loss.

**Config:**
```
sep_loss_weight:  0.0
batch_size:       128  (single forward pass, fits 32GB VRAM comfortably)
epochs:           100
schedule:         cosine
num_trials:       100 (K=100 at evaluation)
out_dir:          results/inkjet_lambda0
```

**Key fix applied:** Trainer gated separation loss computation behind `if sep_loss_weight > 0.0`
to prevent 3× VRAM usage and OOM errors on the baseline run.

**Results (K=100):**
```
Overall  AUROC=0.8325  Acc=0.7895  FPR@95TPR=0.8161  N=266
```

**Output files:**
- `results/inkjet_lambda0/cdm_best.pt`
- `results/inkjet_lambda0_k100/scores.csv`
- `results/inkjet_lambda0.log`

---

## Experiment 2 — Separation Loss λ=0.02

**Goal:** Apply CIFAR-10 optimal λ directly to inkjet.

**Config:**
```
sep_loss_weight:  0.02
batch_size:       64  (3 forward passes per step: main + good + bad)
epochs:           100
```

**Results (K=100, re-evaluated):**
```
Overall  AUROC=0.8541  Acc=0.8008  FPR@95TPR=0.6609  N=266
```

*Note: First evaluation (K=50) showed 0.8581; variance ±0.005 at K=50.*

**Output files:**
- `results/inkjet_lambda0.02/cdm_best.pt`
- `results/inkjet_lambda0.02_k100/scores.csv`
- `results/inkjet_lambda0.02.log`

---

## Experiment 3 — Separation Loss λ=0.01

**Goal:** Test CIFAR-10 sub-optimal value on inkjet (one step below λ=0.02).

**Config:**
```
sep_loss_weight:  0.01
batch_size:       64
epochs:           100
num_trials:       100
```

**Results (K=100, inline with training via --eval_after):**
```
Overall  AUROC=0.8603  Acc=0.8158  FPR@95TPR=0.6264  N=266
```

**→ Best AUROC of all λ values on inkjet.**

**Output files:**
- `results/inkjet_lambda0.01/cdm_best.pt`
- `results/inkjet_lambda0.01/scores_final.csv`
- `results/inkjet_lambda0.01.log`

---

## Experiment 4 — Separation Loss λ=0.05

**Goal:** Test CIFAR-10 upper boundary of optimal zone on inkjet.

**Config:**
```
sep_loss_weight:  0.05
batch_size:       64
epochs:           100
num_trials:       100
```

**Results (K=100):**
```
Overall  AUROC=0.8553  Acc=0.8233  FPR@95TPR=0.5287  N=266
```

**→ Best FPR@95TPR of all λ values. Best accuracy.**

**Output files:**
- `results/inkjet_lambda0.05/cdm_best.pt`
- `results/inkjet_lambda0.05/scores_final.csv`
- `results/inkjet_lambda0.05.log`

---

## Experiment 5 — 5-Fold CV Ablation (Phase 2, FINAL)

**Date:** March 1-2, 2026
**Goal:** Rigorous evaluation with confidence intervals. All four λ values evaluated under identical conditions (batch=64 for fair comparison).

**Config (all runs):**
```
batch_size:       64 (all runs — eliminates confound)
epochs:           100 per fold
seed:             42
num_trials:       100
n_folds:          5 (stratified on label)
Total GPU time:   ~24h
```

**Results:**

| λ | AUROC (5-fold CV) | Accuracy | FPR@95TPR |
|---|:---:|:---:|:---:|
| **0.0** | **0.8673 ± 0.0230** | **0.8094 ± 0.0151** | 0.5631 ± 0.1697 |
| 0.01 | 0.8628 ± 0.0286 | 0.7928 ± 0.0291 | **0.5516 ± 0.1841** |
| 0.02 | 0.8510 ± 0.0326 | 0.8003 ± 0.0246 | 0.6240 ± 0.1334 |
| 0.05 | 0.8670 ± 0.0256 | 0.8071 ± 0.0241 | 0.5700 ± 0.1948 |

**→ Separation loss does NOT improve over baseline on inkjet. All λ values within std.**

**Output files:**
- `results/cv_lambda0.0/cv_summary.json` (+ per-fold scores)
- `results/cv_lambda0.01/cv_summary.json`
- `results/cv_lambda0.02/cv_summary.json`
- `results/cv_lambda0.05/cv_summary.json`

---

## Key Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 5-fold CV for final results | 5-fold stratified | Single-split results unreliable; CV provides confidence intervals |
| batch=64 for ALL runs | 64 | Eliminates batch-size confound; sep loss needs 3 forward passes |
| λ values tested | 0.0, 0.01, 0.02, 0.05 | Covers CIFAR-10 optimal zone; more points not justified |
| K=100 for final eval | K=100 | Stable estimates; CIFAR-10 K-ablation showed diminishing returns after K=25 |
| seed=42 | 42 | Matches CIFAR-10 primary seed |

---

## Known Issues

1. **`angle` feature**: Only 2-4 BAD samples per fold. AUROC highly variable (0.54–0.95). Report with caveat.
2. **MC variance**: ±0.003 at K=100. Negligible compared to cross-fold variance (±0.025).
3. **Separation loss ineffective**: The primary thesis finding for inkjet. See cross-domain comparison in RESULTS.md.

---

## Hardware

- **GPU:** CUDA device 3 (Quadro GV100, 32 GB VRAM)
- **Environment:** conda environment with `requirements.txt` installed
- **Training time per fold:** ~70 min (100 epochs, batch=64, λ>0)
- **Evaluation time per fold (K=100):** ~45 seconds
- **Total 5-fold CV ablation time:** ~24h (4 λ values × 5 folds)
