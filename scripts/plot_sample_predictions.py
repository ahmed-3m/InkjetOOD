"""
plot_sample_predictions.py  (v3 — confident + diverse)
========================================================
Generate a "Sample Predictions – Correctly Classified Examples" figure
using the best inkjet CDM model (λ=0.01, AUROC=0.8603).

Usage:
  cd thesis_cdm_final
  python scripts/plot_sample_predictions.py

Output: results/figures/fig6_sample_predictions.png
"""

import os, sys, random
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict

from configs.default import (
    METADATA_CSV, IMG_DIR, YOLO_WEIGHTS,
    BASE_CHANNELS, NUM_TIMESTEPS, SCHEDULE,
    IMG_SIZE, CONF_THRESHOLD, SEED,
)
from src.model     import NoisePredictorV3
from src.diffusion import DiffusionSchedule
from src.dataset   import (
    META_TO_YOLO, YOLO_CLASS_NAMES,
    FEATURE_TO_ID, TEMPLATE_TO_ID, FEATURE_TO_TEMPLATE,
    WIDE_PADDING_FEATURES,
)
from src.evaluate  import score_sample

# ── Config ────────────────────────────────────────────────────────────────────
CKPT          = Path(__file__).resolve().parents[1] / "results/inkjet_lambda0.01/cdm_best.pt"
OUT_FILE      = Path(__file__).resolve().parents[1] / "results/figures/fig6_sample_predictions.png"
NUM_TRIALS    = 20     # 20 trials is plenty for a qualitative figure
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_N      = 6      # 3 cols × 2 rows
N_CANDIDATES  = 8      # score up to N per (feature, label)
MIN_SCORE_ABS = 0.0005 # reject |score| below this (too borderline)

# Files to skip — visually ambiguous despite correct CDM classification
EXCLUDE_FILES = {"0971.png", "1019.png"}

FEATURE_PRETTY = {
    "angle":    "Angle",  "dist.1": "Dist 1", "dist.6": "Dist 6",
    "dots":     "Dots",   "e.rought1": "Edge 1", "e.rought2": "Edge 2",
    "e.rought3":"Edge 3", "e.rought4": "Edge 4",
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CDM model
# ─────────────────────────────────────────────────────────────────────────────
print(f"[1] Loading CDM …")
schedule = DiffusionSchedule(num_timesteps=NUM_TIMESTEPS, schedule=SCHEDULE, device=DEVICE)
model    = NoisePredictorV3(base_channels=BASE_CHANNELS).to(DEVICE)
state    = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
elif "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.eval()
print(f"   OK (device={DEVICE})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load YOLO
# ─────────────────────────────────────────────────────────────────────────────
print("[2] Loading YOLO …")
yolo = None
try:
    from ultralytics import YOLO as _YOLO
    yolo = _YOLO(str(YOLO_WEIGHTS))
    print("   YOLO OK")
except Exception as e:
    print(f"   WARN: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build test split (reproduce exactly the same split as training)
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Building test split …")
df = pd.read_csv(str(METADATA_CSV))
_, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Score candidates — feature by feature
# ─────────────────────────────────────────────────────────────────────────────
def get_bbox(img_path, feature_yolo):
    if yolo is not None:
        try:
            res   = yolo(str(img_path), conf=CONF_THRESHOLD, verbose=False)
            boxes = res[0].boxes
            # prefer exact class match
            for b in boxes:
                if YOLO_CLASS_NAMES[int(b.cls[0])] == feature_yolo:
                    return b.xywhn[0].cpu().tolist(), int(b.cls[0])
            if len(boxes):
                b = boxes[0]
                return b.xywhn[0].cpu().tolist(), int(b.cls[0])
        except Exception:
            pass
    fid = FEATURE_TO_ID.get(feature_yolo, 0)
    return [0.5, 0.5, 0.4, 0.4], fid


def score_one(file_name, feature_raw, true_label_csv):
    """Run YOLO + CDM on a single image. Returns dict or None on error."""
    img_path     = Path(IMG_DIR) / file_name
    true_quality = 1 - true_label_csv   # 0=GOOD, 1=BAD
    feature_yolo = META_TO_YOLO.get(feature_raw, "edge1")
    template_str = FEATURE_TO_TEMPLATE.get(feature_yolo, "C")
    template_id  = TEMPLATE_TO_ID[template_str]
    feature_id   = FEATURE_TO_ID.get(feature_yolo, 0)

    bbox_xywhn, _ = get_bbox(img_path, feature_yolo)

    img_pil = Image.open(img_path).convert("RGB")
    W, H    = img_pil.size
    x_c, y_c, w, h = bbox_xywhn
    pad  = 0.3 if feature_yolo in WIDE_PADDING_FEATURES else 0.2
    w2   = w * (1 + pad); h2 = h * (1 + pad)
    crop = img_pil.crop((
        max(0, int((x_c - w2/2)*W)), max(0, int((y_c - h2/2)*H)),
        min(W, int((x_c + w2/2)*W)), min(H, int((y_c + h2/2)*H)),
    ))

    img_t  = transform(crop).unsqueeze(0).to(DEVICE)
    tid    = torch.tensor([template_id], device=DEVICE)
    fid    = torch.tensor([feature_id],  device=DEVICE)
    bbox_t = torch.tensor([bbox_xywhn],  device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        sc = score_sample(model, schedule, img_t, tid, fid, bbox_t,
                          num_trials=NUM_TRIALS, device=DEVICE)

    pred_q  = 1 if sc > 0 else 0
    correct = (pred_q == true_quality)
    return {
        "file_name":     file_name,
        "img_path":      img_path,
        "feature_raw":   feature_raw,
        "feature_yolo":  feature_yolo,
        "bbox":          bbox_xywhn,
        "true_label_csv": true_label_csv,
        "true_quality":  true_quality,
        "score":         sc,
        "pred_q":        pred_q,
        "correct":       correct,
    }


# Candidate pairs: all 8 feature types, GOOD and BAD
PAIRS = [
    ("angle",     1), ("angle",     0),
    ("dist.1",    1), ("dist.1",    0),
    ("e.rought1", 1), ("e.rought1", 0),
    ("e.rought2", 1), ("e.rought2", 0),
    ("dist.6",    1), ("dist.6",    0),
    ("e.rought3", 1), ("e.rought3", 0),
    ("e.rought4", 1), ("e.rought4", 0),
    ("dots",      1), ("dots",      0),
]

print(f"[4] Scoring candidates (K={NUM_TRIALS} trials each) …")
all_scored = []

for feat, lbl in PAIRS:
    sub = test_df[(test_df["feature"] == feat) & (test_df["label"] == lbl)]
    if len(sub) == 0:
        continue

    # Score up to N_CANDIDATES, stop early if we find 2 confident correct ones
    confident_found = 0
    for _, row in sub.head(N_CANDIDATES).iterrows():
        lbl_str = "GOOD" if lbl == 1 else "BAD"
        if row["file_name"] in EXCLUDE_FILES:
            print(f"  {row['file_name']}  {feat} / {lbl_str}  [EXCLUDED]")
            continue
        print(f"  {row['file_name']}  {feat} / {lbl_str}", end=" … ", flush=True)
        result = score_one(row["file_name"], feat, lbl)
        all_scored.append(result)
        status = "✓" if result["correct"] else "✗"
        conf   = "★" if abs(result["score"]) >= MIN_SCORE_ABS else "·"
        print(f"score={result['score']:+.4f} {status}{conf}")
        if result["correct"] and abs(result["score"]) >= MIN_SCORE_ABS:
            confident_found += 1
        # Once we have 2 confident correct ones for this pair, move on
        if confident_found >= 2:
            break

print(f"\n  Total scored: {len(all_scored)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Select 6 display items: 3 GOOD + 3 BAD, one per feature, confident
# ─────────────────────────────────────────────────────────────────────────────
confident_correct = [
    s for s in all_scored
    if s["correct"] and abs(s["score"]) >= MIN_SCORE_ABS
]
print(f"  Confident+correct: {len(confident_correct)}")

by_q = defaultdict(list)
for s in confident_correct:
    by_q[s["true_quality"]].append(s)
for q in by_q:
    by_q[q].sort(key=lambda x: abs(x["score"]), reverse=True)


def pick_diverse(pool, n):
    """Pick n items from pool, at most 1 per feature type, sorted by |score|."""
    seen, selected = set(), []
    for s in pool:
        if s["feature_raw"] not in seen:
            selected.append(s)
            seen.add(s["feature_raw"])
        if len(selected) == n:
            break
    return selected


good_sel = pick_diverse(by_q.get(0, []), 3)
bad_sel  = pick_diverse(by_q.get(1, []), 3)

# If we didn't find 3 confident GOODs, pick the best remaining correct GOODs
if len(good_sel) < 3:
    all_good = [s for s in all_scored if s["true_quality"]==0 and s["correct"] and s not in good_sel]
    all_good.sort(key=lambda x: abs(x["score"]), reverse=True)
    for s in all_good:
        if s["feature_raw"] not in {x["feature_raw"] for x in good_sel}:
            good_sel.append(s)
        if len(good_sel) == 3: break
    # fallback without diverse check if still short
    for s in all_good:
        if len(good_sel) == 3: break
        if s not in good_sel: good_sel.append(s)

# Same for BAD
if len(bad_sel) < 3:
    all_bad = [s for s in all_scored if s["true_quality"]==1 and s["correct"] and s not in bad_sel]
    all_bad.sort(key=lambda x: abs(x["score"]), reverse=True)
    for s in all_bad:
        if s["feature_raw"] not in {x["feature_raw"] for x in bad_sel}:
            bad_sel.append(s)
        if len(bad_sel) == 3: break
    for s in all_bad:
        if len(bad_sel) == 3: break
        if s not in bad_sel: bad_sel.append(s)

# Strictly top row GOOD, bottom row BAD
good_sel = good_sel[:3]
bad_sel  = bad_sel[:3]
display_items = good_sel + bad_sel

display_items = display_items[:TARGET_N]
print(f"\n  Final selection ({len(display_items)} items):")
for s in display_items:
    lbl = "GOOD" if s["true_quality"] == 0 else "BAD"
    mark = "✓" if s["correct"] else "✗"
    print(f"    {s['feature_raw']:12s}  {lbl:4s}  score={s['score']:+.4f}  {mark}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Render figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Rendering figure …")


def draw_bbox_overlay(img_path, bbox_xywhn, is_good):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x_c, y_c, w, h = bbox_xywhn
    pad = 0.2
    w2  = w * (1 + pad); h2 = h * (1 + pad)
    x1  = max(0, int((x_c - w2/2)*W))
    y1  = max(0, int((y_c - h2/2)*H))
    x2  = min(W, int((x_c + w2/2)*W))
    y2  = min(H, int((y_c + h2/2)*H))

    draw  = ImageDraw.Draw(img)
    color = "#00EE00" if is_good else "#EE1111"
    lw    = max(3, W // 200)
    for i in range(lw):
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)

    try:
        import matplotlib.font_manager as fm
        font = None
        for candidate in ["DejaVuSans-Bold.ttf", "Arial-Bold.ttf",
                           "LiberationSans-Bold.ttf"]:
            for p in fm.findSystemFonts():
                if os.path.basename(p).lower() == candidate.lower():
                    font = ImageFont.truetype(p, max(14, W // 55))
                    break
            if font:
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    tag = "GOOD" if is_good else "BAD"
    draw.text((x1+6, y1+3), tag, fill="black", font=font)    # shadow
    draw.text((x1+5, y1+2), tag, fill=color,   font=font)

    return np.array(img)


nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
fig.patch.set_facecolor("white")

for i, item in enumerate(display_items):
    ax      = axes[i // ncols][i % ncols]
    is_good = (item["true_quality"] == 0)

    img_arr = draw_bbox_overlay(item["img_path"], item["bbox"], is_good)
    ax.imshow(img_arr)
    ax.axis("off")

    feat_str  = FEATURE_PRETTY.get(item["feature_raw"], item["feature_raw"])
    qual_str  = "GOOD" if is_good else "BAD"
    status    = "Correct" if item["correct"] else "Wrong"
    title_col = "#1a6e1a" if is_good else "#b80000"

    ax.set_title(f"{feat_str} – {qual_str} ({status})",
                 fontsize=13, fontweight="bold", color=title_col, pad=7)

    badge = f"score: {item['score']:+.4f}"
    ax.text(0.98, 0.02, badge, transform=ax.transAxes,
            fontsize=8, color="white", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111", alpha=0.70, ec="none"))

for j in range(len(display_items), nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle(
    "Sample Predictions \u2014 Correctly Classified Examples\n"
    f"CDM  \u03bb=0.01 \u00b7 K={NUM_TRIALS} MC trials \u00b7 AUROC\u202f=\u202f0.8603",
    fontsize=14, fontweight="bold", color="#111", y=1.01,
)
plt.tight_layout(pad=1.5)
plt.savefig(str(OUT_FILE), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n\u2713  Saved: {OUT_FILE}")
