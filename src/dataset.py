from __future__ import annotations
"""
dataset.py
==========
Dataset class for the Inkjet Print Quality Control pipeline.

Data flow:
    metadata.csv  →  per-image rows  →  YOLOv8 detection  →  bbox crop  →  CDM input

Each row in metadata.csv corresponds to one (image, feature, label) triplet.
YOLOv8 localises the relevant feature region; the crop is resized to
`crop_size × crop_size` for CDM input.

Class-imbalance handling
------------------------
The `angle` feature has a severe GOOD:BAD ratio (~9.67:1).  When
`oversample_minority=True` the minority class (BAD) samples are repeated
with augmentation until the ratio drops to at most `max_ratio`:1.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Internal constants  (kept in sync with configs/default.py)
# ---------------------------------------------------------------------------

YOLO_CLASS_NAMES = ['angle', 'dist1', 'dist6', 'dots', 'edge1', 'edge2', 'edge3', 'edge4']

# Each feature may appear in more than one template; the mapping below uses
# the *primary* template (first assignment wins).
TEMPLATE_FEATURES = {
    'A': ['edge1', 'dist1', 'edge2'],
    'B': ['edge2', 'edge3'],
    'C': ['angle', 'dist6', 'dots', 'edge4'],
}
FEATURE_TO_TEMPLATE: dict[str, str] = {}
for _tmpl, _feats in TEMPLATE_FEATURES.items():
    for _f in _feats:
        FEATURE_TO_TEMPLATE.setdefault(_f, _tmpl)

TEMPLATE_TO_ID = {'A': 0, 'B': 1, 'C': 2}
FEATURE_TO_ID  = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}

META_TO_YOLO = {
    'angle'    : 'angle',
    'dist.1'   : 'dist1',
    'dist.6'   : 'dist6',
    'dots'     : 'dots',
    'e.rought1': 'edge1',
    'e.rought2': 'edge2',
    'e.rought3': 'edge3',
    'e.rought4': 'edge4',
}

# Features with narrow bounding boxes benefit from wider padding
WIDE_PADDING_FEATURES = {'edge2', 'edge3'}   # e.rought2 / e.rought3


# ---------------------------------------------------------------------------
# Augmentation pipeline for minority-class oversampling
# ---------------------------------------------------------------------------

def _augment_pil(img: Image.Image) -> Image.Image:
    """Apply a random light augmentation to a PIL image."""
    # Horizontal flip
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Slight rotation  (±10°)
    angle = np.random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    # Brightness / contrast jitter
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.85, 1.15))
    # Occasional mild blur
    if np.random.rand() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class InkjetCDMDataset(Dataset):
    """
    Loads inkjet quality-control images, runs YOLOv8 for feature detection,
    crops the relevant region, and returns a conditioning-ready sample.

    Parameters
    ----------
    metadata_df        : DataFrame with columns [file_name, label, feature]
                         label: 1=GOOD, 0=BAD  (as in the raw metadata)
    img_dir            : path to the image directory
    yolo_model         : YOLO model instance (or None to use default bbox)
    transform          : torchvision transform applied after cropping
    crop_size          : resize target for the cropped patch
    conf_threshold     : minimum YOLO confidence to accept a detection
    oversample_minority: duplicate BAD samples for imbalanced features
    max_ratio          : maximum GOOD:BAD ratio after oversampling
    augment            : apply data augmentation to oversampled BAD samples
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        img_dir: str | Path,
        yolo_model=None,
        transform=None,
        crop_size: int = 128,
        conf_threshold: float = 0.3,
        oversample_minority: bool = True,
        max_ratio: float = 3.0,
        augment: bool = True,
    ):
        self.img_dir          = Path(img_dir)
        self.transform        = transform
        self.crop_size        = crop_size
        self.conf_threshold   = conf_threshold
        self.yolo_model       = yolo_model
        self.augment          = augment
        self._bbox_cache: dict = {}

        self.samples: list[dict] = []
        self._build_samples(metadata_df)

        if oversample_minority:
            self._oversample(max_ratio)

        print(
            f"[Dataset] {len(self.samples)} samples total  "
            f"(GOOD={sum(s['quality']==0 for s in self.samples)}, "
            f"BAD={sum(s['quality']==1 for s in self.samples)})"
        )

    # ------------------------------------------------------------------
    def _build_samples(self, df: pd.DataFrame):
        from tqdm import tqdm
        print("[Dataset] Building samples …")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = self.img_dir / row['file_name']
            if not img_path.exists():
                continue

            raw_label   = int(row['label'])           # 1=GOOD, 0=BAD in metadata
            feature_raw = str(row['feature'])
            feature_yolo = META_TO_YOLO.get(feature_raw, 'edge1')
            template     = FEATURE_TO_TEMPLATE.get(feature_yolo, 'C')

            detections = self._detect(img_path, feature_yolo)

            for det in detections:
                self.samples.append({
                    'path'       : img_path,
                    'template_id': TEMPLATE_TO_ID[template],
                    'feature_id' : det['class_id'],
                    'bbox'       : det['bbox'],
                    # quality: 0=GOOD (in-distrib.), 1=BAD (OOD)
                    'quality'    : 1 - raw_label,
                    'augment'    : False,
                })

    # ------------------------------------------------------------------
    def _detect(self, img_path: Path, feature_yolo: str) -> list[dict]:
        """Return YOLO detections for an image, with caching."""
        key = str(img_path)
        if key in self._bbox_cache:
            return self._bbox_cache[key]

        if self.yolo_model is not None:
            try:
                results = self.yolo_model(str(img_path), conf=self.conf_threshold, verbose=False)
                boxes   = results[0].boxes
                dets = [
                    {
                        'class_id'  : int(b.cls[0]),
                        'bbox'      : b.xywhn[0].cpu().tolist(),
                        'confidence': float(b.conf[0]),
                    }
                    for b in boxes
                ]
                if dets:
                    self._bbox_cache[key] = dets
                    return dets
            except Exception:
                pass

        # Fallback: use feature-specific fixed bbox when YOLO is unavailable
        fallback = {
            'class_id'  : FEATURE_TO_ID.get(feature_yolo, 0),
            'bbox'      : [0.5, 0.5, 0.3, 0.3],
            'confidence': 1.0,
        }
        self._bbox_cache[key] = [fallback]
        return [fallback]

    # ------------------------------------------------------------------
    def _oversample(self, max_ratio: float):
        """
        Duplicate BAD samples (with augmentation flag) until the GOOD:BAD
        ratio per feature type is at most `max_ratio`.
        """
        from collections import defaultdict
        by_feature: dict[int, list] = defaultdict(list)
        for s in self.samples:
            by_feature[s['feature_id']].append(s)

        extra: list[dict] = []
        for fid, samples in by_feature.items():
            good = [s for s in samples if s['quality'] == 0]
            bad  = [s for s in samples if s['quality'] == 1]
            if not bad or not good:
                continue
            target_bad = max(len(bad), int(len(good) / max_ratio))
            deficit    = target_bad - len(bad)
            if deficit <= 0:
                continue
            rng = np.random.default_rng(42)
            for orig in rng.choice(bad, size=deficit, replace=True):
                copy = dict(orig)
                copy['augment'] = True
                extra.append(copy)

        self.samples.extend(extra)
        np.random.default_rng(0).shuffle(self.samples)

    # ------------------------------------------------------------------
    def _crop_bbox(self, image: Image.Image, bbox: list, feature_yolo: str) -> Image.Image:
        """
        Crop an image to the given normalised bbox, applying extra padding
        for narrow edge-roughness features to avoid excessive distortion.
        """
        x_c, y_c, w, h = bbox
        W, H = image.size

        padding = 0.3 if YOLO_CLASS_NAMES[bbox[0] if isinstance(bbox[0], int) else 0] in WIDE_PADDING_FEATURES else 0.2
        padding = 0.3 if feature_yolo in WIDE_PADDING_FEATURES else 0.2

        w = w * (1 + padding)
        h = h * (1 + padding)

        x1 = max(0, int((x_c - w / 2) * W))
        y1 = max(0, int((y_c - h / 2) * H))
        x2 = min(W, int((x_c + w / 2) * W))
        y2 = min(H, int((y_c + h / 2) * H))

        return image.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = Image.open(s['path']).convert('RGB')
        feature_yolo = YOLO_CLASS_NAMES[s['feature_id']]
        img = self._crop_bbox(img, s['bbox'], feature_yolo)

        # Apply augmentation to oversampled minority samples
        if s.get('augment') and self.augment:
            img = _augment_pil(img)

        if self.transform:
            img = self.transform(img)

        return {
            'image'      : img,
            'template_id': s['template_id'],
            'feature_id' : s['feature_id'],
            'quality'    : s['quality'],
            'bbox'       : torch.tensor(s['bbox'], dtype=torch.float32),
        }
