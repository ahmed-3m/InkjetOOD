"""
download_weights.py
===================
Download pretrained model weights from HuggingFace Hub.

This is the first step for reproducing results without training from scratch.

Downloads to models/ directory:
  models/yolo_best.pt       — YOLOv8 feature detector (mAP@50 = 95.0%)
  models/cdm_baseline.pt    — CDM λ=0 baseline  (5-fold CV AUROC 0.867 ± 0.023)
  models/cdm_proposed.pt    — CDM λ=0.01 + YOLO  (single-split AUROC 0.860)

Usage:
  python download_weights.py
  python download_weights.py --token YOUR_HF_TOKEN   # if repo requires auth
  python download_weights.py --also-data             # also download dataset metadata
"""

import argparse
import shutil
import sys
from pathlib import Path

HF_REPO     = "ahmed-3m/InkjetOOD"
MODELS_DIR  = Path(__file__).resolve().parent / "models"
DATA_DIR    = Path(__file__).resolve().parent / "data" / "inkjet_dataset"

# Maps local filename → path inside the HF repo
MODEL_FILES = {
    "yolo_best.pt":      "models/yolo_best.pt",
    "cdm_baseline.pt":   "models/cdm_v3_baseline.pt",
    "cdm_proposed.pt":   "models/cdm_v3_yolo_bbox.pt",
}

DATA_FILES = {
    "metadata.csv":       "data/metadata/feature_cls.csv",
}


def _hf_download(repo_id: str, filename: str, token=None) -> Path:
    """Download a file from HuggingFace Hub and return its cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token))


def download_models(token=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading model weights from https://huggingface.co/{HF_REPO}")
    print(f"Target directory: {MODELS_DIR}\n")

    for local_name, hf_path in MODEL_FILES.items():
        local_path = MODELS_DIR / local_name
        if local_path.exists():
            size_mb = local_path.stat().st_size / 1e6
            print(f"  ✓ Already present: {local_name}  ({size_mb:.1f} MB)")
            continue
        print(f"  Downloading  {hf_path}  →  {local_name} ...")
        cached = _hf_download(HF_REPO, hf_path, token=token)
        shutil.copy(cached, local_path)
        size_mb = local_path.stat().st_size / 1e6
        print(f"  ✓ {local_name}  ({size_mb:.1f} MB)")


def download_data(token=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading dataset metadata from https://huggingface.co/{HF_REPO}")
    print(f"Target directory: {DATA_DIR}\n")

    for local_name, hf_path in DATA_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            print(f"  ✓ Already present: {local_name}")
            continue
        print(f"  Downloading  {hf_path}  →  {local_name} ...")
        cached = _hf_download(HF_REPO, hf_path, token=token)
        shutil.copy(cached, local_path)
        print(f"  ✓ {local_name}")


def main():
    p = argparse.ArgumentParser(
        description='Download pretrained InkjetOOD weights from HuggingFace'
    )
    p.add_argument(
        '--token', type=str, default=None,
        help='HuggingFace token (only needed if the repo is private)'
    )
    p.add_argument(
        '--also-data', action='store_true',
        help='Also download dataset metadata (metadata.csv)'
    )
    args = p.parse_args()

    download_models(token=args.token)

    if args.also_data:
        download_data(token=args.token)

    print("\n" + "=" * 60)
    print("  Download complete.")
    print("=" * 60)
    print(f"\n  Weights saved to: {MODELS_DIR}")
    print("\n  Next steps:")
    print("    • Evaluate pretrained model:")
    print("        export INKJET_DATA_DIR=/path/to/FTI_Zer0P_dataset")
    print("        python evaluate.py --checkpoint models/cdm_proposed.pt --num_trials 100")
    print()
    print("    • Or run 5-fold cross-validation (trains from scratch):")
    print("        python run_cv.py --sep_loss_weight 0.0 --out_dir results/cv_lambda0")
    print()


if __name__ == "__main__":
    main()
