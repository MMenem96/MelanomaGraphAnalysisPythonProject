"""
Quality-assurance helpers: save visual evidence of preprocessing and
augmentation correctness, plus a per-run manifest.

Outputs land in paper_pipeline/output/qa/. The intent is that a human can
glance at the images and confirm: hair was removed, flips are valid, masks
are sensible, etc.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_ROOT = PROJECT_ROOT / "paper_pipeline" / "output" / "qa"


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB array using cv2 (which expects BGR on disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = image_rgb
    cv2.imwrite(str(path), bgr)


def save_hair_removal_qa(
    image_id: str,
    raw_rgb: np.ndarray,
    hair_mask: np.ndarray,
    inpainted_rgb: np.ndarray,
    out_dir: Path | None = None,
) -> None:
    base = (out_dir or (QA_ROOT / "hair_removal")) / image_id
    _save_rgb(base.with_suffix("") .parent / f"{image_id}_01_raw.png", raw_rgb)
    _save_rgb(base.parent / f"{image_id}_02_hair_mask.png", hair_mask)
    _save_rgb(base.parent / f"{image_id}_03_inpainted.png", inpainted_rgb)


def save_augmentation_qa(
    image_id: str,
    orig_rgb: np.ndarray,
    hflip_rgb: np.ndarray,
    vflip_rgb: np.ndarray,
    out_dir: Path | None = None,
) -> None:
    base = (out_dir or (QA_ROOT / "augmentation"))
    base.mkdir(parents=True, exist_ok=True)
    _save_rgb(base / f"{image_id}_orig.png", orig_rgb)
    _save_rgb(base / f"{image_id}_h_flip.png", hflip_rgb)
    _save_rgb(base / f"{image_id}_v_flip.png", vflip_rgb)


def write_manifest(manifest: dict, name: str) -> Path:
    """Write a JSON manifest under output/manifests/."""
    out_dir = PROJECT_ROOT / "paper_pipeline" / "output" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{name}_{stamp}.json"
    manifest_with_meta = {
        "name": name,
        "created_at": stamp,
        "python": sys.version.split()[0],
        **manifest,
    }
    with path.open("w") as f:
        json.dump(manifest_with_meta, f, indent=2, default=str)
    return path
