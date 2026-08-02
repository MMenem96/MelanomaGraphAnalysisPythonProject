"""
Canonical dataset loader.

Returns lists of (image_path, label, image_id) for BCC and BKL. Reads the
canonical ID files produced by data_prep/03 and the segmented PNGs produced
by data_prep/04.

Labels follow the existing convention in `manual_train_features.py`:
    BCC = 1   (positive class / malignant)
    BKL = 0   (negative class / benign)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = PROJECT_ROOT / "data" / "canonical"

BCC_IDS_FILE = CANONICAL / "bcc_ids.txt"
BKL_IDS_FILE = CANONICAL / "bkl_ids.txt"
BCC_SEG_DIR = CANONICAL / "bcc_segmented"
BKL_SEG_DIR = CANONICAL / "bkl_segmented"

LABEL_BCC = 1
LABEL_BKL = 0


@dataclass
class Sample:
    image_id: str
    path: Path
    label: int


def _load_ids(ids_file: Path) -> list[str]:
    if not ids_file.is_file():
        raise FileNotFoundError(
            f"{ids_file} not found. Run paper_pipeline/data_prep/03_filter_bcc_bkl.py first."
        )
    return [line.strip() for line in ids_file.open() if line.strip()]


def _resolve(ids: list[str], folder: Path, label: int) -> list[Sample]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found. Run paper_pipeline/data_prep/04_apply_masks_to_lesions.py first."
        )
    out: list[Sample] = []
    missing: list[str] = []
    for image_id in ids:
        p = folder / f"{image_id}.png"
        if not p.is_file():
            missing.append(image_id)
            continue
        out.append(Sample(image_id=image_id, path=p, label=label))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ids from {folder.name} have no segmented PNG. "
            f"First missing: {missing[:5]}"
        )
    return out


def load_canonical_samples() -> list[Sample]:
    """Return all 514 + 1099 = 1613 canonical samples."""
    bcc_ids = _load_ids(BCC_IDS_FILE)
    bkl_ids = _load_ids(BKL_IDS_FILE)
    bcc = _resolve(bcc_ids, BCC_SEG_DIR, LABEL_BCC)
    bkl = _resolve(bkl_ids, BKL_SEG_DIR, LABEL_BKL)
    return bcc + bkl


def label_name(label: int) -> str:
    if label == LABEL_BCC:
        return "BCC"
    if label == LABEL_BKL:
        return "BKL"
    raise ValueError(f"Unknown label {label}")
