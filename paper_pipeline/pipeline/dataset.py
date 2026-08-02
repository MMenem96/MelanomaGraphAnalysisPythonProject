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
    # Populated only by the 7-class loader; the binary path leaves them None so
    # existing callers are unaffected.
    lesion_id: str | None = None
    dx: str | None = None


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


# ---------------------------------------------------------------------------
# 7-class (multiclass) loader — Track B. Additive: nothing above is affected.
# ---------------------------------------------------------------------------

CANONICAL7 = PROJECT_ROOT / "data" / "canonical7"
SAMPLES7_CSV = CANONICAL7 / "samples.csv"
SEGMENTED7_DIR = CANONICAL7 / "segmented"

# Alphabetical, matching data_prep/05_prepare_all_classes.py. Do not reorder —
# label integers are persisted in samples.csv and in the feature pickles.
CLASSES7 = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL7_OF = {dx: i for i, dx in enumerate(CLASSES7)}


def load_multiclass7_samples() -> list[Sample]:
    """Return all 10,015 HAM10000 samples with `lesion_id` and `dx` populated.

    `lesion_id` is what makes grouped splitting possible: HAM10000 has 10,015
    images but only 7,470 unique lesions, so an image-level split leaks repeat
    photographs of the same lesion across the train/test boundary.
    """
    import csv

    if not SAMPLES7_CSV.is_file():
        raise FileNotFoundError(
            f"{SAMPLES7_CSV} not found. "
            "Run paper_pipeline/data_prep/05_prepare_all_classes.py first."
        )

    out: list[Sample] = []
    missing: list[str] = []
    with SAMPLES7_CSV.open() as f:
        for row in csv.DictReader(f):
            p = SEGMENTED7_DIR / row["dx"] / f"{row['image_id']}.png"
            if not p.is_file():
                missing.append(row["image_id"])
                continue
            out.append(Sample(
                image_id=row["image_id"],
                path=p,
                label=int(row["label"]),
                lesion_id=row["lesion_id"],
                dx=row["dx"],
            ))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ids in samples.csv have no segmented PNG. "
            f"First missing: {missing[:5]}"
        )
    return out


def label7_name(label: int) -> str:
    return CLASSES7[label]
