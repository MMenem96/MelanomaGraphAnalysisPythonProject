"""
Apply binary masks to BCC and BKL images and write segmented lesions to disk.

Reads:
    data/canonical/bcc_ids.txt           (514 lines)
    data/canonical/bkl_ids.txt           (1099 lines)
    data/HAM10000_binary_mask/<id>_segmentation.png
    data/bcc/<id>.jpg  OR  data/sk/<id>.jpg  (raw images)

Writes:
    data/canonical/bcc_segmented/<id>.png  (514 files)
    data/canonical/bkl_segmented/<id>.png  (1099 files)

The output image keeps lesion pixels and zeros the background (binary mask
multiplication). Subsequent feature extraction operates on these images.

Usage:
    python paper_pipeline/data_prep/04_apply_masks_to_lesions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
BCC_IDS_FILE = CANONICAL_DIR / "bcc_ids.txt"
BKL_IDS_FILE = CANONICAL_DIR / "bkl_ids.txt"
BCC_OUT_DIR = CANONICAL_DIR / "bcc_segmented"
BKL_OUT_DIR = CANONICAL_DIR / "bkl_segmented"
MASKS_DIR = PROJECT_ROOT / "data" / "HAM10000_binary_mask"

# Source folders to search for raw images. The script tries them in order.
RAW_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "data" / "bcc",
    PROJECT_ROOT / "data" / "sk",
]

EXPECTED_BCC = 514
EXPECTED_BKL = 1099


def find_image(image_id: str) -> Path | None:
    for root in RAW_IMAGE_CANDIDATES:
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            p = root / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def find_mask(image_id: str) -> Path | None:
    p = MASKS_DIR / f"{image_id}_segmentation.png"
    return p if p.is_file() else None


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Composite the lesion over a WHITE background.

    The downstream feature-extraction pipeline derives a lesion mask by
    thresholding dark pixels (`generate_lesion_mask_from_transparent_background`
    expects bright pixels = background, dark pixels = lesion). Using a white
    background also prevents the hair detector from triggering on a black
    border halo.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    if image.ndim == 3:
        binary3 = binary[:, :, None]
        white = np.full_like(image, 255)
        return (image * binary3 + white * (1 - binary3)).astype(image.dtype)
    # grayscale path
    white = np.full_like(image, 255)
    return (image * binary + white * (1 - binary)).astype(image.dtype)


def process_one(image_id: str, out_dir: Path) -> tuple[bool, str]:
    img_path = find_image(image_id)
    if img_path is None:
        return False, f"image not found in any raw folder"
    mask_path = find_mask(image_id)
    if mask_path is None:
        return False, f"mask not found"

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, f"cv2.imread returned None for image {img_path}"
    if mask is None:
        return False, f"cv2.imread returned None for mask {mask_path}"

    # Resize mask to match image size if needed (rare, but masks may be downsampled).
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    segmented = apply_mask(image, mask)
    out_path = out_dir / f"{image_id}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), segmented)
    return True, ""


def process_class(ids_file: Path, out_dir: Path, expected: int, name: str) -> int:
    if not ids_file.is_file():
        print(f"ERROR: {ids_file} not found. Run 03 first.", file=sys.stderr)
        return 2
    ids = [line.strip() for line in ids_file.open() if line.strip()]
    print(f"\nProcessing {name}: {len(ids)} ids → {out_dir.relative_to(PROJECT_ROOT)}")

    ok = 0
    failures: list[tuple[str, str]] = []
    for image_id in ids:
        success, reason = process_one(image_id, out_dir)
        if success:
            ok += 1
        else:
            failures.append((image_id, reason))
        if ok and ok % 200 == 0:
            print(f"  …{ok}/{len(ids)}")

    print(f"  done: {ok}/{len(ids)} segmented")
    if failures:
        print(f"  failed: {len(failures)} (first 10):")
        for fid, reason in failures[:10]:
            print(f"    {fid}: {reason}")

    if ok != expected:
        print(
            f"\nFAIL: produced {ok} {name} files, expected {expected}.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    print("Applying binary masks to canonical BCC and BKL images…")
    rc1 = process_class(BCC_IDS_FILE, BCC_OUT_DIR, EXPECTED_BCC, "BCC")
    rc2 = process_class(BKL_IDS_FILE, BKL_OUT_DIR, EXPECTED_BKL, "BKL")
    if rc1 or rc2:
        return rc1 or rc2

    print("\nOK — canonical segmented lesions written.")
    print(f"  {BCC_OUT_DIR.relative_to(PROJECT_ROOT)}  ({EXPECTED_BCC} files)")
    print(f"  {BKL_OUT_DIR.relative_to(PROJECT_ROOT)}  ({EXPECTED_BKL} files)")
    print("\nNext step: python paper_pipeline/scripts/run_full_experiment.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
