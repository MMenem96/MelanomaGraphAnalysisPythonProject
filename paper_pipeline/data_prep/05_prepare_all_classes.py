"""
Prepare the full 7-class HAM10000 subset (multi-class extension of 03 + 04).

This is the multi-class sibling of `03_filter_bcc_bkl.py` + `04_apply_masks_to_lesions.py`.
It does NOT touch the canonical binary artefacts used by the published BCC/BKL
protocol — those stay exactly as they are.

Reads:
    data/HAM10000_metadata.csv                       (10,015 rows, 7 dx classes)
    data/HAM10000_binary_mask/<id>_segmentation.png  (10,015 masks)
    <raw image root>/<id>.jpg                        (see RAW_IMAGE_CANDIDATES)

Writes:
    data/canonical7/samples.csv                      (image_id, lesion_id, dx, label)
    data/canonical7/segmented/<dx>/<image_id>.png    (lesion over white background)

The `lesion_id` column is the reason this script exists as a separate stage:
HAM10000 has 10,015 images but only 7,470 unique lesions, so every downstream
split must group on `lesion_id` to avoid putting two photographs of the same
physical lesion on both sides of the train/test boundary.

Usage:
    python paper_pipeline/data_prep/05_prepare_all_classes.py
    python paper_pipeline/data_prep/05_prepare_all_classes.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"
MASKS_DIR = PROJECT_ROOT / "data" / "HAM10000_binary_mask"
OUT_DIR = PROJECT_ROOT / "data" / "canonical7"
SAMPLES_CSV = OUT_DIR / "samples.csv"
SEGMENTED_DIR = OUT_DIR / "segmented"

# Raw image roots, searched in order. The first two are the binary-protocol
# folders (BCC/BKL only); the third is the full HAM10000 image release.
RAW_IMAGE_CANDIDATES = [
    # symlink → ~/Desktop/Work/Master/Images/HAM100000-Images (full HAM10000 release)
    PROJECT_ROOT / "data" / "ham10000_images",
]

MASK_SUFFIX = "_segmentation.png"
EXPECTED_TOTAL = 10015

# Fixed label order — alphabetical, so the mapping is stable and reproducible
# regardless of row order in the metadata CSV.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_OF = {dx: i for i, dx in enumerate(CLASSES)}

EXPECTED_COUNTS = {
    "nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514,
    "akiec": 327, "vasc": 142, "df": 115,
}


def find_image(image_id: str) -> Path | None:
    for root in RAW_IMAGE_CANDIDATES:
        if not root.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            p = root / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite the lesion over a WHITE background (identical to stage 04).

    Downstream feature extraction derives its lesion mask by thresholding dark
    pixels, so background must be bright; white also stops the hair detector
    from firing on a black border halo.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    if image.ndim == 3:
        binary3 = binary[:, :, None]
        white = np.full_like(image, 255)
        return (image * binary3 + white * (1 - binary3)).astype(image.dtype)
    white = np.full_like(image, 255)
    return (image * binary + white * (1 - binary)).astype(image.dtype)


def segment_one(task: tuple[str, str]) -> tuple[str, bool, str]:
    image_id, dx = task
    out_path = SEGMENTED_DIR / dx / f"{image_id}.png"
    if out_path.is_file():
        return image_id, True, "cached"

    img_path = find_image(image_id)
    if img_path is None:
        return image_id, False, "raw image not found"
    mask_path = MASKS_DIR / f"{image_id}{MASK_SUFFIX}"
    if not mask_path.is_file():
        return image_id, False, "mask not found"

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return image_id, False, f"cv2.imread returned None for {img_path}"
    if mask is None:
        return image_id, False, f"cv2.imread returned None for {mask_path}"

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), apply_mask(image, mask))
    return image_id, True, ""


def load_samples() -> list[dict[str, str]]:
    """Metadata rows that have BOTH a mask and a raw image on disk."""
    if not METADATA_CSV.is_file():
        raise FileNotFoundError(f"{METADATA_CSV} not found. Run data_prep/01 first.")

    mask_ids = {
        p.name[: -len(MASK_SUFFIX)]
        for p in MASKS_DIR.iterdir()
        if p.suffix.lower() == ".png" and p.name.endswith(MASK_SUFFIX)
    }

    rows = list(csv.DictReader(METADATA_CSV.open()))
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} metadata rows, got {len(rows)}")

    samples, missing_mask, missing_image = [], [], []
    for r in rows:
        image_id, dx = r["image_id"], r["dx"]
        if dx not in LABEL_OF:
            raise RuntimeError(f"Unknown dx {dx!r} for {image_id}")
        if image_id not in mask_ids:
            missing_mask.append(image_id)
            continue
        if find_image(image_id) is None:
            missing_image.append(image_id)
            continue
        samples.append({
            "image_id": image_id,
            "lesion_id": r["lesion_id"],
            "dx": dx,
            "label": str(LABEL_OF[dx]),
        })

    if missing_mask:
        print(f"WARNING: {len(missing_mask)} ids have no mask "
              f"(first 5: {missing_mask[:5]})", file=sys.stderr)
    if missing_image:
        print(f"WARNING: {len(missing_image)} ids have no raw image "
              f"(first 5: {missing_image[:5]})", file=sys.stderr)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for mask application.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N samples (debug).")
    args = ap.parse_args()

    samples = load_samples()
    if args.limit:
        samples = samples[: args.limit]

    counts = Counter(s["dx"] for s in samples)
    lesions = {s["lesion_id"] for s in samples}
    print(f"Usable samples: {len(samples)} images across {len(lesions)} unique lesions")
    print(f"{'class':<7} {'label':>5} {'images':>7} {'expected':>9} {'lesions':>8}")
    for dx in CLASSES:
        n_les = len({s['lesion_id'] for s in samples if s['dx'] == dx})
        exp = EXPECTED_COUNTS[dx] if not args.limit else "-"
        print(f"{dx:<7} {LABEL_OF[dx]:>5} {counts[dx]:>7} {str(exp):>9} {n_les:>8}")

    if not args.limit:
        for dx, exp in EXPECTED_COUNTS.items():
            if counts[dx] != exp:
                print(f"\nFAIL: {dx} has {counts[dx]} usable images, expected {exp}.",
                      file=sys.stderr)
                return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLES_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "lesion_id", "dx", "label"])
        w.writeheader()
        w.writerows(samples)
    print(f"\nWrote {SAMPLES_CSV.relative_to(PROJECT_ROOT)} ({len(samples)} rows)")

    print(f"Applying masks with {args.workers} workers → "
          f"{SEGMENTED_DIR.relative_to(PROJECT_ROOT)}/<dx>/")
    tasks = [(s["image_id"], s["dx"]) for s in samples]
    ok, cached, failures = 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (image_id, success, reason) in enumerate(ex.map(segment_one, tasks, chunksize=32), 1):
            if success:
                ok += 1
                if reason == "cached":
                    cached += 1
            else:
                failures.append((image_id, reason))
            if i % 1000 == 0:
                print(f"  …{i}/{len(tasks)}  ok={ok} failed={len(failures)}")

    print(f"  done: {ok}/{len(tasks)} segmented ({cached} already on disk)")
    if failures:
        print(f"  FAILED: {len(failures)} (first 10):", file=sys.stderr)
        for fid, reason in failures[:10]:
            print(f"    {fid}: {reason}", file=sys.stderr)
        return 1

    print("\nOK — 7-class dataset ready.")
    print("Next: python paper_pipeline/pipeline/feature_extraction.py --task multiclass7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
