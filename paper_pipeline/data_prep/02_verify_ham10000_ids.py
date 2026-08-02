"""
Verify that the HAM10000 metadata file matches the binary-mask folder.

Reads:
    data/HAM10000_metadata.csv
    data/HAM10000_binary_mask/  (10,015 PNGs named <isic_id>_segmentation.png)

Asserts:
    * Metadata has exactly 10,015 rows.
    * Every image_id in metadata has a corresponding mask file.
    * No mask file is orphaned (no extra masks without metadata).

Usage:
    python paper_pipeline/data_prep/02_verify_ham10000_ids.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"
MASKS_DIR = PROJECT_ROOT / "data" / "HAM10000_binary_mask"
EXPECTED_TOTAL = 10015
ID_COLUMN = "image_id"
MASK_SUFFIX = "_segmentation.png"


def load_metadata_ids(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}. "
            "Run 01_fetch_ham10000_metadata.py first."
        )
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if ID_COLUMN not in reader.fieldnames:
            raise RuntimeError(
                f"Column '{ID_COLUMN}' missing in metadata CSV. "
                f"Got: {reader.fieldnames}"
            )
        return {row[ID_COLUMN] for row in reader}


def load_mask_ids(masks_dir: Path) -> set[str]:
    if not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Masks folder not found at {masks_dir}. "
            "The folder must contain <isic_id>_segmentation.png files."
        )
    out: set[str] = set()
    for p in masks_dir.iterdir():
        if p.suffix.lower() == ".png" and p.name.endswith(MASK_SUFFIX):
            out.add(p.name[: -len(MASK_SUFFIX)])
    return out


def main() -> int:
    print("Verifying HAM10000 metadata vs masks folder…")
    print(f"  metadata: {METADATA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  masks   : {MASKS_DIR.relative_to(PROJECT_ROOT)}")

    metadata_ids = load_metadata_ids(METADATA_CSV)
    mask_ids = load_mask_ids(MASKS_DIR)

    print(f"\n  metadata rows = {len(metadata_ids):,}")
    print(f"  mask files    = {len(mask_ids):,}")

    ok = True

    if len(metadata_ids) != EXPECTED_TOTAL:
        print(
            f"\nFAIL: metadata has {len(metadata_ids)} rows, expected {EXPECTED_TOTAL}.",
            file=sys.stderr,
        )
        ok = False

    if len(mask_ids) != EXPECTED_TOTAL:
        print(
            f"\nWARN: masks folder has {len(mask_ids)} files, expected {EXPECTED_TOTAL}.",
            file=sys.stderr,
        )

    missing_masks = metadata_ids - mask_ids
    extra_masks = mask_ids - metadata_ids

    if missing_masks:
        print(f"\nFAIL: {len(missing_masks)} metadata IDs have no mask file.", file=sys.stderr)
        for sample in sorted(missing_masks)[:10]:
            print(f"    missing: {sample}", file=sys.stderr)
        ok = False

    if extra_masks:
        print(f"\nNOTE: {len(extra_masks)} mask files are not in metadata (will be ignored):")
        for sample in sorted(extra_masks)[:10]:
            print(f"    extra: {sample}")

    if ok and not missing_masks:
        print("\nOK — metadata and masks are consistent.")
        return 0

    print("\nFix the inconsistencies above before proceeding to step 03.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
