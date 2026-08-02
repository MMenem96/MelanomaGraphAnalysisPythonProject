"""
Filter HAM10000 to the canonical BCC and BKL subsets used in the paper.

Reads:
    data/HAM10000_metadata.csv
    data/HAM10000_binary_mask/  (10,015 PNGs)

Writes:
    data/canonical/bcc_ids.txt   (514 lines)
    data/canonical/bkl_ids.txt   (1099 lines)

Filtering rule:
    BCC = {row.image_id : row.dx == 'bcc'} intersect {mask folder ids}
    BKL = {row.image_id : row.dx == 'bkl'} intersect {mask folder ids}

The intersection guarantees every selected image has a segmentation mask
available, and the dx-column filter pins us to the frozen 2018 release.

Usage:
    python paper_pipeline/data_prep/03_filter_bcc_bkl.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"
MASKS_DIR = PROJECT_ROOT / "data" / "HAM10000_binary_mask"
CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
BCC_IDS_FILE = CANONICAL_DIR / "bcc_ids.txt"
BKL_IDS_FILE = CANONICAL_DIR / "bkl_ids.txt"

ID_COLUMN = "image_id"
DX_COLUMN = "dx"
BCC_LABEL = "bcc"
BKL_LABEL = "bkl"
EXPECTED_BCC = 514
EXPECTED_BKL = 1099
MASK_SUFFIX = "_segmentation.png"


def load_mask_ids(masks_dir: Path) -> set[str]:
    out: set[str] = set()
    for p in masks_dir.iterdir():
        if p.suffix.lower() == ".png" and p.name.endswith(MASK_SUFFIX):
            out.add(p.name[: -len(MASK_SUFFIX)])
    return out


def load_metadata_by_dx(csv_path: Path) -> dict[str, list[str]]:
    by_dx: dict[str, list[str]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if ID_COLUMN not in reader.fieldnames or DX_COLUMN not in reader.fieldnames:
            raise RuntimeError(
                f"Metadata CSV is missing required columns. "
                f"Need '{ID_COLUMN}' and '{DX_COLUMN}'. Got: {reader.fieldnames}"
            )
        for row in reader:
            by_dx.setdefault(row[DX_COLUMN], []).append(row[ID_COLUMN])
    return by_dx


def write_ids(ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for image_id in ids:
            f.write(image_id + "\n")


def main() -> int:
    if not METADATA_CSV.is_file():
        print(f"ERROR: {METADATA_CSV} not found. Run 01 first.", file=sys.stderr)
        return 2
    if not MASKS_DIR.is_dir():
        print(f"ERROR: {MASKS_DIR} not found.", file=sys.stderr)
        return 2

    print("Filtering HAM10000 to canonical BCC and BKL…")
    mask_ids = load_mask_ids(MASKS_DIR)
    print(f"  masks available     = {len(mask_ids):,}")

    by_dx = load_metadata_by_dx(METADATA_CSV)
    print(f"  diagnoses present   = {sorted(by_dx.keys())}")

    bcc_in_metadata = set(by_dx.get(BCC_LABEL, []))
    bkl_in_metadata = set(by_dx.get(BKL_LABEL, []))
    print(f"  bcc rows in metadata = {len(bcc_in_metadata):,}")
    print(f"  bkl rows in metadata = {len(bkl_in_metadata):,}")

    bcc_ids = sorted(bcc_in_metadata & mask_ids)
    bkl_ids = sorted(bkl_in_metadata & mask_ids)
    print(f"\n  BCC ∩ masks = {len(bcc_ids):,}   (expected {EXPECTED_BCC})")
    print(f"  BKL ∩ masks = {len(bkl_ids):,}   (expected {EXPECTED_BKL})")

    ok = True
    if len(bcc_ids) != EXPECTED_BCC:
        print(
            f"\nFAIL: BCC count {len(bcc_ids)} != expected {EXPECTED_BCC}.",
            file=sys.stderr,
        )
        ok = False
    if len(bkl_ids) != EXPECTED_BKL:
        print(
            f"\nFAIL: BKL count {len(bkl_ids)} != expected {EXPECTED_BKL}.",
            file=sys.stderr,
        )
        ok = False
    if not ok:
        print(
            "\nCounts must match the original 2018 HAM10000 release. "
            "Either the metadata file or the masks folder is the wrong version.",
            file=sys.stderr,
        )
        return 1

    write_ids(bcc_ids, BCC_IDS_FILE)
    write_ids(bkl_ids, BKL_IDS_FILE)
    print(f"\nWrote {BCC_IDS_FILE.relative_to(PROJECT_ROOT)}  ({len(bcc_ids)} ids)")
    print(f"Wrote {BKL_IDS_FILE.relative_to(PROJECT_ROOT)}  ({len(bkl_ids)} ids)")

    print("\nNext step: python paper_pipeline/data_prep/04_apply_masks_to_lesions.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
