"""
Fetch the original HAM10000 metadata file from Harvard Dataverse.

Outputs:
    data/HAM10000_metadata.csv   (10,015 rows, columns:
                                  lesion_id, image_id, dx, dx_type, age, sex, localization)

The Dataverse dataset is the frozen 2018 release. Citing it pins your work to a
specific, reproducible snapshot — anyone re-running can fetch the same file.

Usage:
    python paper_pipeline/data_prep/01_fetch_ham10000_metadata.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"

# Harvard Dataverse: doi:10.7910/DVN/DBW86T contains the HAM10000 metadata
# file. The Dataverse Search API resolves the file by name so we don't have
# to hard-code a fileId that could rotate on the server.
DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
EXPECTED_ROW_COUNT = 10015
DX_COLUMN = "dx"
ID_COLUMN = "image_id"


def find_metadata_file_id() -> int:
    """Ask the Dataverse API for the file id of HAM10000_metadata."""
    url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    files = payload["data"]["latestVersion"]["files"]
    for f in files:
        label = f.get("dataFile", {}).get("filename", "")
        if "metadata" in label.lower():
            return int(f["dataFile"]["id"])
    raise RuntimeError(
        "Could not find a metadata file in the Dataverse dataset. "
        "Available files: " + ", ".join(f["dataFile"]["filename"] for f in files)
    )


def download_metadata(file_id: int) -> bytes:
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def parse_and_normalise(raw: bytes) -> list[dict]:
    """Parse the Dataverse .tab payload into row dicts. Handles both TSV and CSV."""
    text = raw.decode("utf-8", errors="replace")
    # Dataverse serves .tab as tab-separated; some mirrors return CSV. Sniff.
    sample = text[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = list(reader)
    if not rows:
        raise RuntimeError("Downloaded metadata file is empty.")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print(f"Fetching HAM10000 metadata from Harvard Dataverse ({DATAVERSE_DOI})…")
    file_id = find_metadata_file_id()
    print(f"  file id = {file_id}")
    raw = download_metadata(file_id)
    md5 = hashlib.md5(raw).hexdigest()
    print(f"  bytes downloaded = {len(raw):,}    md5 = {md5}")

    rows = parse_and_normalise(raw)
    print(f"  parsed rows = {len(rows):,}")
    print(f"  columns     = {list(rows[0].keys())}")

    if len(rows) != EXPECTED_ROW_COUNT:
        print(
            f"\nERROR: expected {EXPECTED_ROW_COUNT} rows, got {len(rows)}. "
            "Aborting — the Dataverse may have updated the file. "
            "Verify the dataset version before proceeding.",
            file=sys.stderr,
        )
        return 2

    if DX_COLUMN not in rows[0] or ID_COLUMN not in rows[0]:
        print(
            f"ERROR: expected columns '{DX_COLUMN}' and '{ID_COLUMN}' missing. "
            f"Got: {list(rows[0].keys())}",
            file=sys.stderr,
        )
        return 3

    write_csv(rows, OUT_CSV)
    print(f"\nSaved: {OUT_CSV.relative_to(PROJECT_ROOT)}")

    # quick class-balance report
    from collections import Counter
    counts = Counter(r[DX_COLUMN] for r in rows)
    print("\nDiagnosis breakdown (dx column):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:>5}  {k}")
    print(f"\n  bcc = {counts.get('bcc', 0)}  (paper expects 514)")
    print(f"  bkl = {counts.get('bkl', 0)}  (paper expects 1099)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
