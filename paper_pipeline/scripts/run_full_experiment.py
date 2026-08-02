"""
One-button orchestrator: data prep → feature extraction (4 λ) → train/eval (4 λ).

Usage:
    python paper_pipeline/scripts/run_full_experiment.py
    python paper_pipeline/scripts/run_full_experiment.py --skip-data-prep
    python paper_pipeline/scripts/run_full_experiment.py --skip-feature-extraction
    python paper_pipeline/scripts/run_full_experiment.py --lambdas low_pass,high_pass
    python paper_pipeline/scripts/run_full_experiment.py --limit 5    # smoke test
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

DATA_PREP = [
    PROJECT_ROOT / "paper_pipeline" / "data_prep" / "01_fetch_ham10000_metadata.py",
    PROJECT_ROOT / "paper_pipeline" / "data_prep" / "02_verify_ham10000_ids.py",
    PROJECT_ROOT / "paper_pipeline" / "data_prep" / "03_filter_bcc_bkl.py",
    PROJECT_ROOT / "paper_pipeline" / "data_prep" / "04_apply_masks_to_lesions.py",
]

FEATURE_EXTRACTION = PROJECT_ROOT / "paper_pipeline" / "pipeline" / "feature_extraction.py"
TRAIN_EVAL = PROJECT_ROOT / "paper_pipeline" / "pipeline" / "train_eval.py"

LOG = logging.getLogger("orchestrator")


def run(cmd: list[str]) -> None:
    LOG.info(">>> %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed with exit code {proc.returncode}: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--lambdas", type=str, default="low_pass,high_pass,dft,odd_harmonic")
    parser.add_argument("--n-features", type=int, default=360)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test mode: extract from first N samples per class.")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    if not args.skip_data_prep:
        for script in DATA_PREP:
            run([PYTHON, str(script)])

    if not args.skip_feature_extraction:
        fe = [PYTHON, str(FEATURE_EXTRACTION),
              "--lambdas", args.lambdas]
        if args.limit:
            fe += ["--limit", str(args.limit)]
        if args.no_augment:
            fe.append("--no-augment")
        run(fe)

    te = [PYTHON, str(TRAIN_EVAL),
          "--lambdas", args.lambdas,
          "--n-features", str(args.n_features),
          "--cv-splits", str(args.cv_splits)]
    run(te)

    LOG.info("All stages complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
