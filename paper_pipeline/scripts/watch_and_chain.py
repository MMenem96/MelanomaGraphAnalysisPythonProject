"""
Watch for the running feature_extraction job, then auto-chain training and
post-processing once it finishes.

Behaviour:
    1. Wait until the feature_extraction PID is no longer alive.
    2. Verify all 4 train + 4 test pickles exist.
    3. Run train_eval.py for all 4 λ.
    4. Run generate_figures.py (best classifier per λ).
    5. Run results_to_paper.py (LaTeX tables).
    6. Write paper_pipeline/output/SUMMARY.md.

Usage:
    nohup .venv/bin/python paper_pipeline/scripts/watch_and_chain.py \\
        > paper_pipeline/output/logs/watch.log 2>&1 &
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PID_FILE = PROJECT_ROOT / "paper_pipeline" / "output" / "logs" / "full_features.pid"
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
LOGS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "logs"

LAMBDAS = ["low_pass", "high_pass", "dft", "odd_harmonic"]

PYTHON = sys.executable
TRAIN_EVAL = PROJECT_ROOT / "paper_pipeline" / "pipeline" / "train_eval.py"
GEN_FIG = PROJECT_ROOT / "paper_pipeline" / "scripts" / "generate_figures.py"
RES_TO_TEX = PROJECT_ROOT / "paper_pipeline" / "scripts" / "results_to_paper.py"
FAMILY = PROJECT_ROOT / "paper_pipeline" / "scripts" / "feature_family_analysis.py"

LOG = logging.getLogger("watch_and_chain")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # process exists, just can't signal


def wait_for_features() -> bool:
    if not PID_FILE.is_file():
        LOG.error("PID file not found at %s — is feature_extraction running?", PID_FILE)
        return False
    pid = int(PID_FILE.read_text().strip())
    LOG.info("Watching feature_extraction (PID %d)…", pid)
    while pid_alive(pid):
        time.sleep(60)
        LOG.info("  still running…")
    LOG.info("Feature extraction process %d has exited.", pid)
    return True


def verify_pickles() -> bool:
    missing = []
    for lam in LAMBDAS:
        for split in ("train", "test"):
            p = FEATURES_DIR / f"features_{split}_{lam}.pkl"
            if not p.is_file():
                missing.append(p.name)
    if missing:
        LOG.error("Expected pickles missing: %s", missing)
        return False
    LOG.info("All 8 pickles present.")
    return True


def run(cmd: list[str], log_name: str) -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / log_name
    LOG.info(">>> %s    (log: %s)", " ".join(str(c) for c in cmd), log_path.name)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        LOG.error("Command failed (rc=%d): %s", proc.returncode, " ".join(str(c) for c in cmd))
    return proc.returncode


def latest(pattern: str, root: Path) -> Path | None:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def write_summary() -> None:
    import pandas as pd

    summary_csv = latest("summary_*.csv", RESULTS_DIR)
    if summary_csv is None:
        LOG.warning("No summary CSV found — cannot write SUMMARY.md")
        return

    df = pd.read_csv(summary_csv)
    lines = [
        "# Paper-pipeline run summary",
        "",
        f"- Summary CSV: `{summary_csv.relative_to(PROJECT_ROOT)}`",
        f"- Pickles: `paper_pipeline/output/features/`",
        f"- Per-classifier predictions: `paper_pipeline/output/predictions/`",
        f"- Figures: `paper_pipeline/output/figures/`",
        f"- LaTeX tables: `paper_pipeline/output/tables/`",
        "",
        "## Best classifier per λ (by test accuracy)",
        "",
        "| λ | Classifier | Acc | Sens | Spec | Prec | F1 | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lam, sub in df.groupby("lambda"):
        best = sub.loc[sub["test_accuracy"].idxmax()]
        lines.append(
            f"| {lam} | {best['classifier']} | "
            f"{100*best['test_accuracy']:.1f} | {100*best['test_sensitivity']:.1f} | "
            f"{100*best['test_specificity']:.1f} | {100*best['test_precision']:.1f} | "
            f"{100*best['test_f1']:.1f} | {100*best['test_auc']:.1f} |"
        )

    lines += [
        "",
        "## All results (one row per classifier × λ)",
        "",
        "See the CSV for full numbers. Top-5 by test accuracy:",
        "",
    ]
    top5 = df.nlargest(5, "test_accuracy")[
        ["lambda", "classifier", "test_accuracy", "test_sensitivity",
         "test_specificity", "test_auc", "cv_accuracy_mean", "cv_accuracy_std"]
    ]
    lines.append("| λ | Classifier | Test Acc | Sens | Spec | AUC | CV Acc (mean ± std) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in top5.iterrows():
        lines.append(
            f"| {r['lambda']} | {r['classifier']} | "
            f"{100*r['test_accuracy']:.1f} | {100*r['test_sensitivity']:.1f} | "
            f"{100*r['test_specificity']:.1f} | {100*r['test_auc']:.1f} | "
            f"{100*r['cv_accuracy_mean']:.1f} ± {100*r['cv_accuracy_std']:.1f} |"
        )

    out = PROJECT_ROOT / "paper_pipeline" / "output" / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    LOG.info("Wrote %s", out.relative_to(PROJECT_ROOT))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    if not wait_for_features():
        return 1
    if not verify_pickles():
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rc = run([PYTHON, str(TRAIN_EVAL)], f"train_eval_{timestamp}.log")
    if rc != 0:
        return rc

    run([PYTHON, str(GEN_FIG)], f"generate_figures_{timestamp}.log")
    run([PYTHON, str(RES_TO_TEX)], f"results_to_paper_{timestamp}.log")
    run([PYTHON, str(FAMILY)], f"feature_family_{timestamp}.log")
    write_summary()

    LOG.info("Chain complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
