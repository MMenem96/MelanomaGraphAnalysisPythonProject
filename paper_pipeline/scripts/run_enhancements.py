"""
Chain the enhancement scripts after the base experiment.

Order:
    1. Feature importance + MI top-30           (~10 min)
    2. Threshold tuning + stacking ensemble     (~30 min)
    3. n_features sweep                         (~15-20 min)
    4. Hyperparameter tuning (best 2 λ, 4 clfs) (~30-60 min)

If a step fails, we log and continue. Final SUMMARY of best
methods written to paper_pipeline/output/ENHANCED_RESULTS.md.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
LOGS = PROJECT_ROOT / "paper_pipeline" / "output" / "logs"

STEPS = [
    ("feature_importance",   PROJECT_ROOT / "paper_pipeline" / "scripts" / "feature_importance.py", []),
    ("threshold_and_stacking", PROJECT_ROOT / "paper_pipeline" / "scripts" / "threshold_and_stacking.py", []),
    ("n_features_sweep",     PROJECT_ROOT / "paper_pipeline" / "scripts" / "n_features_sweep.py", []),
    ("hyperparameter_tuning", PROJECT_ROOT / "paper_pipeline" / "scripts" / "hyperparameter_tuning.py", []),
]

LOG = logging.getLogger("enhancements")


def run(name: str, script: Path, extra_args: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [PYTHON, str(script)] + extra_args
    LOG.info(">>> %s    log: %s", " ".join(cmd), log.name)
    t0 = time.time()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        LOG.error("    failed (rc=%d) in %.1fs", proc.returncode, elapsed)
    else:
        LOG.info("    done in %.1fs", elapsed)
    return proc.returncode


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    LOG.info("Starting enhancement chain.")
    for name, script, args in STEPS:
        rc = run(name, script, args)
        if rc != 0:
            LOG.warning("Step %s failed but continuing.", name)
    LOG.info("Enhancement chain finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
