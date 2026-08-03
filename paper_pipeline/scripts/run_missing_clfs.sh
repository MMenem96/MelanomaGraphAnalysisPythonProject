#!/bin/bash
# CatBoost and Deep DNN failed on the first 7-class run (binary-only configs).
# classifiers7.py fixes both; this backfills them into the 461-feature arms so
# every arm reports the paper's full set of 9 classifiers.
# Runs last, after the CNN chain, so it never competes for CPU.
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
PY="$PROJECT_ROOT/.venv/bin/python"
LOGDIR="$PROJECT_ROOT/paper_pipeline/output/logs"
MASTER="$LOGDIR/overnight7_master.log"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
say() { echo "$(date '+%F %T')  backfill  $*" | tee -a "$MASTER"; }

while pgrep -f "run_cnn7_after.sh" > /dev/null; do sleep 120; done
say "CNN chain done — backfilling CatBoost + Deep DNN"
for SUFFIX in lesion_equalize image_equalize; do
    [[ -f "$PROJECT_ROOT/paper_pipeline/output/features7/features_train_odd_harmonic_${SUFFIX}.pkl" ]] || continue
    say "backfill $SUFFIX"
    caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py --suffix "$SUFFIX" \
        --classifiers "CatBoost,Deep DNN" >> "$LOGDIR/train7_${SUFFIX}_backfill.log" 2>&1
    say "backfill $SUFFIX finished (exit $?)"
done
say "=========== BACKFILL COMPLETE ==========="
