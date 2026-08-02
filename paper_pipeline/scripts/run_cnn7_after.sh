#!/bin/bash
# Stage 5-6: ResNet-50 features + hybrid (2,509-feature) training for Track B.
#
# Chained rather than folded into run_overnight7.sh because that script is
# already executing — bash reads a running script incrementally, so editing it
# mid-run can corrupt execution.
#
# Waits for the overnight run to finish, then:
#   5. cnn_features7   → 2,048 ResNet-50 features aligned to the handcrafted rows
#   6. train_eval7     → the 9 classifiers on all 2,509 features
#
# Both arms (lesion_equalize and, if it exists, image_equalize) are processed.
#
# Usage: nohup bash paper_pipeline/scripts/run_cnn7_after.sh > /dev/null 2>&1 &
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PY="$PROJECT_ROOT/.venv/bin/python"
LOGDIR="$PROJECT_ROOT/paper_pipeline/output/logs"
MASTER="$LOGDIR/overnight7_master.log"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2

say() { echo "$(date '+%F %T')  cnn-chain  $*" | tee -a "$MASTER"; }

say "waiting for the overnight run to finish before starting CNN stages"
while pgrep -f "run_overnight7.sh" > /dev/null; do
    sleep 120
done
say "overnight run finished — starting CNN stages"

for SUFFIX in lesion_equalize image_equalize; do
    HC="$PROJECT_ROOT/paper_pipeline/output/features7/features_train_odd_harmonic_${SUFFIX}.pkl"
    if [[ ! -f "$HC" ]]; then
        say "skip $SUFFIX — no handcrafted features (arm did not complete)"
        continue
    fi

    say "STAGE 5  ResNet-50 features — $SUFFIX"
    caffeinate -dimsu "$PY" paper_pipeline/pipeline/cnn_features7.py \
        --suffix "$SUFFIX" >> "$LOGDIR/cnn7_${SUFFIX}.log" 2>&1
    rc=$?
    say "STAGE 5 finished for $SUFFIX (exit $rc)"

    if [[ -f "$PROJECT_ROOT/paper_pipeline/output/features7/features_train_odd_harmonic_${SUFFIX}_hybrid.pkl" ]]; then
        say "STAGE 6  training on 2,509 hybrid features — ${SUFFIX}_hybrid"
        caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py \
            --suffix "${SUFFIX}_hybrid" >> "$LOGDIR/train7_${SUFFIX}_hybrid.log" 2>&1
        say "STAGE 6 finished for ${SUFFIX}_hybrid (exit $?)"
    else
        say "STAGE 6 skipped for $SUFFIX — hybrid pickles missing"
    fi
done

say "=========== CNN CHAIN COMPLETE ==========="
say "compare: results7_${SUFFIX}_*.csv (461 feats) vs results7_*_hybrid_*.csv (2509 feats)"
