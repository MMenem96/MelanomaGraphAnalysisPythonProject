#!/bin/bash
# Unattended overnight 7-class pipeline (Track B).
#
# Stage 1  extract features, lesion-grouped split + equalise augmentation  (~2.5 h)
# Stage 2  train + evaluate the 9 classifiers on that arm                  (~1-3 h)
# Stage 3  extract features, image-level split (the leaky comparison arm)  (~2.5 h)
# Stage 4  train + evaluate the 9 classifiers on the comparison arm        (~1-3 h)
#
# Stage 3/4 exist to measure how much our own published protocol inflates:
# the gap between the two arms IS the leakage estimate.
#
# Every stage is resumable. Extraction skips shards already on disk; training
# writes its results CSV after every classifier. Nothing is lost to a crash,
# a sleep, or a power cut beyond the item in flight.
#
# Usage: nohup bash paper_pipeline/scripts/run_overnight7.sh > /dev/null 2>&1 &
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PY="$PROJECT_ROOT/.venv/bin/python"
LOGDIR="$PROJECT_ROOT/paper_pipeline/output/logs"
MASTER="$LOGDIR/overnight7_master.log"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "$(date '+%F %T')  overnight  $*" | tee -a "$MASTER"; }

say "=========== OVERNIGHT RUN START ==========="

# ---- Stage 1: lesion-grouped extraction (supervised, auto-resumes) ----------
say "STAGE 1/4  extraction — lesion-grouped, equalise"
bash paper_pipeline/scripts/run_extract7_supervised.sh
if ! grep -q "Done\. Next:" "$LOGDIR/extract7_equalize_lesion.log"; then
    say "STAGE 1 FAILED — stopping. See extract7_equalize_lesion.log"
    exit 1
fi
say "STAGE 1 COMPLETE"

# ---- Stage 2: training on the correct arm ----------------------------------
say "STAGE 2/4  training — lesion_equalize (9 classifiers, grouped CV)"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py \
    --suffix lesion_equalize >> "$LOGDIR/train7_lesion_equalize.log" 2>&1
say "STAGE 2 finished (exit $?) — results in paper_pipeline/output/results7/"

# ---- Stage 3: image-level comparison arm -----------------------------------
say "STAGE 3/4  extraction — image-level split (leaky comparison arm)"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/feature_extraction7.py \
    --balance equalize --split-unit image --workers 8 --shard-size 250 \
    >> "$LOGDIR/extract7_equalize_image.log" 2>&1
say "STAGE 3 finished (exit $?)"

# ---- Stage 4: training on the comparison arm -------------------------------
if grep -q "Done\. Next:" "$LOGDIR/extract7_equalize_image.log" 2>/dev/null; then
    say "STAGE 4/4  training — image_equalize"
    caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py \
        --suffix image_equalize >> "$LOGDIR/train7_image_equalize.log" 2>&1
    say "STAGE 4 finished (exit $?)"
else
    say "STAGE 4 skipped — stage 3 did not complete"
fi

say "=========== OVERNIGHT RUN END ==========="
say "results: paper_pipeline/output/results7/*.csv"
