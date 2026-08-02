#!/bin/bash
# Supervised 7-class feature extraction.
#
# Two failure modes have already cost us time on this run:
#   1. Workers killed under memory pressure -> ProcessPoolExecutor deadlocks and
#      the parent waits forever with no error and no exit.
#   2. The machine sleeps (battery critical) -> everything suspends.
#
# Shard checkpointing means a restart is cheap, so this wrapper simply watches
# the log: if nothing has been written for STALL_SECS, it kills the run and
# relaunches. The relaunch skips every shard already on disk, so at most one
# shard (<=250 images) is redone.
#
# Usage: bash paper_pipeline/scripts/run_extract7_supervised.sh
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PY="$PROJECT_ROOT/.venv/bin/python"
LOG="$PROJECT_ROOT/paper_pipeline/output/logs/extract7_equalize_lesion.log"
SUPLOG="$PROJECT_ROOT/paper_pipeline/output/logs/extract7_supervisor.log"
STALL_SECS=${STALL_SECS:-1800}     # 30 min of silence = stalled
MAX_RESTARTS=${MAX_RESTARTS:-30}
WORKERS=${WORKERS:-8}
SHARD=${SHARD:-250}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say() { echo "$(date '+%F %T')  supervisor  $*" | tee -a "$SUPLOG"; }

for ((attempt = 1; attempt <= MAX_RESTARTS; attempt++)); do
    say "attempt $attempt/$MAX_RESTARTS — launching (workers=$WORKERS, shard=$SHARD)"
    caffeinate -dimsu "$PY" paper_pipeline/pipeline/feature_extraction7.py \
        --balance equalize --split-unit lesion \
        --workers "$WORKERS" --shard-size "$SHARD" >> "$LOG" 2>&1 &
    RUN_PID=$!

    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep 60
        if [[ -f "$LOG" ]]; then
            last_write=$(stat -f %m "$LOG")
            idle=$(( $(date +%s) - last_write ))
            if (( idle > STALL_SECS )); then
                say "STALLED — no log output for ${idle}s, killing and resuming"
                kill -9 "$RUN_PID" 2>/dev/null
                pkill -9 -f feature_extraction7 2>/dev/null
                sleep 5
                break
            fi
        fi
    done

    wait "$RUN_PID" 2>/dev/null
    if grep -q "Done\. Next:" "$LOG"; then
        say "EXTRACTION COMPLETE"
        exit 0
    fi
    if grep -qE "Traceback|IMAGE LEAKAGE|LESION LEAKAGE" "$LOG"; then
        say "hard error in log — stopping, needs a human"
        exit 1
    fi
    say "run ended without completing; resuming from shards on disk"
    sleep 5
done

say "gave up after $MAX_RESTARTS attempts"
exit 1
