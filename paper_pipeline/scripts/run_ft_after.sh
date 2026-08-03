#!/bin/bash
# Train on the fine-tuned-backbone features once the metadata run frees the CPU.
set -u
PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PR" || exit 1
PY="$PR/.venv/bin/python"; LOG="$PR/paper_pipeline/output/logs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
say(){ echo "$(date '+%F %T')  ft-chain  $*" | tee -a "$LOG/overnight7_master.log"; }
# marker-file wait (never grep for a process name from a command containing it)
while [[ -f "$PR/.metadata_run_active" ]]; do sleep 60; done
say "starting FT training (fine-tuned EfficientNet + handcrafted + MKT + metadata)"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py --suffix lesion_equalize_ft_meta >> "$LOG/train7_ft_meta.log" 2>&1
say "FT+meta done (exit $?)"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py --suffix lesion_equalize_ft >> "$LOG/train7_ft.log" 2>&1
say "FT (no metadata) done (exit $?) — ablation pair complete"
