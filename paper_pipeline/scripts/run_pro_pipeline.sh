#!/bin/bash
# "Do it properly" chain: aggressive fine-tune -> combine every feature family ->
# train -> k sweep. Waits for the current evaluation to finish first.
set -u
PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PR" || exit 1
PY="$PR/.venv/bin/python"; L="$PR/paper_pipeline/output/logs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
say(){ echo "$(date '+%F %T')  pro  $*" | tee -a "$L/overnight7_master.log"; }

say "waiting for the running FT evaluations to finish"
while [[ -n "$(ls "$PR"/.ft_chain_active 2>/dev/null)" ]] || pgrep -f "run_ft_after" >/dev/null; do sleep 60; done

say "STEP 1  aggressive fine-tune (full unfreeze, 45 epochs, balanced-acc early stop)"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/cnn_finetune7.py \
    --suffix lesion_equalize --backbone efficientnetb0 --input-size 224 \
    >> "$L/finetune7_v2.log" 2>&1
say "STEP 1 done (exit $?)"

say "STEP 2  combine all feature families (461 + 2048 + 1280 + 19)"
"$PY" paper_pipeline/pipeline/combine7.py --suffix lesion_equalize >> "$L/combine7.log" 2>&1
say "STEP 2 done (exit $?)"

say "STEP 3  train on the combined representation"
caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py \
    --suffix lesion_equalize_all >> "$L/train7_all.log" 2>&1
say "STEP 3 done (exit $?)"

say "STEP 4  k sweep on the strongest models (CV only, test untouched)"
for K in 800 1500; do
    caffeinate -dimsu "$PY" paper_pipeline/pipeline/train_eval7.py \
        --suffix lesion_equalize_all --n-features $K \
        --classifiers "LightGBM,CatBoost,Logistic Regression,SVM (RBF)" \
        >> "$L/train7_all_k${K}.log" 2>&1
    say "  k=$K done (exit $?)"
done
say "=========== PRO PIPELINE COMPLETE ==========="
