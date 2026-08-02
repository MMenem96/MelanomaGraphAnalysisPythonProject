# paper_pipeline — reproducible run for the Elsevier paper

This is the single, clean folder containing the code used to produce the
numbers reported in the BCC vs BKL classification paper. Everything in
this folder is either an orchestration script, a thin wrapper, or
configuration. The heavy library code (feature extractors, segmenter,
deep DNN) still lives in `src/` so we don't duplicate ~100 KB of working
implementations — but every entry point for reproducing the paper lives
here.

## What is in this folder

```
paper_pipeline/
├── configs/
│   └── pipeline.yaml             # one place for all knobs (N, p, k-fold, seed…)
├── data_prep/
│   ├── 01_fetch_ham10000_metadata.py     # download HAM10000_metadata.csv
│   ├── 02_verify_ham10000_ids.py         # assert 10,015 ids match masks folder
│   ├── 03_filter_bcc_bkl.py              # → 514 BCC + 1,099 BKL canonical ids
│   └── 04_apply_masks_to_lesions.py      # write segmented PNGs (white bg)
├── pipeline/
│   ├── dataset.py                # canonical sample loader
│   ├── preprocessing.py          # hair removal + Telea inpainting + Gaussian blur
│   ├── augmentation_transform.py # train-only horizontal + vertical flip
│   ├── feature_extraction.py     # SPLIT first, then extract (train ×3, test ×1)
│   ├── classifiers.py            # the 9 active classifiers + hyperparameters
│   └── train_eval.py             # MI selection → RobustScaler → CV → eval
├── scripts/
│   └── run_full_experiment.py    # one-button entry point
├── output/                       # generated (features, results, qa visualisations)
│   ├── features/                 # features_train_aug.pkl, features_test.pkl
│   ├── results/                  # results_<timestamp>.csv  one row per classifier
│   └── qa/                       # QA visualisations (hair removal, augmentation)
└── README.md
```

## What changed vs the old code

| Concern | Old | New |
|---|---|---|
| Train/test split | After augmentation | **Before** augmentation, on image IDs |
| Augmentation | Vertical flip only, applied to all data on disk | **Horizontal + vertical** flip, applied **only to training images at extraction time** |
| Hair removal | `hair_detected = False` in graph path; correctly enabled in feature-extraction path | Confirmed enabled, Telea inpainting (`cv2.INPAINT_TELEA`) |
| BKL count | 1,229 (unknown filter) | **1,099** (canonical HAM10000 intersected with available masks) |
| BCC count | 514 | 514 (unchanged) |
| Classifier set | 14 in `CLASSIFIERS` dict, only some in each table | **9** active classifiers, all 9 in every result row |
| Test set usage | Touched repeatedly | Touched once per (classifier, run) |
| MI selection | Fit on train only | Fit on train only (unchanged) |
| Scaling | Fit on train only | Fit on train only (unchanged) |
| Cross-validation | Stratified k-fold on train | Stratified k-fold on train (unchanged) |

## How to reproduce the paper numbers

Requires the same Python env as the rest of the repo (`uv sync` or
`poetry install`). The `src/` library must be importable, which it is by
default when you run from the project root.

### One command

```bash
python paper_pipeline/scripts/run_full_experiment.py
```

This will: download metadata → verify → filter → segment → split →
extract features (train ×3, test ×1) → MI-select → RobustScale → k-fold CV
→ evaluate the 9 classifiers on the held-out test set ONCE → write
`output/results/results_<timestamp>.csv`.

### Step-by-step (recommended the first time)

```bash
python paper_pipeline/data_prep/01_fetch_ham10000_metadata.py
python paper_pipeline/data_prep/02_verify_ham10000_ids.py
python paper_pipeline/data_prep/03_filter_bcc_bkl.py            # expect 514 + 1099
python paper_pipeline/data_prep/04_apply_masks_to_lesions.py    # writes segmented PNGs
python paper_pipeline/pipeline/feature_extraction.py            # writes two pickles
python paper_pipeline/pipeline/train_eval.py                    # writes CSV
```

### Debug shortcuts

* `--limit 5` in `feature_extraction.py` to sanity-check the pipeline on 5 BCC + 5 BKL.
* `--no-augment` in `feature_extraction.py` or the orchestrator to disable training-set augmentation (debug only — paper numbers require augmentation).

## Reproducibility guarantees

* `random_state=42` is used for the train/test split (`feature_extraction.py`) and for every classifier (`classifiers.py`).
* `verify_no_leakage` in `feature_extraction.py` raises if any `source_id` appears in both the train and test pickles.
* `train_eval.py` re-checks the same invariant on load.
* `X_test` appears in `train_eval.py` only in `selector.transform`, `scaler.transform`, `clf.predict`, `clf.predict_proba`, and the metric calls. Grep is your friend:
  ```bash
  grep -n "X_test" paper_pipeline/pipeline/train_eval.py
  ```

## Where the published numbers live in the new layout

After a run, `output/results/results_<timestamp>.csv` contains one row per
classifier with columns: `test_accuracy, test_sensitivity, test_specificity,
test_precision, test_f1, test_auc, cv_accuracy_mean, cv_accuracy_std,
cv_auc_mean, cv_auc_std, fit_seconds`. Those are the numbers that go into
Tables 2–5 of the paper.

## What is NOT in this folder

* The old monolithic scripts (`manual_train_features.py`, `manual_run_main.py`,
  `app.py`, `advanced_training.py`, etc.) — these will be moved to `ARCHIVE/`.
* Graph-pipeline code — never used in the paper.
* Stale data folders (`bcc_segmented_augmented`, `bcc_segmented_vertical_flipped`,
  `sk_filtered`, …) — these are moved to `ARCHIVE/data/` for historical reference.

## Library code in `src/` that this folder uses

* `src.segmentation.skin_lesion_processor.SkinLesionProcessor` — hair detection and Telea inpainting.
* `src.conventional_features.ConventionalFeatureExtractor` — geometric, color, texture, MKT features.
* `src.tabular_dnn_classifier.TabularDNNClassifier` — the Deep DNN with focal loss.

These are imported by `paper_pipeline/pipeline/*.py`; they are not modified.
