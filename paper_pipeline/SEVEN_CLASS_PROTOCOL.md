# 7-class HAM10000 — what the field does, and the protocol we will follow

Compiled 2026-08-02 from a literature sweep + our own audit of the dataset.
Nothing in this file is a measured result of our pipeline. Numbers attributed
to other papers are their claims; numbers attributed to us are dataset facts
computed from `data/HAM10000_metadata.csv`.

---

## 1. The dataset facts (computed, not quoted)

| class | images | unique `lesion_id` | images/lesion |
|---|---|---|---|
| nv    | 6,705 | 5,403 | 1.24 |
| mel   | 1,113 |   614 | 1.81 |
| bkl   | 1,099 |   727 | 1.51 |
| bcc   |   514 |   327 | 1.57 |
| akiec |   327 |   228 | 1.43 |
| vasc  |   142 |    98 | 1.45 |
| df    |   115 |    73 | 1.58 |
| **total** | **10,015** | **7,470** | 1.34 |

Two consequences:

1. **Majority-class baseline = 66.9%.** Any 7-class accuracy must be read
   against this floor, not against zero. A model predicting `nv` always
   scores 66.9%.
2. **10,015 images are not 10,015 independent samples.** 2,545 images are
   repeat photographs of a lesion already present elsewhere in the dataset.
   A random image-level split puts the *same physical lesion* on both sides.

## 2. How the field actually handles this

### 2.1 Splitting — the dominant methodological flaw

The single biggest determinant of a reported HAM10000 number is **where the
split is taken**, not the architecture.

- **Image-level random split (most common).** Ignores `lesion_id`. Leaks
  near-duplicate views across the boundary.
- **Split-after-augmentation / after-SMOTE (common in the 97–99% papers).**
  The minority classes are oversampled to balance the dataset *first*, then
  the balanced pool is partitioned. Synthetic derivatives of test images end
  up in training. A published leakage audit on HAM10000 found **1,095 of
  4,810 lesions (22.8%) crossed the split boundary** under this scheme; after
  rebuilding with a verified lesion-level partition, the strongest
  configuration fell from a **94.57% baseline to 78.41% ± 0.74 accuracy
  (macro-F1 0.804)**. That is a ~16-point correction attributable purely to
  the partitioning scheme.
- **Lesion-grouped split (correct).** `GroupShuffleSplit` / `StratifiedGroupKFold`
  on `lesion_id`. Abhishek, Jain & Hamarneh (*Nature Scientific Data*, 2025,
  "Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological
  Image Datasets", arXiv:2401.14497) document duplication, train/test leakage,
  and mislabeling in HAM10000 and publish corrected splits.

### 2.2 Imbalance handling

Ranked by how the literature treats them:

1. **Class weighting in the loss / `class_weight='balanced'`** — the default,
   safe choice. No leakage risk. Our `classifiers.py` already uses
   `class_weight='balanced'` for SVM.
2. **Focal loss** (α-balanced, γ modulation) — standard for CNN training on
   this dataset; down-weights the easy `nv` bulk.
3. **SMOTE / ADASYN on the feature vectors** — acceptable *only* if fitted
   strictly on the training fold, after the split. Most of the inflated
   results in the literature come from violating this.
4. **Geometric augmentation** — flips/rotations, train-only, post-split
   (already how our pipeline works).

### 2.3 Metrics

For a 66.9%-majority dataset, overall accuracy alone is not interpretable.
The reporting standard is:

- **Balanced multiclass accuracy (BMA)** — the official ISIC 2018 Task 3
  metric (mean per-class recall).
- **Macro-F1** — headline for imbalanced multiclass.
- **Macro one-vs-rest AUC** — forgiving of imbalance; expect it to stay high
  (0.95+) even when macro-F1 is mediocre. Do not use it as the headline.
- **Per-class recall + full 7×7 confusion matrix** — mandatory; this is where
  `df` (115 images) and `vasc` (142) tell the truth.

### 2.4 Realistic performance envelope

- **ISIC 2018 Task 3** (this exact 7-class task, curated held-out test set):
  winning entry ≈ **0.885 balanced multiclass accuracy**. Expert dermatologist
  readers scored substantially lower.
- **Leak-free published re-evaluations**: ~78–88% accuracy, macro-F1 ~0.75–0.85.
- **Papers reporting 97–99%**: essentially always explained by (a) split after
  balancing, (b) image-level split with lesion leakage, or (c) overall accuracy
  on an untouched imbalance. Treat as non-comparable.

Cited in our own bibliography, for calibration: Pacal2024 (Swin, ISIC 2019,
89.36%), Khan2025 (CNN, HAM10000, 91.63%), Arshad2025 (>90%), Halawani2025
(EViT+DenseNet169, ISIC 2018, per-class 92.2/92.3 for BCC/BKL),
Babatunde2025 (98.20% — the outlier pattern described above).

### 2.5 Architectures

- End-to-end **fine-tuned CNN** (EfficientNet, DenseNet, ResNet) — the workhorse.
- **Transformers** (Swin, ViT hybrids) — current best on the large benchmarks.
- **Frozen deep features + classical classifier** (our family) — well
  represented but consistently below end-to-end fine-tuning on multiclass,
  because one global feature ranking must serve all 21 class pairs.

---

## 3. Protocol for our 7-class run

Decisions, and why:

| Decision | Choice | Rationale |
|---|---|---|
| Split unit | **`lesion_id`, `StratifiedGroupKFold` / `GroupShuffleSplit`** | §2.1. Also run the image-level split for direct comparability with our binary paper. |
| Split ratio | 80/20, `random_state=42` | Matches the binary protocol. |
| Augmentation | h+v flip, **train-only, post-split** | Unchanged from the binary pipeline; already correct. |
| Imbalance | `class_weight='balanced'`; no SMOTE in the primary run | Avoids the dominant literature flaw. SMOTE only as a labelled secondary arm, fitted on the training fold. |
| Feature selection | mutual info, fit on train only | Unchanged. `k` may need to exceed 360 for 7 classes — sweep it. |
| Scaling | RobustScaler, fit on train only | Unchanged. |
| Metrics | balanced accuracy, macro-F1, macro-OvR AUC, per-class recall, 7×7 confusion matrix | §2.3. Overall accuracy reported but not headlined. |
| Baseline | majority-class (66.9%) reported alongside | Makes the number interpretable. |

### Secondary experiment worth more than the 7-class run itself

Re-run the **binary BCC vs BKL protocol with a lesion-grouped split**
(BCC 514 img / 327 lesions = 1.57×; BKL 1,099 / 727 = 1.51×). Our current
93.50% uses `train_test_split` on `image_id` with stratification
(`pipeline/feature_extraction.py:214`), which carries exactly the exposure
described in §2.1. This is cheap — the features already exist — and it
pre-empts the strongest reviewer attack on the paper.
