# Master research — two-track state

Canonical status file for **both** papers and **both** practical projects.
Last updated: **2026-08-02**.

Read this first in any new session. If you edit either track, update this file.

---

## Track A — BINARY paper (BCC vs BKL) — **FROZEN, UNDER JOURNAL REVIEW**

> ⚠️ **Do not modify anything in Track A without an explicit instruction.**
> It is under review at a journal. Any change must be a deliberate decision,
> not a side effect of Track B work.

| | |
|---|---|
| Paper repo | `~/Desktop/Work/Master/Mohamed_Shoieb_Master_Paper_ElsevierV7/` |
| Main file | `main.tex` (Elsevier `elsarticle` class) |
| Practical project | `~/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/` |
| Reproducible code | `MelanomaGraphAnalysisV3/paper_pipeline/` |
| Git state | branch `main`, clean, last commit `1e8684e` "Remove headers" (2026-06-18) |
| Journal / submission date | **TBD — fill in** |

### What the paper claims

- Task: **binary** BCC vs BKL, HAM10000. 514 BCC + 1,099 BKL = 1,613 images.
- Features: handcrafted (geometric, color, texture) + **MKT = 76** (RGB per-channel)
  + frozen ResNet-50 (2,048) → **2,509 total**. Handcrafted subtotal 385 (380 mask).
- MI filtering fit on train only → **k = 360** features. RobustScaler, train only.
- Split: **80/20 on `image_id`**, stratified, `random_state=42`. Augmentation
  (h+v flip) applied **train-only, post-split**, BCC only.
- Best result: **MLP**, complex odd-harmonic λ_k = e^{i(2k+1)π/N}, p=0.5, N=64:
  **93.50%** accuracy (95% CI [90.71, 95.98]), 91.26% sensitivity,
  94.55% specificity, 89.95% F1, **96.95% AUC** (CI [95.13, 98.51]),
  on 323 held-out images.
- Canonical ablation CSV: `phase_a_ablation_20260522_154502.csv`.

### Last edits made (2026-06-18 session)

1. `5caa0cf`/`c82c05a` — Figure 4 preprocessing TikZ diagram enlarged (1.4 → 2.6 cm).
2. `156d8ac` — **MKT count fix**: transform documented as per-RGB-channel, not
   grayscale; 23 stats/channel × 3 + 7 cross-channel = 76; magnitude *Minimum*
   replaced by *Entropy* in `tab:features`; totals reconciled to 2,509.
3. `1e8684e` — removed bold `\emph{...}` lead-in headers from ablation paragraphs.

### Known open risk (not yet answered, relevant if reviewers push back)

The split is on `image_id`, **not** `lesion_id`. HAM10000 has 10,015 images but
only 7,470 unique lesions (BCC 514 img / 327 lesions = 1.57×; BKL 1,099 / 727 =
1.51×), so repeat photographs of the same physical lesion can land on both sides
of the train/test boundary. The paper is transparent about this — the abstract
says "split by image identifier" and §Future Work lists group-stratified CV —
but a published leakage audit on HAM10000 measured a **94.57% → 78.41%** drop
when a comparable protocol was corrected to lesion-level partitioning.

**Planned mitigation (not yet run):** re-run the binary protocol with
`GroupShuffleSplit` on `lesion_id`. Features already exist, so it is cheap.
If 93.50% holds, it strengthens the paper; if it drops, we need to know before
a reviewer says it.

---

## Track B — 7-CLASS paper — **ACTIVE, IN SETUP**

A **separate paper**, but the **same code repo** as Track A — it lives on the
branch `feature/seven-class-multiclass`. Not a revision of Track A's paper.

| | |
|---|---|
| Code | `MelanomaGraphAnalysisV3/paper_pipeline/`, branch `feature/seven-class-multiclass` |
| Paper repo | **not created yet** — planned `~/Desktop/Work/Master/Mohamed_Shoieb_Master_Paper_7Class/` |
| Started | 2026-08-02 |

### Layout (inside this repo, on the Track B branch)

```
paper_pipeline/
├── PROJECT_STATE.md                     ← this file
├── SEVEN_CLASS_PROTOCOL.md              ← literature synthesis + protocol decisions
├── data_prep/05_prepare_all_classes.py  ← 7-class filter + lesion_id map + masking
├── pipeline/                            ← shared with Track A; extend, never break binary
data/
├── HAM10000_metadata.csv                ← shared
├── HAM10000_binary_mask/                ← shared, 10,015 masks
├── ham10000_images/                     → symlink to ~/Desktop/Work/Master/Images/HAM100000-Images
├── canonical/                           ← Track A (binary) segmented images
└── canonical7/                          ← Track B, 10,015 segmented PNGs + samples.csv (1.0 GB)
```

Isolation rule (branch-level): Track A's frozen code is preserved by
`release/paper-v7-binary-stable` + tag `paper-v7-submitted`. Track B edits shared
files (`dataset.py`, `feature_extraction.py`, `train_eval.py`) **additively** —
the binary path must keep working and producing identical numbers.

### Dataset facts (computed 2026-08-02 from the metadata CSV)

| class | label | images | unique lesions | img/lesion |
|---|---|---|---|---|
| akiec | 0 | 327 | 228 | 1.43 |
| bcc | 1 | 514 | 327 | 1.57 |
| bkl | 2 | 1,099 | 727 | 1.51 |
| df | 3 | 115 | 73 | 1.58 |
| mel | 4 | 1,113 | 614 | 1.81 |
| nv | 5 | 6,705 | 5,403 | 1.24 |
| vasc | 6 | 142 | 98 | 1.45 |
| **total** | | **10,015** | **7,470** | 1.34 |

All 10,015 metadata IDs have both a mask and a raw image on disk (verified).
The image folder holds 11,720 JPGs; the extra 1,705 (`ISIC_0034321`+) are the
ISIC-2018 val/test additions and are excluded by the metadata filter.

Majority-class baseline = **66.9%** (nv). Every accuracy number must be read
against this floor.

### Agreed protocol

- **Split:** 80/20, `random_state=42`, grouped on `lesion_id`
  (`StratifiedGroupKFold`). 5-fold CV on train for model selection; test touched
  **once** per (config, classifier), same audit rule as Track A.
- **Two arms:** Arm A = image-level split (matches Track A), Arm B =
  lesion-grouped. The A−B gap measures leakage inside our own pipeline.
- **Imbalance (user decision, 2026-08-02): `--balance equalize` is the primary
  arm.** Every class is augmented up to the largest class (nv, ~4,290 train
  images). Flips alone cap at 4 variants, so the op pool adds rotations
  (15° steps × 4 flips = up to 96 variants); per class, the first
  `ceil(target / n_c)` ops are drawn deterministically (seeded). Needed
  multipliers: mel 6×, bkl 6×, bcc 13×, akiec 21×, vasc 47×, **df 58×**.
  Augmentation stays **train-only, post-split**. `class_weight='balanced'`
  is left on but becomes a no-op once classes are equal.
  Secondary arm `--balance capped4`: `min(4, ceil(3000/n_c))`, nv ×1 → 58:1
  becomes ~14:1, ~19k rows instead of ~32k.
  **No SMOTE in either arm** — that is the practice inflating the 97–99%
  literature.
  ⚠️ Caveat to keep in the paper: df's ~4,290 augmented rows come from only
  **73 unique lesions**, so train/CV scores will overstate. Held-out per-class
  recall and the confusion matrix are the honest numbers and must be reported.
- **Metrics:** balanced accuracy (ISIC 2018 official metric), macro-F1,
  macro-OvR AUC, per-class recall, 7×7 confusion matrix, majority baseline.
  Overall accuracy reported but **not** headlined.
- **Unchanged from Track A:** preprocessing, MKT configs, the 9 classifiers,
  MI selection + RobustScaler fit on train only. `k=360` was tuned for one
  decision boundary, so it gets swept on train CV.

### Realistic expectation (from the literature, not measured)

ISIC 2018 Task 3 — this exact 7-class task, curated held-out test set — was won
at **≈0.885 balanced multiclass accuracy**. Leak-free re-evaluations land at
~78–88% accuracy, macro-F1 ~0.75–0.85. Papers claiming 97–99% almost always
split after balancing or split at image level. We should expect our numbers to
be **lower than Track A's 93.50%**, and that is the correct, expected outcome —
not a failure.

### Current state — where we stopped

- [x] Literature sweep → `SEVEN_CLASS_PROTOCOL.md`
- [x] Verified all 7 classes' images + masks present
- [x] Isolated project created; V3 restored to its prior state
- [x] `data_prep/05_prepare_all_classes.py` written — **not yet run**
- [ ] Fix `PROJECT_ROOT` depth in copied scripts (was `parents[2]` for the V3
      layout, must be `parents[1]` here)
- [ ] `dataset.py` — 7-class loader reading `data/canonical7/samples.csv`
- [ ] `feature_extraction.py` — lesion-grouped split + per-class augmentation
      multipliers + N-class support
- [ ] `train_eval.py` — multiclass metrics
- [ ] Run the 7-class experiment
- [ ] Create the Track B paper repo

### Immediate next command

```bash
cd ~/Desktop/Work/Master/Practical\ Project/MelanomaMulticlass7
python data_prep/05_prepare_all_classes.py --workers 8
```

---

## Git layout (set 2026-08-02)

**Two repos, not three.** Both tracks share the `MelanomaGraphAnalysisV3` code
repo; the 7-class work is a *branch*, not a separate project. Track B will still
become its own paper — only the code is shared.

| repo | frozen branch | working branch |
|---|---|---|
| `Mohamed_Shoieb_Master_Paper_ElsevierV7` (Track A paper) | `release/elsevier-v7-under-review` | `work/paper-v7-revisions` |
| `MelanomaGraphAnalysisV3` (both tracks' code) | `release/paper-v7-binary-stable` (tag `paper-v7-submitted`) | Track A: `work/binary-lesion-grouped-recheck`<br>Track B: `feature/seven-class-multiclass` |

V3's `paper_pipeline/` (44 source files) was previously **untracked** and is now
committed on its release branch. Outputs/features/data stay gitignored (1.0 GB
each for `paper_pipeline/output/` and `data/canonical7/`).
Track B's **paper** repo does not exist yet.

Superseded 2026-08-02: the standalone `MelanomaMulticlass7/` project was folded
into this repo as `feature/seven-class-multiclass` and deleted.

## Cross-track rules

1. **Never fabricate a number.** Every value in either `main.tex` must trace to
   a real CSV in the corresponding practical project.
2. **Track A is frozen** while under review. Track B changes must never reach
   into `MelanomaGraphAnalysisV3/` or `Mohamed_Shoieb_Master_Paper_ElsevierV7/`.
3. `MelanomaGraphAnalysisV3/paper_pipeline/` is **untracked in git**. It is the
   reproducibility folder for a paper under review and should be committed.
