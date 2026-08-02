# Tier-1 + Tier-2 experiments — run guide

This guide is for running the post-revision experiments identified in
`Mohamed_Shoieb_Master_Paper_ElsevierV7/REVISION_LOG.md`.

**Headline configuration these scripts assume:**
- λ = DFT (`e^{i 2π k / N}`)
- Classifier = MLP (`hidden_layer_sizes=(100, 50)`, `alpha=0.001`)
- Features = hybrid (handcrafted + MKT + frozen ResNet-50)
- N_features kept after SelectKBest = 360
- Baseline test result we're starting from: 93.19% accuracy, 97.36% AUC

---

## What each script does

### `tier1_stats_and_ablation.py`
- Retrains MLP on the hybrid features (~30 sec).
- Bootstrap 95% CIs on the 6 headline metrics (10,000 resamples).
- McNemar's test against Extra Trees + λ=low_pass (the joint runner-up).
- Feature-family ablation: 7 retrains
  (ALL / no-MKT / no-CNN / no-Handcrafted / MKT-only / CNN-only / Handcrafted-only).

**Total time:** ~15–20 min on CPU.

**Resolves:** P0-1 (ablation), P0-2 (CIs + McNemar). These were raised by every reviewer in the simulated panel.

### `tier2_p_sensitivity.py`
- Re-extracts MKT features at p ∈ {0.3, 0.5, 0.7} (only for λ=DFT).
- Reuses the existing frozen-CNN features (CNN doesn't depend on p).
- Trains MLP on each p, reports the 6 headline metrics.

**Total time:** ~3–6 hours on CPU. The MKT extractor is the expensive part — it has to process all ~2,400 images twice (once for p=0.3, once for p=0.7; p=0.5 can reuse the existing pickle with `--reuse-p05`).

**Resolves:** P1-4 (Krawtchouk p-justification).

---

## Run order

Run from the practical project root:

```bash
cd "/Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3"
```

### Step 1: install the one new dependency (if not already present)

```bash
pip install statsmodels    # for McNemar's test
```

### Step 2: Tier-1 (quick — do this first)

```bash
python paper_pipeline/scripts/tier1_stats_and_ablation.py
```

Outputs (in `paper_pipeline/output/results/`):
- `tier1_bootstrap_ci_<ts>.csv` — point estimate + 95% CI for each metric
- `tier1_mcnemar_<ts>.csv` — McNemar's test result
- `tier1_ablation_<ts>.csv` — 7-row ablation table
- `tier1_predictions_<ts>.npz` — cached MLP predictions (reusable later)
- `tier1_report_<ts>.txt` — human-readable summary of everything above

### Step 3: Tier-2 (slow — start this and let it run)

```bash
# Full sweep (3 p values, ~3-6 hours)
python paper_pipeline/scripts/tier2_p_sensitivity.py

# OR: skip p=0.5 (we already have results for it)
python paper_pipeline/scripts/tier2_p_sensitivity.py --p-values 0.3,0.7

# OR: smoke test at just one p
python paper_pipeline/scripts/tier2_p_sensitivity.py --p-values 0.3
```

Outputs:
- `paper_pipeline/output/results/tier2_p_sensitivity_<ts>.csv` — one row per p
- `paper_pipeline/output/features/tier2_p_features/` — re-extracted features per p

---

## After both scripts finish

Send me (or paste here) the contents of:
1. `tier1_report_<ts>.txt`
2. `tier1_bootstrap_ci_<ts>.csv`
3. `tier1_ablation_<ts>.csv`
4. `tier2_p_sensitivity_<ts>.csv`

I'll then integrate the numbers into `main.tex`:
- Bootstrap CIs → §5.5 (after the headline accuracy)
- McNemar's p-value → §5.5 (one sentence)
- Ablation table → new subsection just before §5.5
- p-sensitivity table or plot → new subsection in §5

---

## Things that can go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: statsmodels` | Not installed | `pip install statsmodels` |
| `ModuleNotFoundError: pandas` | Wrong Python env | `cd` into project root and use the project's venv |
| Tier-2 CNN columns not found | Heuristic missed the CNN col names in your hybrid pickle | Run `python -c "import pandas as pd; df = pd.read_pickle('paper_pipeline/output/features/features_train_dft_hybrid.pkl'); print(df.columns.tolist()[:20]); print('--'); print([c for c in df.columns if 'resnet' in c.lower() or 'cnn' in c.lower()][:5])"` — then either rename your CNN columns or edit `cnn_cols` heuristic in `tier2_p_sensitivity.py` |
| Tier-1 results don't reproduce the 93.19% headline | sklearn version mismatch | Lock your sklearn version with `pip show scikit-learn` and compare to the one used for the cached `summary_20260517_183355.csv` results |

---

## What I'll do with the results

Once you send me the four output files, the paper updates I'll make to `main.tex` are:

1. **Abstract** — change `92.57% / 97.61%` (CatBoost + odd-harmonic) to `93.19% / 97.36%` (MLP + DFT), update sens/spec/F1, and add the 95% CI for accuracy and AUC.

2. **§5.5 (Overall comparison)** — re-headline as MLP + DFT, add the McNemar p-value sentence, add the bootstrap CIs in parentheses for each metric.

3. **§5 (NEW subsection before §5.5)** — feature-family ablation table + one paragraph interpreting it.

4. **§5 (NEW subsection)** — Krawtchouk p-sensitivity table or 3-point line plot + one paragraph.

5. **§6 Limitations** — remove P0-1, P0-2, and P1-4 from the "still to do" list since they're now done.

6. **Figure 6 (proposed_framework_figure.png)** — update the metric block in panel (b) to show 93.19% / 97.36% and change "Headline: CatBoost" → "Headline: MLP", λ formula from odd-harmonic → DFT.
