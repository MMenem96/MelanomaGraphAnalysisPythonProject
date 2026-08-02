"""Phase A — full 36-cell re-run on the flip-only May 17 hybrid pickles.

This generates ALL the numbers needed for the paper's Tables 2-5 + headline
in ONE consistent run, with reproducible seeds. It also saves prediction
arrays for the headline cell so bootstrap CIs and McNemar's can be derived.

Headline: λ=DFT, classifier=MLP, expected ~93.19% accuracy.

Outputs (paper_pipeline/output/results/):
    phase_a_summary_<ts>.csv         36-row CSV: one row per (classifier, λ)
    phase_a_predictions_<ts>.npz     dict of 36 (y_pred, y_proba) arrays
    phase_a_report_<ts>.txt          summary report

Then this same script runs:
    - Bootstrap CIs for MLP+DFT (10,000 resamples)
    - McNemar's test vs runner-up (best non-MLP+DFT cell)
    - Feature-family ablation for MLP+DFT (7 configs)

Run from project root:
    cd "/Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3"
    python paper_pipeline/scripts/phase_a_full_grid.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
import catboost as cb
import lightgbm as lgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"

LAMBDAS = ["low_pass", "high_pass", "dft", "odd_harmonic"]
HEADLINE_LAMBDA = "odd_harmonic"
HEADLINE_CLASSIFIER = "MLP"

N_FEATURES = 360
RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
BOOKKEEPING_COLS = {"source_id", "aug_tag", "label"}


# ----------------------------------------------------------------------------
# Classifier builders — match the paper's "9 classifiers" list verbatim
# ----------------------------------------------------------------------------

def build_classifier(name: str):
    if name == "SVM (RBF)":
        return SVC(C=100, kernel="rbf", gamma="scale",
                   class_weight="balanced", probability=True,
                   random_state=RANDOM_STATE)
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1,
        )
    if name == "CatBoost":
        return cb.CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=False, random_state=RANDOM_STATE,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50), alpha=0.001,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
    if name == "Deep DNN":
        # Use a simple PyTorch-free fallback: a deeper MLP with dropout-equivalent
        # via L2 + larger hidden layers (the original src/tabular_dnn_classifier.py
        # uses TF; we mirror its capacity here).
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), alpha=0.01,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown classifier: {name}")


CLASSIFIERS_IN_ORDER = [
    "SVM (RBF)", "LightGBM", "CatBoost", "Gradient Boosting", "Extra Trees",
    "KNN", "Logistic Regression", "MLP", "Deep DNN",
]


# ----------------------------------------------------------------------------
# Feature-family classifier (used by ablation)
# ----------------------------------------------------------------------------

MKT_TOKENS = ("mdfkt", "_mkt_", "real_", "imag_", "magnitude_", "phase_",
              "_real_", "_imag_", "_magnitude_", "_phase_")
COLOR_TOKENS = ("_rgb_", "_hsv_", "_lab_", "_cielab_", "rgb_", "hsv_", "lab_",
                "cielab_", "channel_", "histogram_", "dominant_color",
                "color_asym", "color_uniform")
TEXTURE_TOKENS = ("glcm", "lbp_", "gabor", "wavelet", "gradient_",
                  "haralick", "_contrast", "_dissimilarity",
                  "_homogeneity", "_energy_", "_correlation_")
GEOMETRIC_TOKENS = ("area", "perimeter", "eccentric", "circularity",
                    "compactness", "convexity", "diameter", "asymmetry",
                    "fractal", "curvature", "smoothness", "hu_moment")
CNN_TOKENS = ("resnet", "cnn_", "_cnn", "deep_feat", "emb_",
              "efficientnet", "densenet")


def classify_feature(name: str) -> str:
    s = name.lower()
    if any(t in s for t in MKT_TOKENS): return "MKT"
    if any(t in s for t in CNN_TOKENS): return "CNN"
    if any(t in s for t in COLOR_TOKENS): return "Color"
    if any(t in s for t in TEXTURE_TOKENS): return "Texture"
    if any(t in s for t in GEOMETRIC_TOKENS): return "Geometric"
    return "Unknown"


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def split_xy(df, drop_families=None):
    cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    if drop_families:
        cols = [c for c in cols if classify_feature(c) not in drop_families]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, cols


def _mi_classif_seeded(X, y):
    # mutual_info_classif's internal kNN estimator uses a random tie-breaker;
    # without an explicit seed, sklearn picks a different feature subset on
    # each run, which propagates downstream. Pin both seed and n_jobs for
    # full determinism.
    return mutual_info_classif(X, y, random_state=RANDOM_STATE,
                                n_neighbors=3, n_jobs=1)


def fit_select_scale(X_tr, y_tr, X_te, n_features=N_FEATURES):
    k = min(n_features, X_tr.shape[1])
    sel = SelectKBest(_mi_classif_seeded, k=k).fit(X_tr, y_tr)
    X_tr_s = sel.transform(X_tr); X_te_s = sel.transform(X_te)
    sca = RobustScaler().fit(X_tr_s)
    return sca.transform(X_tr_s), sca.transform(X_te_s)


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0.0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision":   precision_score(y_true, y_pred, zero_division=0.0),
        "f1":          f1_score(y_true, y_pred, zero_division=0.0),
        "auc":         roc_auc_score(y_true, y_proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    out = {k: [] for k in ["accuracy", "sensitivity", "specificity",
                            "precision", "f1", "auc"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2: continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        acc = (tn + tp) / n
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        out["accuracy"].append(acc); out["sensitivity"].append(sens)
        out["specificity"].append(spec); out["precision"].append(prec)
        out["f1"].append(f1)
        try: out["auc"].append(roc_auc_score(yt, ypr))
        except ValueError: pass
    return {k: (float(np.mean(v)),
                float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))
            for k, v in out.items()}


def mcnemar_test(y_true, yp_a, yp_b):
    ok_a, ok_b = (yp_a == y_true), (yp_b == y_true)
    n00 = int(np.sum(~ok_a & ~ok_b)); n01 = int(np.sum(~ok_a & ok_b))
    n10 = int(np.sum(ok_a & ~ok_b));  n11 = int(np.sum(ok_a & ok_b))
    use_exact = (n01 + n10) < 25
    res = mcnemar([[n11, n10], [n01, n00]],
                  exact=use_exact, correction=True)
    return {"n_both_correct": n11, "n_A_only_correct": n10,
            "n_B_only_correct": n01, "n_both_wrong": n00,
            "method": "exact_binomial" if use_exact else "chi2_with_continuity",
            "statistic": float(res.statistic), "pvalue": float(res.pvalue)}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase_a")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Phase A — full 36-cell re-run on flip-only May 17 hybrid pickles")
    log.info("=" * 70)

    # ------------------------ Step 1: full grid ------------------------
    grid_rows = []
    predictions = {}  # key = f"{lambda}_{classifier}"
    for lam in LAMBDAS:
        log.info("\nλ = %s", lam)
        train_pkl = FEATURES_DIR / f"features_train_{lam}_hybrid.pkl"
        test_pkl  = FEATURES_DIR / f"features_test_{lam}_hybrid.pkl"
        train_df = pd.read_pickle(train_pkl)
        test_df  = pd.read_pickle(test_pkl)

        X_tr, y_tr, _ = split_xy(train_df)
        X_te, y_te, _ = split_xy(test_df)
        X_tr_s, X_te_s = fit_select_scale(X_tr, y_tr, X_te)

        for clf_name in CLASSIFIERS_IN_ORDER:
            t0 = time.time()
            clf = build_classifier(clf_name)
            try:
                clf.fit(X_tr_s, y_tr)
                y_pred = clf.predict(X_te_s)
                try:
                    y_proba = clf.predict_proba(X_te_s)[:, 1]
                except Exception:
                    y_proba = clf.decision_function(X_te_s)
                m = compute_metrics(y_te, y_pred, y_proba)
                t = time.time() - t0
                log.info("  %-20s acc=%.4f sens=%.4f spec=%.4f auc=%.4f  (%.1fs)",
                         clf_name, m["accuracy"], m["sensitivity"],
                         m["specificity"], m["auc"], t)
                grid_rows.append({"lambda": lam, "classifier": clf_name,
                                   **{k: v for k, v in m.items()
                                      if isinstance(v, (int, float))},
                                   "fit_seconds": round(t, 2)})
                predictions[f"{lam}_{clf_name}"] = {
                    "y_test": y_te, "y_pred": y_pred, "y_proba": y_proba
                }
            except Exception as e:
                log.error("  %-20s FAILED: %s", clf_name, e)
                grid_rows.append({"lambda": lam, "classifier": clf_name,
                                   "error": str(e), "fit_seconds": time.time() - t0})

    grid_df = pd.DataFrame(grid_rows)
    grid_csv = RESULTS_DIR / f"phase_a_summary_{ts}.csv"
    grid_df.to_csv(grid_csv, index=False)
    log.info("\n→ %s", grid_csv)

    # Save all 36 prediction arrays in one npz
    pred_npz = RESULTS_DIR / f"phase_a_predictions_{ts}.npz"
    np.savez(pred_npz,
             **{f"{k}_y_test":  v["y_test"]  for k, v in predictions.items()},
             **{f"{k}_y_pred":  v["y_pred"]  for k, v in predictions.items()},
             **{f"{k}_y_proba": v["y_proba"] for k, v in predictions.items()})
    log.info("→ %s (36 cells × 3 arrays)", pred_npz)

    # ------------------------ Step 2: bootstrap CIs for headline ------------------------
    head_key = f"{HEADLINE_LAMBDA}_{HEADLINE_CLASSIFIER}"
    head_pred = predictions[head_key]
    log.info("\n" + "=" * 70)
    log.info("Bootstrap CIs for HEADLINE (%s)...", head_key)
    log.info("=" * 70)
    cis = bootstrap_ci(head_pred["y_test"], head_pred["y_pred"], head_pred["y_proba"])
    head_metrics = compute_metrics(head_pred["y_test"], head_pred["y_pred"], head_pred["y_proba"])
    ci_rows = []
    for k, (mean, lo, hi) in cis.items():
        ci_rows.append({"metric": k, "point_estimate": head_metrics[k],
                        "bootstrap_mean": mean,
                        "ci95_low": lo, "ci95_high": hi})
        log.info("  %-12s point=%.4f mean=%.4f CI95=[%.4f, %.4f]",
                 k, head_metrics[k], mean, lo, hi)
    pd.DataFrame(ci_rows).to_csv(
        RESULTS_DIR / f"phase_a_bootstrap_ci_{ts}.csv", index=False)

    # ------------------------ Step 3: McNemar's vs runner-up ------------------------
    # Find runner-up = second-highest accuracy across all 36 cells
    grid_df_sorted = grid_df.sort_values("accuracy", ascending=False).reset_index()
    log.info("\nTop 5 cells by test accuracy:")
    for i, row in grid_df_sorted.head(5).iterrows():
        log.info("  %d. λ=%-13s %-22s acc=%.4f auc=%.4f",
                 i + 1, row["lambda"], row["classifier"], row["accuracy"], row["auc"])

    # Runner-up = highest-accuracy cell that is NOT the headline
    ru_row = grid_df_sorted[~((grid_df_sorted["lambda"] == HEADLINE_LAMBDA) &
                              (grid_df_sorted["classifier"] == HEADLINE_CLASSIFIER))].iloc[0]
    ru_key = f"{ru_row['lambda']}_{ru_row['classifier']}"
    log.info("\nMcNemar's: %s vs %s (acc=%.4f vs %.4f)",
             head_key, ru_key, head_metrics["accuracy"], ru_row["accuracy"])
    mn = mcnemar_test(head_pred["y_test"], head_pred["y_pred"],
                     predictions[ru_key]["y_pred"])
    for k, v in mn.items(): log.info("  %s: %s", k, v)
    pd.DataFrame([{
        "comparison": f"{head_key} vs {ru_key}",
        "headline_acc": head_metrics["accuracy"],
        "runnerup_acc": ru_row["accuracy"],
        **mn,
        "interpretation": ("not significant"
                           if mn["pvalue"] >= 0.05 else "significantly different"),
    }]).to_csv(RESULTS_DIR / f"phase_a_mcnemar_{ts}.csv", index=False)

    # ------------------------ Step 4: Feature-family ablation ------------------------
    log.info("\n" + "=" * 70)
    log.info("Feature-family ablation on headline (MLP+%s)", HEADLINE_LAMBDA)
    log.info("=" * 70)
    train_pkl = FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_hybrid.pkl"
    test_pkl  = FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_hybrid.pkl"
    train_df = pd.read_pickle(train_pkl); test_df = pd.read_pickle(test_pkl)

    abl_configs = [
        ("ALL",              set()),
        ("no_MKT",           {"MKT"}),
        ("no_CNN",           {"CNN"}),
        ("no_Handcrafted",   {"Color", "Texture", "Geometric"}),
        ("MKT_only",         {"CNN", "Color", "Texture", "Geometric", "Unknown"}),
        ("CNN_only",         {"MKT", "Color", "Texture", "Geometric", "Unknown"}),
        ("Handcrafted_only", {"MKT", "CNN", "Unknown"}),
    ]
    abl_rows = []
    for label, drop in abl_configs:
        X_tr, y_tr, cols_tr = split_xy(train_df, drop)
        X_te, y_te, _ = split_xy(test_df, drop)
        if X_tr.shape[1] == 0:
            log.warning("  %s: empty subset, skipping", label); continue
        X_tr_s, X_te_s = fit_select_scale(X_tr, y_tr, X_te)
        clf = build_classifier(HEADLINE_CLASSIFIER)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        y_proba = clf.predict_proba(X_te_s)[:, 1]
        m = compute_metrics(y_te, y_pred, y_proba)
        log.info("  %-18s inputs=%-5d selected=%-3d acc=%.4f auc=%.4f",
                 label, X_tr.shape[1], X_tr_s.shape[1],
                 m["accuracy"], m["auc"])
        abl_rows.append({"config": label, "input_features": X_tr.shape[1],
                         "selected_features": X_tr_s.shape[1],
                         **{k: v for k, v in m.items()
                            if isinstance(v, (int, float))}})
    pd.DataFrame(abl_rows).to_csv(
        RESULTS_DIR / f"phase_a_ablation_{ts}.csv", index=False)

    # ------------------------ Step 5: Human-readable report ------------------------
    report = RESULTS_DIR / f"phase_a_report_{ts}.txt"
    with open(report, "w") as f:
        f.write("Phase A — full 36-cell re-run + headline statistics\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write("PROTOCOL\n")
        f.write("  Feature pickle: features_*_<lambda>_hybrid.pkl (1,758 train rows, flip-only)\n")
        f.write("  Selection: SelectKBest(mutual_info_classif, k=360)\n")
        f.write("  Scaling: RobustScaler fitted on train only\n")
        f.write("  Test set: 323 images (103 BCC + 220 BKL)\n\n")
        f.write(f"HEADLINE: λ={HEADLINE_LAMBDA}, classifier={HEADLINE_CLASSIFIER}\n")
        for k in ["accuracy", "sensitivity", "specificity",
                  "precision", "f1", "auc"]:
            f.write(f"  {k:<12} point={head_metrics[k]:.4f}  "
                    f"95%% CI=[{cis[k][1]:.4f}, {cis[k][2]:.4f}]\n")
        f.write(f"  Confusion: TN={head_metrics['tn']} FP={head_metrics['fp']} "
                f"FN={head_metrics['fn']} TP={head_metrics['tp']}\n\n")
        f.write(f"McNemar's vs runner-up ({ru_key}):\n")
        for k, v in mn.items(): f.write(f"  {k}: {v}\n")
        f.write("\nFEATURE-FAMILY ABLATION:\n")
        f.write(pd.DataFrame(abl_rows).to_string(index=False))
        f.write("\n\nFULL GRID (36 cells):\n")
        f.write(grid_df.to_string(index=False))
        f.write("\n")
    log.info("\n→ %s", report)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
