"""Tier-1 V2 driver — bootstrap CIs + McNemar's + ablation for the
cross-feature STACKING ensemble headline.

Headline configuration (verified from cross_stack_20260518_111637.csv):
    λ = low_pass  (λ_k = 1/(k+1))
    Base classifiers (6):
        hybrid:MLP, hybrid:CatBoost, hybrid:LightGBM, hybrid:Gradient Boosting,
        fthybrid:MLP, fthybrid:Extra Trees
    Meta-learner: Logistic Regression
    Training data: flip-only, 1,758 samples (879 BCC + 879 BKL)
    Test set: 323 samples
    Expected: Acc 93.81%, Sens 91.26%, Spec 95.00%, F1 90.38%, AUC 96.91%

This script:
    1) Fits each base classifier with 5-fold CV-stacking on the training set,
       producing out-of-fold predictions used to train the meta-learner.
    2) Fits each base classifier on the full training set and predicts on test.
    3) Combines base test predictions via the trained meta-learner.
    4) Computes bootstrap 95% CIs (10k resamples) on the 6 headline metrics.
    5) Runs McNemar's test against the runner-up (CatBoost+odd-harmonic from
       same flip-only canonical CSV).
    6) Runs feature-family ablation by removing each family (MKT, CNN, hand-
       crafted) from BOTH hybrid and fthybrid pickles and re-training the full
       stacking ensemble.

Outputs:
    tier1_stacking_predictions_<ts>.npz
    tier1_stacking_bootstrap_ci_<ts>.csv
    tier1_stacking_mcnemar_<ts>.csv
    tier1_stacking_ablation_<ts>.csv
    tier1_stacking_report_<ts>.txt
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.contingency_tables import mcnemar
import catboost as cb
import lightgbm as lgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"

HEADLINE_LAMBDA = "low_pass"
RUNNER_UP_LAMBDA = "odd_harmonic"
RUNNER_UP_CLASSIFIER = "CatBoost"   # 92.57% in flip-only canonical CSV
N_FEATURES = 360
RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
N_CV_FOLDS = 5
BOOKKEEPING_COLS = {"source_id", "aug_tag", "label"}


# ----------------------------------------------------------------------------
# Base classifier builders (matching practical project conventions)
# ----------------------------------------------------------------------------

def build_classifier(name: str) -> object:
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50), alpha=0.001,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
    if name == "CatBoost":
        return cb.CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=False, random_state=RANDOM_STATE,
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1,
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
    raise ValueError(f"Unknown classifier: {name}")


# ----------------------------------------------------------------------------
# Feature-family classification
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
# Core pipeline
# ----------------------------------------------------------------------------

def split_xy(df: pd.DataFrame, drop_families: set[str] | None = None
             ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    if drop_families:
        cols = [c for c in cols if classify_feature(c) not in drop_families]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, cols


def fit_select_scale(X_train, y_train, X_test, n_features=N_FEATURES):
    k = min(n_features, X_train.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k).fit(X_train, y_train)
    X_train_s = sel.transform(X_train)
    X_test_s = sel.transform(X_test)
    sca = RobustScaler().fit(X_train_s)
    X_train_s = sca.transform(X_train_s)
    X_test_s = sca.transform(X_test_s)
    return X_train_s, X_test_s


def get_oof_and_test_proba(clf_name, X_train, y_train, X_test, n_folds=N_CV_FOLDS):
    """Out-of-fold predictions on training set (for meta-learner training) +
    final-model predictions on test set (for meta-learner application)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train))
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = build_classifier(clf_name)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        oof[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]
    final_clf = build_classifier(clf_name)
    final_clf.fit(X_train, y_train)
    test_proba = final_clf.predict_proba(X_test)[:, 1]
    return oof, test_proba


def fit_stacking_ensemble(
    train_hybrid: pd.DataFrame, test_hybrid: pd.DataFrame,
    train_fthybrid: pd.DataFrame, test_fthybrid: pd.DataFrame,
    drop_families: set[str] | None = None,
    log: logging.Logger | None = None,
):
    """Fit the 6-base + 1-meta stacking ensemble and return test predictions."""
    L = log or logging.getLogger("stacker")

    # Restrict columns per family-drop request (applied to BOTH hybrid pickles)
    X_h_tr, y_tr, _ = split_xy(train_hybrid, drop_families)
    X_h_te, y_te, _ = split_xy(test_hybrid, drop_families)
    X_ft_tr, _, _ = split_xy(train_fthybrid, drop_families)
    X_ft_te, _, _ = split_xy(test_fthybrid, drop_families)

    # Apply k-best + RobustScaler on each feature view independently
    X_h_tr_s, X_h_te_s = fit_select_scale(X_h_tr, y_tr, X_h_te)
    X_ft_tr_s, X_ft_te_s = fit_select_scale(X_ft_tr, y_tr, X_ft_te)

    L.info("  Hybrid view: %d → %d features  |  FT-hybrid view: %d → %d features",
           X_h_tr.shape[1], X_h_tr_s.shape[1],
           X_ft_tr.shape[1], X_ft_tr_s.shape[1])

    stack_components = [
        ("hybrid:MLP",                "MLP",               X_h_tr_s, X_h_te_s),
        ("hybrid:CatBoost",           "CatBoost",          X_h_tr_s, X_h_te_s),
        ("hybrid:LightGBM",           "LightGBM",          X_h_tr_s, X_h_te_s),
        ("hybrid:Gradient Boosting",  "Gradient Boosting", X_h_tr_s, X_h_te_s),
        ("fthybrid:MLP",              "MLP",               X_ft_tr_s, X_ft_te_s),
        ("fthybrid:Extra Trees",      "Extra Trees",       X_ft_tr_s, X_ft_te_s),
    ]

    oofs, test_probas, names = [], [], []
    for tag, clf_name, X_tr, X_te in stack_components:
        L.info("  Fitting %s …", tag)
        oof, tp = get_oof_and_test_proba(clf_name, X_tr, y_tr, X_te)
        oofs.append(oof); test_probas.append(tp); names.append(tag)

    OOF = np.column_stack(oofs)       # (n_train, 6)
    TEST = np.column_stack(test_probas)  # (n_test, 6)

    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(OOF, y_tr)
    y_proba_stack = meta.predict_proba(TEST)[:, 1]
    y_pred_stack = (y_proba_stack >= 0.5).astype(int)

    return y_te, y_pred_stack, y_proba_stack, names, meta.coef_.flatten()


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "sensitivity":  recall_score(y_true, y_pred, zero_division=0.0),
        "specificity":  tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision":    precision_score(y_true, y_pred, zero_division=0.0),
        "f1":           f1_score(y_true, y_pred, zero_division=0.0),
        "auc":          roc_auc_score(y_true, y_proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    metrics = {k: [] for k in
               ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        acc = (tn + tp) / (tn + tp + fp + fn) if (tn + tp + fp + fn) else 0.0
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        metrics["accuracy"].append(acc)
        metrics["sensitivity"].append(sens)
        metrics["specificity"].append(spec)
        metrics["precision"].append(prec)
        metrics["f1"].append(f1)
        try: metrics["auc"].append(roc_auc_score(yt, ypr))
        except ValueError: pass
    out = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        out[k] = (float(np.mean(arr)),
                  float(np.percentile(arr, 100 * alpha / 2)),
                  float(np.percentile(arr, 100 * (1 - alpha / 2))))
    return out


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    a_ok = (y_pred_a == y_true); b_ok = (y_pred_b == y_true)
    n00 = int(np.sum(~a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    n11 = int(np.sum(a_ok & b_ok))
    use_exact = (n01 + n10) < 25
    result = mcnemar([[n11, n10], [n01, n00]],
                     exact=use_exact, correction=True)
    return {
        "n_both_correct": n11, "n_A_only_correct": n10,
        "n_B_only_correct": n01, "n_both_wrong": n00,
        "method": "exact_binomial" if use_exact else "chi2_with_continuity",
        "statistic": float(result.statistic), "pvalue": float(result.pvalue),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("tier1_stack")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Tier-1 V2: bootstrap CIs + McNemar + ablation for STACKING")
    log.info("Headline: 6-base + LR meta on λ=%s, flip-only", HEADLINE_LAMBDA)
    log.info("=" * 70)

    # ---------- Load all 4 feature pickles ----------
    log.info("Loading hybrid + fthybrid pickles for λ=%s ...", HEADLINE_LAMBDA)
    train_hybrid   = pd.read_pickle(FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_hybrid.pkl")
    test_hybrid    = pd.read_pickle(FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_hybrid.pkl")
    train_fthybrid = pd.read_pickle(FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_fthybrid.pkl")
    test_fthybrid  = pd.read_pickle(FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_fthybrid.pkl")
    log.info("  hybrid    train=%s  test=%s", train_hybrid.shape, test_hybrid.shape)
    log.info("  fthybrid  train=%s  test=%s", train_fthybrid.shape, test_fthybrid.shape)

    # ---------- Headline: full stacking ensemble ----------
    log.info("")
    log.info("Fitting the 6-base + LR-meta stacking ensemble ...")
    t0 = time.time()
    y_test, y_pred_stack, y_proba_stack, names, meta_coef = \
        fit_stacking_ensemble(train_hybrid, test_hybrid,
                              train_fthybrid, test_fthybrid, log=log)
    log.info("  Done in %.1fs", time.time() - t0)

    metrics = compute_metrics(y_test, y_pred_stack, y_proba_stack)
    log.info("")
    log.info("Stacking headline metrics:")
    for k in ["accuracy", "sensitivity", "specificity",
              "precision", "f1", "auc"]:
        log.info("  %-12s %.4f", k, metrics[k])
    log.info("  confusion: TN=%d FP=%d FN=%d TP=%d",
             metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
    log.info("")
    log.info("Meta-learner LR coefficients:")
    for n, c in zip(names, meta_coef):
        log.info("  %-30s %+.3f", n, c)

    np.savez(RESULTS_DIR / f"tier1_stacking_predictions_{ts}.npz",
             y_test=y_test, y_pred=y_pred_stack, y_proba=y_proba_stack,
             stack_components=names, meta_coef=meta_coef)

    # ---------- Bootstrap CIs ----------
    log.info("")
    log.info("Bootstrap CIs (%d resamples)...", N_BOOTSTRAP)
    cis = bootstrap_ci(y_test, y_pred_stack, y_proba_stack)
    rows = []
    for k, (m, lo, hi) in cis.items():
        rows.append({"metric": k, "point_estimate": metrics[k],
                     "bootstrap_mean": m, "ci95_low": lo, "ci95_high": hi})
        log.info("  %-12s point=%.4f mean=%.4f CI95=[%.4f, %.4f]",
                 k, metrics[k], m, lo, hi)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"tier1_stacking_bootstrap_ci_{ts}.csv",
                              index=False)

    # ---------- McNemar's: stacking vs runner-up ----------
    log.info("")
    log.info("Runner-up: %s + λ=%s (single classifier)",
             RUNNER_UP_CLASSIFIER, RUNNER_UP_LAMBDA)
    # Train the runner-up from scratch on its own hybrid pickle
    train_ru = pd.read_pickle(FEATURES_DIR / f"features_train_{RUNNER_UP_LAMBDA}_hybrid.pkl")
    test_ru  = pd.read_pickle(FEATURES_DIR / f"features_test_{RUNNER_UP_LAMBDA}_hybrid.pkl")
    X_tr, y_tr_ru, _ = split_xy(train_ru)
    X_te, y_te_ru, _ = split_xy(test_ru)
    X_tr_s, X_te_s = fit_select_scale(X_tr, y_tr_ru, X_te)
    clf_ru = build_classifier(RUNNER_UP_CLASSIFIER)
    clf_ru.fit(X_tr_s, y_tr_ru)
    y_pred_ru = clf_ru.predict(X_te_s)
    metrics_ru = compute_metrics(y_te_ru, y_pred_ru,
                                  clf_ru.predict_proba(X_te_s)[:, 1])
    log.info("  Runner-up acc=%.4f  auc=%.4f",
             metrics_ru["accuracy"], metrics_ru["auc"])

    # Sanity: y_test (stacking) and y_te_ru (runner-up) must match — same test split
    assert np.array_equal(y_test, y_te_ru), \
        "Test labels differ — pickles use different splits!"

    mn = mcnemar_test(y_test, y_pred_stack, y_pred_ru)
    log.info("")
    log.info("McNemar's test (stacking vs %s+%s):", RUNNER_UP_CLASSIFIER, RUNNER_UP_LAMBDA)
    for k, v in mn.items(): log.info("  %-22s %s", k, v)

    pd.DataFrame([{
        "comparison": f"stacking_{HEADLINE_LAMBDA} vs {RUNNER_UP_CLASSIFIER}_{RUNNER_UP_LAMBDA}",
        **mn,
        "interpretation": "no significant difference"
        if mn["pvalue"] >= 0.05 else "significantly different",
    }]).to_csv(RESULTS_DIR / f"tier1_stacking_mcnemar_{ts}.csv", index=False)

    # ---------- Feature-family ablation ----------
    log.info("")
    log.info("=" * 70)
    log.info("Feature-family ablation for the stacking ensemble")
    log.info("=" * 70)
    abl_rows = []
    ablation_configs = [
        ("ALL", set()),
        ("no_MKT", {"MKT"}),
        ("no_CNN", {"CNN"}),
        ("no_Handcrafted", {"Color", "Texture", "Geometric"}),
        ("MKT_only", {"CNN", "Color", "Texture", "Geometric", "Unknown"}),
        ("CNN_only", {"MKT", "Color", "Texture", "Geometric", "Unknown"}),
        ("Handcrafted_only", {"MKT", "CNN", "Unknown"}),
    ]
    for label, drop in ablation_configs:
        log.info("")
        log.info("Ablation: %s (drop=%s)", label, drop or "{}")
        t0 = time.time()
        try:
            _, ypa, ypra, _, _ = fit_stacking_ensemble(
                train_hybrid, test_hybrid,
                train_fthybrid, test_fthybrid,
                drop_families=drop, log=log,
            )
            m = compute_metrics(y_test, ypa, ypra)
            abl_rows.append({"config": label,
                             **{k: v for k, v in m.items()
                                if isinstance(v, (int, float))},
                             "elapsed_s": round(time.time() - t0, 1)})
            log.info("  → acc=%.4f sens=%.4f spec=%.4f auc=%.4f (%.0fs)",
                     m["accuracy"], m["sensitivity"], m["specificity"],
                     m["auc"], time.time() - t0)
        except Exception as e:
            log.error("  FAILED: %s", e)
            abl_rows.append({"config": label, "error": str(e),
                             "elapsed_s": round(time.time() - t0, 1)})
    pd.DataFrame(abl_rows).to_csv(RESULTS_DIR / f"tier1_stacking_ablation_{ts}.csv",
                                  index=False)

    # ---------- Report ----------
    report = RESULTS_DIR / f"tier1_stacking_report_{ts}.txt"
    with open(report, "w") as f:
        f.write("Tier-1 V2 stacking-ensemble results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write("Headline: 6-base + LR-meta stacking, λ=low_pass, flip-only\n")
        f.write(f"  Base classifiers: {names}\n")
        f.write("  Meta-learner: Logistic Regression\n\n")
        for k in ["accuracy", "sensitivity", "specificity",
                  "precision", "f1", "auc"]:
            f.write("  %-12s point=%.4f  95%%CI=[%.4f, %.4f]\n" %
                    (k, metrics[k], cis[k][1], cis[k][2]))
        f.write("\nConfusion: TN=%d FP=%d FN=%d TP=%d\n" %
                (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]))
        f.write("\nMcNemar's test vs %s+%s:\n" %
                (RUNNER_UP_CLASSIFIER, RUNNER_UP_LAMBDA))
        for k, v in mn.items(): f.write(f"  {k}: {v}\n")
        f.write("\nFeature-family ablation (full stacking ensemble):\n")
        f.write(pd.DataFrame(abl_rows).to_string(index=False))
        f.write("\n")
    log.info("")
    log.info("Report → %s", report)
    log.info("Done.")


if __name__ == "__main__":
    main()
