"""Tier-1 driver: bootstrap CIs + McNemar's test + feature-family ablation.

Resolves the three P0 reviewer concerns identified in REVISION_LOG.md:
    P0-1 — formal classifier-level feature-family ablation
    P0-2 — bootstrap 95% CIs + McNemar's significance test
    (P0-3 — GroupKFold CV — already done in `hyperparameter_tuning.py`)

Operates on the *hybrid* feature pickles (handcrafted + MKT + frozen ResNet-50
CNN), which is the configuration the paper actually claims to evaluate.

Headline configuration:
    λ = DFT (e^{i2πk/N})
    Classifier = MLP (off-the-shelf, hidden_layer_sizes=(100, 50))
    Test set = 323 images (103 BCC + 220 BKL)
    Expected baseline: ~93.19% acc, ~97.36% AUC, sens ≈ spec ≈ 93.2%

Outputs (`paper_pipeline/output/results/`):
    tier1_bootstrap_ci_<timestamp>.csv          per-metric 95% CIs
    tier1_mcnemar_<timestamp>.csv               headline vs runner-up
    tier1_ablation_<timestamp>.csv              6-cell feature-family ablation
    tier1_predictions_<timestamp>.npz           cached MLP predictions for reuse
    tier1_report_<timestamp>.txt                human-readable summary

Run with (from the project root):
    cd "/Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3"
    python paper_pipeline/scripts/tier1_stats_and_ablation.py
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.contingency_tables import mcnemar


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"

HEADLINE_LAMBDA = "dft"               # λ_k = e^{i 2π k / N}
RUNNER_UP_LAMBDA = "low_pass"         # for cross-λ McNemar's (Extra Trees @ low_pass also = 93.19%)
HEADLINE_PICKLE_TRAIN = FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_hybrid.pkl"
HEADLINE_PICKLE_TEST = FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_hybrid.pkl"
RUNNER_UP_PICKLE_TRAIN = FEATURES_DIR / f"features_train_{RUNNER_UP_LAMBDA}_hybrid.pkl"
RUNNER_UP_PICKLE_TEST = FEATURES_DIR / f"features_test_{RUNNER_UP_LAMBDA}_hybrid.pkl"

N_FEATURES = 360
RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
BOOKKEEPING_COLS = {"source_id", "aug_tag", "label"}


def _build_mlp() -> MLPClassifier:
    """The off-the-shelf MLP that produced the 93.19% headline."""
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def _build_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
    )


# ----------------------------------------------------------------------------
# Feature-family classification (used by the ablation)
# ----------------------------------------------------------------------------

MKT_TOKENS = (
    "mdfkt", "krawtchouk", "_mkt_", "real_", "imag_", "magnitude_", "phase_",
    "_real_", "_imag_", "_magnitude_", "_phase_",
)
COLOR_TOKENS = (
    "_rgb_", "_hsv_", "_lab_", "_cielab_", "rgb_", "hsv_", "lab_", "cielab_",
    "channel_mean", "channel_std", "channel_skew", "channel_kurt", "channel_min",
    "channel_max", "channel_entropy", "histogram_", "dominant_color",
    "color_asym", "color_uniform",
)
TEXTURE_TOKENS = (
    "glcm", "lbp_", "gabor", "wavelet", "gradient_", "haralick",
    "_contrast", "_dissimilarity", "_homogeneity", "_energy_", "_correlation_",
)
GEOMETRIC_TOKENS = (
    "area", "perimeter", "eccentric", "circularity", "compactness", "convexity",
    "diameter", "asymmetry", "fractal", "curvature", "smoothness", "hu_",
    "hu_moment",
)
CNN_TOKENS = (
    "resnet", "cnn_", "_cnn", "deep_feat", "_emb_", "embedding_",
    "efficientnet", "densenet",
)


def classify_feature(name: str) -> str:
    s = name.lower()
    if any(t in s for t in MKT_TOKENS): return "MKT"
    if any(t in s for t in CNN_TOKENS): return "CNN"
    if any(t in s for t in COLOR_TOKENS): return "Color"
    if any(t in s for t in TEXTURE_TOKENS): return "Texture"
    if any(t in s for t in GEOMETRIC_TOKENS): return "Geometric"
    return "Unknown"


def family_breakdown(feature_names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"MKT": [], "CNN": [], "Color": [],
                                 "Texture": [], "Geometric": [], "Unknown": []}
    for n in feature_names:
        out[classify_feature(n)].append(n)
    return out


# ----------------------------------------------------------------------------
# Core pipeline (load → select → scale → train → predict)
# ----------------------------------------------------------------------------

def _split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, cols


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_subset: list[str] | None,
    classifier_name: str = "MLP",
    n_features: int = N_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    """Returns (y_test, y_pred, y_proba, metrics_dict, selected_feature_names)."""
    if feature_subset is not None:
        keep = list(feature_subset) + ["label"]
        # restrict columns (also drops the other bookkeeping fields)
        train_df = train_df[[c for c in train_df.columns if c in keep]]
        test_df  = test_df[[c for c in test_df.columns if c in keep]]

    X_train, y_train, train_cols = _split_xy(train_df)
    X_test, y_test, _ = _split_xy(test_df)

    # SelectKBest with MI — fit on train ONLY
    k = min(n_features, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k).fit(X_train, y_train)
    keep_idx = selector.get_support(indices=True)
    selected_cols = [train_cols[i] for i in keep_idx]
    X_train_s = selector.transform(X_train)
    X_test_s = selector.transform(X_test)

    scaler = RobustScaler().fit(X_train_s)
    X_train_s = scaler.transform(X_train_s)
    X_test_s = scaler.transform(X_test_s)

    if classifier_name == "MLP":
        clf = _build_mlp()
    elif classifier_name == "Extra Trees":
        clf = _build_extra_trees()
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    return y_test, y_pred, y_proba, metrics, selected_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0.0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0.0),
        "f1": f1_score(y_true, y_pred, zero_division=0.0),
        "auc": roc_auc_score(y_true, y_proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ----------------------------------------------------------------------------
# Bootstrap confidence intervals
# ----------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics_per_boot = {k: [] for k in
                        ["accuracy", "sensitivity", "specificity",
                         "precision", "f1", "auc"]}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; yp = y_pred[idx]; ypr = y_proba[idx]
        # Skip iterations where the resample misses a whole class
        if len(np.unique(yt)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        metrics_per_boot["accuracy"].append((tn + tp) / (tn + tp + fn + fp))
        metrics_per_boot["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics_per_boot["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics_per_boot["precision"].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics_per_boot["f1"].append(
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )
        try:
            metrics_per_boot["auc"].append(roc_auc_score(yt, ypr))
        except ValueError:
            pass

    out = {}
    for k, vals in metrics_per_boot.items():
        arr = np.array(vals)
        lo = np.percentile(arr, 100 * alpha / 2)
        hi = np.percentile(arr, 100 * (1 - alpha / 2))
        out[k] = (float(np.mean(arr)), float(lo), float(hi))
    return out


# ----------------------------------------------------------------------------
# McNemar's test
# ----------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """McNemar's test on paired predictions A vs B against the same y_true."""
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    n00 = int(np.sum(~a_correct & ~b_correct))   # both wrong
    n01 = int(np.sum(~a_correct & b_correct))    # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))    # A right, B wrong
    n11 = int(np.sum(a_correct & b_correct))     # both right

    table = [[n11, n10], [n01, n00]]
    # exact=True if n10+n01 < 25, else use approximate chi-squared with continuity correction
    use_exact = (n10 + n01) < 25
    result = mcnemar(table, exact=use_exact, correction=True)

    return {
        "n_both_correct": n11,
        "n_A_only_correct": n10,
        "n_B_only_correct": n01,
        "n_both_wrong": n00,
        "method": "exact_binomial" if use_exact else "chi2_with_continuity",
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("tier1")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Tier-1 driver: bootstrap CIs + McNemar + feature-family ablation")
    log.info("=" * 70)

    # ---------- Load hybrid pickles ----------
    log.info("Loading hybrid feature pickles...")
    train_df = pd.read_pickle(HEADLINE_PICKLE_TRAIN)
    test_df = pd.read_pickle(HEADLINE_PICKLE_TEST)
    log.info("  train: %d rows × %d cols  test: %d rows × %d cols",
             len(train_df), train_df.shape[1], len(test_df), test_df.shape[1])

    train_df_ru = pd.read_pickle(RUNNER_UP_PICKLE_TRAIN)
    test_df_ru = pd.read_pickle(RUNNER_UP_PICKLE_TEST)

    # ---------- Headline run: MLP on dft hybrid ----------
    log.info("")
    log.info("Headline: MLP on λ=%s (full 2,509-feature hybrid → top-%d MI)",
             HEADLINE_LAMBDA, N_FEATURES)
    t0 = time.time()
    y_test, y_pred_head, y_proba_head, head_metrics, selected_head = \
        train_and_predict(train_df, test_df, feature_subset=None,
                          classifier_name="MLP", n_features=N_FEATURES)
    log.info("  trained in %.1fs", time.time() - t0)
    for k, v in head_metrics.items():
        if isinstance(v, float):
            log.info("    %-12s %.4f", k, v)
        else:
            log.info("    %-12s %d", k, v)

    # Save headline predictions
    np.savez(RESULTS_DIR / f"tier1_predictions_{ts}.npz",
             y_test=y_test, y_pred=y_pred_head, y_proba=y_proba_head,
             classifier="MLP", lambda_name=HEADLINE_LAMBDA,
             features="hybrid")

    # ---------- Family breakdown of the 360 selected features ----------
    fb = family_breakdown(selected_head)
    log.info("")
    log.info("Family breakdown of selected features:")
    for fam in ["MKT", "CNN", "Color", "Texture", "Geometric", "Unknown"]:
        log.info("  %-10s %d", fam, len(fb[fam]))

    # ---------- Bootstrap CIs ----------
    log.info("")
    log.info("Bootstrap CIs (%d resamples)...", N_BOOTSTRAP)
    cis = bootstrap_ci(y_test, y_pred_head, y_proba_head, n_boot=N_BOOTSTRAP)
    boot_rows = []
    for metric, (mean, lo, hi) in cis.items():
        boot_rows.append({"metric": metric, "point_estimate": head_metrics[metric],
                          "bootstrap_mean": mean,
                          "ci95_low": lo, "ci95_high": hi})
        log.info("  %-12s point=%.4f mean=%.4f CI95=[%.4f, %.4f]",
                 metric, head_metrics[metric], mean, lo, hi)

    boot_csv = RESULTS_DIR / f"tier1_bootstrap_ci_{ts}.csv"
    pd.DataFrame(boot_rows).to_csv(boot_csv, index=False)
    log.info("  → %s", boot_csv)

    # ---------- Runner-up: Extra Trees on low_pass hybrid ----------
    log.info("")
    log.info("Runner-up: Extra Trees on λ=%s (for McNemar comparison)",
             RUNNER_UP_LAMBDA)
    _, y_pred_ru, y_proba_ru, ru_metrics, _ = \
        train_and_predict(train_df_ru, test_df_ru, feature_subset=None,
                          classifier_name="Extra Trees", n_features=N_FEATURES)
    for k, v in ru_metrics.items():
        if isinstance(v, float):
            log.info("    %-12s %.4f", k, v)

    # ---------- McNemar's test ----------
    log.info("")
    log.info("McNemar's test (headline MLP+%s vs runner-up Extra Trees+%s)...",
             HEADLINE_LAMBDA, RUNNER_UP_LAMBDA)
    mn = mcnemar_test(y_test, y_pred_head, y_pred_ru)
    for k, v in mn.items():
        log.info("  %-22s %s", k, v)

    mn_csv = RESULTS_DIR / f"tier1_mcnemar_{ts}.csv"
    pd.DataFrame([{
        "comparison": f"MLP+{HEADLINE_LAMBDA} vs ExtraTrees+{RUNNER_UP_LAMBDA}",
        **mn,
        "interpretation": (
            "no significant difference"
            if mn["pvalue"] >= 0.05 else "significantly different"
        ),
    }]).to_csv(mn_csv, index=False)
    log.info("  → %s", mn_csv)

    # ---------- Feature-family ablation (P0-1) ----------
    log.info("")
    log.info("=" * 70)
    log.info("Feature-family ablation (MLP, λ=%s)", HEADLINE_LAMBDA)
    log.info("=" * 70)

    train_cols = [c for c in train_df.columns if c not in BOOKKEEPING_COLS]
    fb_all = family_breakdown(train_cols)
    handcrafted = fb_all["Color"] + fb_all["Texture"] + fb_all["Geometric"]
    mkt = fb_all["MKT"]
    cnn = fb_all["CNN"]
    log.info("  Total features: %d (MKT=%d, CNN=%d, Handcrafted=%d, Unknown=%d)",
             len(train_cols), len(mkt), len(cnn), len(handcrafted),
             len(fb_all["Unknown"]))

    ablation_configs = [
        ("ALL",            train_cols),
        ("no_MKT",         [c for c in train_cols if c not in mkt]),
        ("no_CNN",         [c for c in train_cols if c not in cnn]),
        ("no_Handcrafted", [c for c in train_cols if c not in handcrafted]),
        ("MKT_only",       mkt),
        ("CNN_only",       cnn),
        ("Handcrafted_only", handcrafted),
    ]

    ablation_rows = []
    for label, subset in ablation_configs:
        if len(subset) == 0:
            log.warning("  Skipping %s (empty subset)", label)
            continue
        log.info("")
        log.info("  Config: %s (%d input features)", label, len(subset))
        try:
            _, _, _, m, sel = train_and_predict(
                train_df, test_df,
                feature_subset=subset,
                classifier_name="MLP",
                n_features=N_FEATURES,
            )
            ablation_rows.append({
                "config": label,
                "input_features": len(subset),
                "selected_features": len(sel),
                **{k: v for k, v in m.items() if isinstance(v, (int, float))},
            })
            log.info("    acc=%.4f  sens=%.4f  spec=%.4f  auc=%.4f",
                     m["accuracy"], m["sensitivity"],
                     m["specificity"], m["auc"])
        except Exception as e:
            log.error("    FAILED: %s", e)
            ablation_rows.append({
                "config": label, "input_features": len(subset),
                "selected_features": 0, "error": str(e),
            })

    abl_csv = RESULTS_DIR / f"tier1_ablation_{ts}.csv"
    pd.DataFrame(ablation_rows).to_csv(abl_csv, index=False)
    log.info("")
    log.info("Ablation results → %s", abl_csv)

    # ---------- Human-readable report ----------
    report = RESULTS_DIR / f"tier1_report_{ts}.txt"
    with open(report, "w") as f:
        f.write("Tier-1 results report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write("Headline config: MLP on hybrid features, λ=DFT\n")
        f.write("  - Accuracy   : %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["accuracy"], cis["accuracy"][1], cis["accuracy"][2]))
        f.write("  - Sensitivity: %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["sensitivity"], cis["sensitivity"][1], cis["sensitivity"][2]))
        f.write("  - Specificity: %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["specificity"], cis["specificity"][1], cis["specificity"][2]))
        f.write("  - Precision  : %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["precision"], cis["precision"][1], cis["precision"][2]))
        f.write("  - F1         : %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["f1"], cis["f1"][1], cis["f1"][2]))
        f.write("  - AUC        : %.4f  (95%% CI: [%.4f, %.4f])\n" %
                (head_metrics["auc"], cis["auc"][1], cis["auc"][2]))
        f.write("\n")
        f.write("McNemar's test vs Extra Trees + λ=low_pass:\n")
        for k, v in mn.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("Feature-family ablation (MLP):\n")
        f.write(pd.DataFrame(ablation_rows).to_string(index=False))
        f.write("\n")
    log.info("Report → %s", report)
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
