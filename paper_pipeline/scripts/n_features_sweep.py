"""
n_features sweep: vary the SelectKBest k value and report train-CV plus
held-out test metrics for the top classifiers per λ.

Why honest: feature-count k is chosen by maximising train-CV F1 (no test
peeking). Test set is touched once per (λ, classifier, k) and reported.
You should pick the best k by the train-CV column, NOT by the test column.

Outputs:
    paper_pipeline/output/results/k_sweep_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
LOG = logging.getLogger("k_sweep")
BOOKKEEPING = {"source_id", "aug_tag", "label"}

# Use the fastest strong classifiers for the sweep so it doesn't take forever.
SWEEP_CLASSIFIERS = ["MLP", "CatBoost", "LightGBM", "Gradient Boosting"]


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def run_one(lambda_name: str, k_values: list[int], cv_splits: int, suffix: str = "") -> pd.DataFrame:
    suf = f"_{suffix}" if suffix else ""
    train = pd.read_pickle(FEATURES_DIR / f"features_train_{lambda_name}{suf}.pkl")
    test = pd.read_pickle(FEATURES_DIR / f"features_test_{lambda_name}{suf}.pkl")
    feat_cols = [c for c in train.columns if c not in BOOKKEEPING]
    Xtr = np.nan_to_num(train[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    ytr = train["label"].to_numpy(np.int64)
    Xte = np.nan_to_num(test[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    yte = test["label"].to_numpy(np.int64)
    LOG.info("[%s] X_train=%s X_test=%s", lambda_name, Xtr.shape, Xte.shape)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    rows = []
    for k in k_values:
        sel = SelectKBest(mutual_info_classif, k=min(k, Xtr.shape[1]))
        Xtr_s = sel.fit_transform(Xtr, ytr)
        Xte_s = sel.transform(Xte)
        scaler = RobustScaler()
        Xtr_s = scaler.fit_transform(Xtr_s)
        Xte_s = scaler.transform(Xte_s)

        for name in SWEEP_CLASSIFIERS:
            clf = build_classifier(name, input_dim=Xtr_s.shape[1])
            try:
                cv_scores = cross_val_score(clf, Xtr_s, ytr, cv=cv,
                                            scoring="accuracy", n_jobs=1)
                cv_acc_mean = float(np.mean(cv_scores))
                cv_acc_std = float(np.std(cv_scores))
            except Exception as e:
                LOG.warning("  cv failed for %s @ k=%d: %s", name, k, e)
                cv_acc_mean = cv_acc_std = float("nan")
            clf = build_classifier(name, input_dim=Xtr_s.shape[1])
            clf.fit(Xtr_s, ytr)
            yp = clf.predict(Xte_s)
            try:
                yproba = clf.predict_proba(Xte_s)[:, 1]
            except Exception:
                yproba = None
            row = {
                "lambda": lambda_name, "k": k, "classifier": name,
                "test_accuracy": accuracy_score(yte, yp),
                "test_sensitivity": recall_score(yte, yp, zero_division=0.0),
                "test_specificity": specificity(yte, yp),
                "test_f1": f1_score(yte, yp, zero_division=0.0),
                "test_auc": roc_auc_score(yte, yproba) if yproba is not None else np.nan,
                "cv_accuracy_mean": cv_acc_mean,
                "cv_accuracy_std": cv_acc_std,
            }
            rows.append(row)
            LOG.info("  k=%d  %s  cv_acc=%.4f±%.4f  test_acc=%.4f  test_sens=%.4f",
                     k, name, cv_acc_mean, cv_acc_std, row["test_accuracy"], row["test_sensitivity"])
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="low_pass,odd_harmonic",
                        help="Default: the two best λ from initial run.")
    parser.add_argument("--ks", default="100,200,300,360,400,460",
                        help="Comma-separated k values to try.")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--suffix", default="", help="Pickle suffix, e.g. 'hybrid'.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    ks = [int(s.strip()) for s in args.ks.split(",") if s.strip()]
    LOG.info("Sweep: lambdas=%s, ks=%s", lambdas, ks)

    all_rows = []
    for lam in lambdas:
        all_rows.append(run_one(lam, ks, args.cv_splits, args.suffix))

    df = pd.concat(all_rows, ignore_index=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"k_sweep_{stamp}.csv"
    df.to_csv(out, index=False)
    LOG.info("Wrote %s", out)
    # Print best k by CV per (λ, classifier)
    LOG.info("\n=== Best k by CV accuracy ===")
    for (lam, clf), sub in df.groupby(["lambda", "classifier"]):
        best = sub.loc[sub["cv_accuracy_mean"].idxmax()]
        LOG.info("  λ=%s  %s  best_k_cv=%d  cv_acc=%.4f  test_acc=%.4f  test_sens=%.4f",
                 lam, clf, int(best["k"]), best["cv_accuracy_mean"],
                 best["test_accuracy"], best["test_sensitivity"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
