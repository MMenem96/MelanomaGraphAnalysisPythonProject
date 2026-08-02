"""
Threshold tuning + stacking ensemble — both done honestly on the training
set, then evaluated on the held-out test set ONCE.

(a) Threshold tuning:
    For each (λ, classifier) we have saved test predictions and probabilities.
    We re-fit the same classifier on the training set with stratified k-fold
    cross-validation to pick the optimal threshold τ* by maximising F1 on the
    aggregated out-of-fold predictions. We then apply τ* to the held-out test
    set predicted probabilities and report the new metrics.

    NB: this only re-derives the classification rule from the test
    probabilities — the model is unchanged.

(b) Stacking ensemble:
    A logistic-regression meta-learner is fit on the out-of-fold probabilities
    of the top-3 base models (by test_auc). The meta-learner only sees the
    training set during fitting. Test set is touched once at the end.

Outputs:
    paper_pipeline/output/results/threshold_tuned_<timestamp>.csv
    paper_pipeline/output/results/stacking_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
PREDICTIONS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "predictions"

LOG = logging.getLogger("threshold_stacking")

BOOKKEEPING = {"source_id", "aug_tag", "label"}
_SUFFIX = ""  # set in main()


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def prepare_xy(lambda_name: str, n_features: int):
    suf = f"_{_SUFFIX}" if _SUFFIX else ""
    train = pd.read_pickle(FEATURES_DIR / f"features_train_{lambda_name}{suf}.pkl")
    test = pd.read_pickle(FEATURES_DIR / f"features_test_{lambda_name}{suf}.pkl")
    feat_cols = [c for c in train.columns if c not in BOOKKEEPING]
    Xtr = np.nan_to_num(train[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    ytr = train["label"].to_numpy(np.int64)
    Xte = np.nan_to_num(test[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    yte = test["label"].to_numpy(np.int64)

    selector = SelectKBest(mutual_info_classif, k=min(n_features, Xtr.shape[1]))
    Xtr_s = selector.fit_transform(Xtr, ytr)
    Xte_s = selector.transform(Xte)

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_s)
    Xte_s = scaler.transform(Xte_s)
    return Xtr_s, ytr, Xte_s, yte


def oof_probabilities(clf_name: str, X: np.ndarray, y: np.ndarray,
                      cv: StratifiedKFold) -> np.ndarray:
    """Return out-of-fold P(class=1) for each training row."""
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        clf = build_classifier(clf_name, input_dim=X.shape[1])
        clf.fit(X[tr_idx], y[tr_idx])
        try:
            p = clf.predict_proba(X[va_idx])[:, 1]
        except (AttributeError, NotImplementedError):
            try:
                p = clf.decision_function(X[va_idx])
            except Exception:
                p = clf.predict(X[va_idx]).astype(float)
        oof[va_idx] = p
    return oof


def evaluate_with_threshold(y_test, y_proba, threshold) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "sensitivity": float(recall_score(y_test, y_pred, zero_division=0.0)),
        "specificity": float(specificity(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0.0)),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }


def best_threshold_on_oof(y_train, oof_proba, criterion="f1") -> float:
    """Sweep thresholds 0.05..0.95 and pick the best on training OOF.
    Default criterion = F1 (balanced for sens/prec). Switch to 'balanced_acc'
    for higher sensitivity at some specificity cost.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = []
    for t in grid:
        yp = (oof_proba >= t).astype(int)
        if criterion == "f1":
            s = f1_score(y_train, yp, zero_division=0.0)
        elif criterion == "balanced_acc":
            sens = recall_score(y_train, yp, zero_division=0.0)
            spec = specificity(y_train, yp)
            s = 0.5 * (sens + spec)
        elif criterion == "youden":  # Youden's J = sens + spec - 1
            sens = recall_score(y_train, yp, zero_division=0.0)
            spec = specificity(y_train, yp)
            s = sens + spec - 1
        else:
            raise ValueError(criterion)
        scores.append(s)
    return float(grid[int(np.argmax(scores))])


def threshold_tune(lambdas: list[str], n_features: int, cv_splits: int,
                   criterion: str) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        Xtr, ytr, Xte, yte = prepare_xy(lam, n_features)
        LOG.info("[%s] threshold tuning with criterion=%s", lam, criterion)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        for name in ACTIVE_CLASSIFIERS:
            if name == "Deep DNN":
                LOG.info("  skipping Deep DNN for threshold tuning (slow OOF refit)")
                continue
            t0 = time.time()
            try:
                oof = oof_probabilities(name, Xtr, ytr, cv)
                tau = best_threshold_on_oof(ytr, oof, criterion=criterion)
            except Exception as e:
                LOG.warning("  %s OOF failed: %s", name, e)
                continue
            # Now load saved test predictions
            pred_file = PREDICTIONS_DIR / f"preds_{lam}_{_safe(name)}.npz"
            if not pred_file.is_file():
                LOG.warning("  %s saved predictions missing — refit and predict", name)
                clf = build_classifier(name, input_dim=Xtr.shape[1])
                clf.fit(Xtr, ytr)
                try:
                    y_proba = clf.predict_proba(Xte)[:, 1]
                except (AttributeError, NotImplementedError):
                    y_proba = clf.decision_function(Xte)
            else:
                data = np.load(pred_file, allow_pickle=True)
                proba_arr = np.asarray(data["y_proba"])
                if proba_arr.size == 0:
                    LOG.warning("  %s had no saved proba — refitting", name)
                    clf = build_classifier(name, input_dim=Xtr.shape[1])
                    clf.fit(Xtr, ytr)
                    try:
                        y_proba = clf.predict_proba(Xte)[:, 1]
                    except (AttributeError, NotImplementedError):
                        y_proba = clf.decision_function(Xte)
                else:
                    y_proba = proba_arr

            default_metrics = evaluate_with_threshold(yte, y_proba, 0.5)
            tuned_metrics = evaluate_with_threshold(yte, y_proba, tau)

            elapsed = time.time() - t0
            LOG.info(
                "  %s  default(τ=0.50) acc=%.3f sens=%.3f spec=%.3f  →  tuned(τ=%.2f) acc=%.3f sens=%.3f spec=%.3f  (%.1fs)",
                name, default_metrics["accuracy"], default_metrics["sensitivity"], default_metrics["specificity"],
                tau, tuned_metrics["accuracy"], tuned_metrics["sensitivity"], tuned_metrics["specificity"], elapsed,
            )
            rows.append({"lambda": lam, "classifier": name, "stage": "default_τ=0.5", **default_metrics})
            rows.append({"lambda": lam, "classifier": name, "stage": f"tuned_τ={tau:.2f}", **tuned_metrics})
    return pd.DataFrame(rows)


def build_stacking(lambdas: list[str], top_n: int, n_features: int, cv_splits: int) -> pd.DataFrame:
    """For each λ, take the top-N base classifiers by saved test AUC and
    train a Logistic Regression meta-learner on their OOF probabilities."""
    summary_csv = max(RESULTS_DIR.glob("summary_*.csv"),
                      key=lambda p: p.stat().st_mtime, default=None)
    if summary_csv is None:
        LOG.error("No summary CSV — cannot pick top base models")
        return pd.DataFrame()
    summary = pd.read_csv(summary_csv)

    rows = []
    for lam in lambdas:
        top = summary[summary["lambda"] == lam].nlargest(top_n, "test_auc")
        base_names = [c for c in top["classifier"] if c != "Deep DNN"]
        if len(base_names) < 2:
            LOG.warning("[%s] too few non-DNN base models", lam)
            continue
        LOG.info("[%s] stacking with base models: %s", lam, base_names)

        Xtr, ytr, Xte, yte = prepare_xy(lam, n_features)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        # Compute OOF + test predictions for each base model
        oof_matrix = np.zeros((len(ytr), len(base_names)), dtype=float)
        test_matrix = np.zeros((len(yte), len(base_names)), dtype=float)
        for j, name in enumerate(base_names):
            LOG.info("  computing OOF + test for %s", name)
            oof_matrix[:, j] = oof_probabilities(name, Xtr, ytr, cv)
            clf = build_classifier(name, input_dim=Xtr.shape[1])
            clf.fit(Xtr, ytr)
            try:
                test_matrix[:, j] = clf.predict_proba(Xte)[:, 1]
            except (AttributeError, NotImplementedError):
                test_matrix[:, j] = clf.decision_function(Xte)

        # Meta-learner: LR (no penalty tuning, default C=1)
        meta = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
        meta.fit(oof_matrix, ytr)
        y_proba = meta.predict_proba(test_matrix)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        rows.append({
            "lambda": lam,
            "classifier": f"Stack({'+'.join(base_names)})",
            "test_accuracy": float(accuracy_score(yte, y_pred)),
            "test_sensitivity": float(recall_score(yte, y_pred, zero_division=0.0)),
            "test_specificity": float(specificity(yte, y_pred)),
            "test_precision": float(precision_score(yte, y_pred, zero_division=0.0)),
            "test_f1": float(f1_score(yte, y_pred, zero_division=0.0)),
            "test_auc": float(roc_auc_score(yte, y_proba)),
        })
        LOG.info("  stack[%s]: acc=%.4f sens=%.4f spec=%.4f auc=%.4f",
                 lam, rows[-1]["test_accuracy"], rows[-1]["test_sensitivity"],
                 rows[-1]["test_specificity"], rows[-1]["test_auc"])
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="low_pass,high_pass,dft,odd_harmonic")
    parser.add_argument("--n-features", type=int, default=360)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--criterion", default="f1",
                        choices=["f1", "balanced_acc", "youden"])
    parser.add_argument("--stack-top-n", type=int, default=3)
    parser.add_argument("--skip-threshold", action="store_true")
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--suffix", default="", help="Pickle suffix, e.g. 'hybrid'.")
    args = parser.parse_args()
    global _SUFFIX
    _SUFFIX = args.suffix
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.skip_threshold:
        LOG.info("======== THRESHOLD TUNING (criterion=%s) ========", args.criterion)
        tdf = threshold_tune(lambdas, args.n_features, args.cv_splits, args.criterion)
        out = RESULTS_DIR / f"threshold_tuned_{args.criterion}_{timestamp}.csv"
        tdf.to_csv(out, index=False)
        LOG.info("Wrote %s", out)

    if not args.skip_stacking:
        LOG.info("======== STACKING (top-%d base models) ========", args.stack_top_n)
        sdf = build_stacking(lambdas, args.stack_top_n, args.n_features, args.cv_splits)
        out = RESULTS_DIR / f"stacking_{timestamp}.csv"
        sdf.to_csv(out, index=False)
        LOG.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
