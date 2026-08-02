"""
Hyperparameter tuning on TRAIN-ONLY data, with GroupKFold by source_id so
the augmented copies stay in the same fold (closes the optimistic-CV gap).

This is the only legitimate way to "raise the numbers" — by finding better
hyperparameters via cross-validation on the training set, then evaluating
the chosen model ONCE on the held-out test set.

Configurable models to tune (each with a small/medium grid):
    SVM (RBF)      — C in {1, 10, 100, 1000}, gamma in {'scale', 'auto', 0.01, 0.001}
    LightGBM       — n_estimators in {500, 1000}, num_leaves in {31, 63, 127}
    CatBoost       — iterations in {500, 1000}, depth in {6, 8, 10}, lr in {0.03, 0.05, 0.1}
    MLP            — hidden in {(100,50), (200,100,50), (256,128)}, alpha in {1e-4, 1e-3, 1e-2}

Outputs:
    paper_pipeline/output/results/hp_tune_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.classifiers import build_classifier

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
LOG = logging.getLogger("hp_tune")
BOOKKEEPING = {"source_id", "aug_tag", "label"}


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def prepare(lambda_name: str, n_features: int, suffix: str = ""):
    suf = f"_{suffix}" if suffix else ""
    train = pd.read_pickle(FEATURES_DIR / f"features_train_{lambda_name}{suf}.pkl")
    test = pd.read_pickle(FEATURES_DIR / f"features_test_{lambda_name}{suf}.pkl")
    feat_cols = [c for c in train.columns if c not in BOOKKEEPING]
    Xtr = np.nan_to_num(train[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    ytr = train["label"].to_numpy(np.int64)
    groups = train["source_id"].to_numpy()
    Xte = np.nan_to_num(test[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    yte = test["label"].to_numpy(np.int64)
    sel = SelectKBest(mutual_info_classif, k=min(n_features, Xtr.shape[1]))
    Xtr_s = sel.fit_transform(Xtr, ytr)
    Xte_s = sel.transform(Xte)
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_s)
    Xte_s = scaler.transform(Xte_s)
    return Xtr_s, ytr, groups, Xte_s, yte


def evaluate_on_test(name: str, params: dict, Xtr, ytr, Xte, yte) -> dict:
    """Fit one config on full train, evaluate ONCE on test."""
    clf = build_classifier(name, input_dim=Xtr.shape[1])
    # Override hyperparameters
    for k, v in params.items():
        if hasattr(clf, "set_params"):
            try:
                clf.set_params(**{k: v})
                continue
            except Exception:
                pass
        setattr(clf, k, v)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    fit_t = time.time() - t0
    yp = clf.predict(Xte)
    try:
        yproba = clf.predict_proba(Xte)[:, 1]
    except Exception:
        yproba = None
    return {
        "params": params,
        "test_accuracy": accuracy_score(yte, yp),
        "test_sensitivity": recall_score(yte, yp, zero_division=0.0),
        "test_specificity": specificity(yte, yp),
        "test_f1": f1_score(yte, yp, zero_division=0.0),
        "test_auc": roc_auc_score(yte, yproba) if yproba is not None else float("nan"),
        "fit_seconds": fit_t,
    }


def build_grid(name: str) -> list[dict]:
    if name == "SVM (RBF)":
        return [{"C": c, "gamma": g}
                for c, g in product([1, 10, 100, 1000], ["scale", "auto", 0.01, 0.001])]
    if name == "LightGBM":
        return [{"n_estimators": n, "num_leaves": l, "learning_rate": lr}
                for n, l, lr in product([500, 1000], [31, 63, 127], [0.03, 0.05])]
    if name == "CatBoost":
        return [{"iterations": it, "depth": d, "learning_rate": lr}
                for it, d, lr in product([500, 1000], [6, 8, 10], [0.03, 0.05, 0.1])]
    if name == "MLP":
        return [{"hidden_layer_sizes": h, "alpha": a, "learning_rate_init": lri}
                for h, a, lri in product([(100, 50), (200, 100, 50), (256, 128)],
                                          [1e-4, 1e-3, 1e-2], [1e-3])]
    raise ValueError(f"No grid for {name}")


def tune_one(name: str, lambda_name: str, n_features: int, cv_splits: int, suffix: str = "") -> pd.DataFrame:
    Xtr, ytr, groups, Xte, yte = prepare(lambda_name, n_features, suffix)
    LOG.info("[%s][%s] grid size = %d", lambda_name, name, len(build_grid(name)))
    cv = GroupKFold(n_splits=cv_splits)
    rows = []
    grid = build_grid(name)
    for i, params in enumerate(grid):
        clf = build_classifier(name, input_dim=Xtr.shape[1])
        for k, v in params.items():
            try:
                clf.set_params(**{k: v})
            except Exception:
                setattr(clf, k, v)
        try:
            cv_scores = cross_val_score(clf, Xtr, ytr, groups=groups,
                                        cv=cv.split(Xtr, ytr, groups),
                                        scoring="f1", n_jobs=1)
            cv_f1 = float(np.mean(cv_scores))
        except Exception as e:
            LOG.warning("    [%d/%d] CV failed for %s %s: %s", i+1, len(grid), name, params, e)
            cv_f1 = float("nan")
        rows.append({
            "lambda": lambda_name, "classifier": name,
            **{f"param_{k}": v for k, v in params.items()},
            "cv_f1_mean": cv_f1,
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(grid):
            LOG.info("  [%d/%d] %s  best so far cv_f1=%.4f",
                     i+1, len(grid), name,
                     max(r["cv_f1_mean"] for r in rows if not np.isnan(r["cv_f1_mean"])))
    df = pd.DataFrame(rows)
    # Pick the winner and evaluate on test once.
    winner = df.loc[df["cv_f1_mean"].idxmax()]
    win_params = {k.replace("param_", ""): v for k, v in winner.items()
                  if k.startswith("param_") and not (isinstance(v, float) and np.isnan(v))}
    LOG.info("[%s][%s] WINNER params=%s cv_f1=%.4f", lambda_name, name, win_params, winner["cv_f1_mean"])
    test_metrics = evaluate_on_test(name, win_params, Xtr, ytr, Xte, yte)
    LOG.info("[%s][%s] TEST acc=%.4f sens=%.4f spec=%.4f auc=%.4f",
             lambda_name, name, test_metrics["test_accuracy"], test_metrics["test_sensitivity"],
             test_metrics["test_specificity"], test_metrics["test_auc"])
    summary_row = {
        "lambda": lambda_name, "classifier": name,
        "best_params": str(win_params), "cv_f1_mean": winner["cv_f1_mean"],
        **{k: v for k, v in test_metrics.items() if k != "params"},
    }
    return df, summary_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="low_pass,odd_harmonic",
                        help="Default: best 2 λ.")
    parser.add_argument("--classifiers", default="SVM (RBF),LightGBM,CatBoost,MLP")
    parser.add_argument("--n-features", type=int, default=360)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--suffix", default="", help="Pickle suffix, e.g. 'hybrid'.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    classifiers = [s.strip() for s in args.classifiers.split(",") if s.strip()]
    LOG.info("HP tuning: lambdas=%s, classifiers=%s", lambdas, classifiers)

    all_summary = []
    full_grid_rows = []
    for lam in lambdas:
        for clf in classifiers:
            df, summary = tune_one(clf, lam, args.n_features, args.cv_splits, args.suffix)
            full_grid_rows.append(df)
            all_summary.append(summary)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_csv = RESULTS_DIR / f"hp_tune_full_grid_{stamp}.csv"
    summary_csv = RESULTS_DIR / f"hp_tune_summary_{stamp}.csv"
    pd.concat(full_grid_rows, ignore_index=True).to_csv(grid_csv, index=False)
    pd.DataFrame(all_summary).to_csv(summary_csv, index=False)
    LOG.info("Wrote %s and %s", grid_csv, summary_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
