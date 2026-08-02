"""
Train + evaluate the 9 paper classifiers across all 4 MKT λ configurations.

Inputs (produced by `feature_extraction.py`):
    paper_pipeline/output/features/features_train_<lambda>.pkl
    paper_pipeline/output/features/features_test_<lambda>.pkl

Outputs:
    paper_pipeline/output/results/results_<lambda>_<timestamp>.csv
    paper_pipeline/output/results/summary_<timestamp>.csv   (all λ combined)
    paper_pipeline/output/manifests/train_eval_<timestamp>.json

The test set is touched exactly once per (λ, classifier) pair.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline import qa
from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier
from paper_pipeline.pipeline.mkt_lambda import all_lambda_names

LOG = logging.getLogger("train_eval")

BOOKKEEPING_COLS = {"source_id", "aug_tag", "label"}
DEFAULT_N_FEATURES = 360
DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, feature_cols


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_one_classifier(
    name: str,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    X_test_sel: np.ndarray,
    y_test: np.ndarray,
    cv_splits: int,
    random_state: int,
    predictions_dir: Path | None = None,
    lambda_name: str | None = None,
) -> dict[str, float | str]:
    LOG.info("  ---- %s ----", name)
    clf = build_classifier(name, input_dim=X_train_sel.shape[1])

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_acc_mean = cv_auc_mean = cv_acc_std = cv_auc_std = float("nan")
    try:
        cv_scores = cross_validate(
            clf, X_train_sel, y_train, cv=cv,
            scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
            n_jobs=1, return_estimator=False,
        )
        cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
        cv_auc_mean = float(np.mean(cv_scores["test_roc_auc"]))
        cv_acc_std = float(np.std(cv_scores["test_accuracy"]))
        cv_auc_std = float(np.std(cv_scores["test_roc_auc"]))
    except Exception as e:
        LOG.warning("CV failed for %s: %s", name, e)

    clf = build_classifier(name, input_dim=X_train_sel.shape[1])
    t0 = time.time()
    clf.fit(X_train_sel, y_train)
    fit_seconds = time.time() - t0

    # === Test set touched HERE — exactly once ===
    y_pred = clf.predict(X_test_sel)
    try:
        y_proba = clf.predict_proba(X_test_sel)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_proba = clf.decision_function(X_test_sel)
        except Exception:
            y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, zero_division=0.0)
    specificity = specificity_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0.0)
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    LOG.info(
        "    acc=%.4f  sens=%.4f  spec=%.4f  prec=%.4f  f1=%.4f  auc=%.4f  (fit=%.1fs)",
        accuracy, sensitivity, specificity, precision, f1, auc, fit_seconds,
    )

    # Save the raw test predictions so figures (ROC, PR, confusion) can be
    # regenerated later without retraining.
    if predictions_dir is not None and lambda_name is not None:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        out = predictions_dir / f"preds_{lambda_name}_{safe}.npz"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=np.array([]) if y_proba is None else y_proba,
            classifier=name,
            lambda_name=lambda_name,
        )

    return {
        "classifier": name,
        "test_accuracy": accuracy,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_precision": precision,
        "test_f1": f1,
        "test_auc": auc,
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std": cv_acc_std,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "fit_seconds": fit_seconds,
    }


def run_one_lambda(
    lambda_name: str,
    train_pickle: Path,
    test_pickle: Path,
    n_features: int,
    cv_splits: int,
    random_state: int,
    results_dir: Path,
    timestamp: str,
) -> tuple[Path, list[dict]]:
    LOG.info("================  λ = %s  ================", lambda_name)
    train_df = pd.read_pickle(train_pickle)
    test_df = pd.read_pickle(test_pickle)
    LOG.info("Loaded train (%d × %d) and test (%d × %d) from %s",
             len(train_df), train_df.shape[1], len(test_df), test_df.shape[1],
             train_pickle.parent.name)

    # No-leakage re-check
    train_ids = set(train_df["source_id"])
    test_ids = set(test_df["source_id"])
    if train_ids & test_ids:
        raise RuntimeError(
            f"LEAKAGE in λ={lambda_name}: {len(train_ids & test_ids)} overlapping source_ids"
        )

    X_train, y_train, feature_cols = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)
    LOG.info("Feature columns: %d   train shape: %s   test shape: %s",
             len(feature_cols), X_train.shape, X_test.shape)
    LOG.info("Class balance — train: %s,  test: %s",
             dict(Counter(y_train)), dict(Counter(y_test)))

    k = min(n_features, X_train.shape[1])
    LOG.info("Mutual-information selection: k=%d (fit on train only)", k)
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    LOG.info("After selection: train=%s, test=%s", X_train_sel.shape, X_test_sel.shape)

    scaler = RobustScaler()
    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)

    rows: list[dict] = []
    predictions_dir = results_dir.parent / "predictions"
    for name in ACTIVE_CLASSIFIERS:
        row = evaluate_one_classifier(
            name, X_train_sel, y_train, X_test_sel, y_test,
            cv_splits=cv_splits, random_state=random_state,
            predictions_dir=predictions_dir, lambda_name=lambda_name,
        )
        row["lambda"] = lambda_name
        rows.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"results_{lambda_name}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    LOG.info("Wrote %s", out)
    return out, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path,
        default=PROJECT_ROOT / "paper_pipeline" / "output" / "features",
    )
    parser.add_argument("--lambdas", type=str, default=",".join(all_lambda_names()))
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--results-dir", type=Path,
        default=PROJECT_ROOT / "paper_pipeline" / "output" / "results",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Filename suffix: features_<split>_<lambda>_<suffix>.pkl (default empty)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    chosen_lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows: list[dict] = []
    per_lambda_files: list[str] = []
    suffix_part = f"_{args.suffix}" if args.suffix else ""
    for name in chosen_lambdas:
        train_pickle = args.features_dir / f"features_train_{name}{suffix_part}.pkl"
        test_pickle = args.features_dir / f"features_test_{name}{suffix_part}.pkl"
        if not train_pickle.is_file() or not test_pickle.is_file():
            LOG.error("Missing pickles for λ=%s — expected %s and %s",
                      name, train_pickle.name, test_pickle.name)
            return 2
        out_csv, rows = run_one_lambda(
            name, train_pickle, test_pickle,
            n_features=args.n_features,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            results_dir=args.results_dir,
            timestamp=timestamp,
        )
        per_lambda_files.append(str(out_csv))
        all_rows.extend(rows)

    summary_path = args.results_dir / f"summary_{timestamp}.csv"
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(summary_path, index=False)
    LOG.info("Wrote combined summary: %s", summary_path)

    qa.write_manifest({
        "lambdas": chosen_lambdas,
        "n_features": args.n_features,
        "cv_splits": args.cv_splits,
        "random_state": args.random_state,
        "per_lambda_files": per_lambda_files,
        "summary_file": str(summary_path),
        "best_row_per_lambda": (
            summary_df.loc[summary_df.groupby("lambda")["test_accuracy"].idxmax()]
                      [["lambda", "classifier", "test_accuracy", "test_auc"]]
                      .to_dict(orient="records")
        ),
    }, "train_eval")

    # Pretty-print best classifier per lambda
    LOG.info("\n================  BEST CLASSIFIER PER λ  ================")
    for name in chosen_lambdas:
        sub = summary_df[summary_df["lambda"] == name]
        if sub.empty:
            continue
        best = sub.loc[sub["test_accuracy"].idxmax()]
        LOG.info("  λ=%-12s  %-20s  acc=%.4f  sens=%.4f  spec=%.4f  auc=%.4f",
                 name, best["classifier"], best["test_accuracy"],
                 best["test_sensitivity"], best["test_specificity"], best["test_auc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
