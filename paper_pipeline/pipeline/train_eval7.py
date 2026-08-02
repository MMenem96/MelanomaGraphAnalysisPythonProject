"""
7-class (Track B) training + evaluation with imbalance-aware metrics.

Sibling of `train_eval.py`; the binary module is untouched.

Three things differ from the binary protocol, all forced by the 7-class setting:

  1. **Grouped CV.** Augmentation puts up to 21 rows of the same source image in
     the training frame, and HAM10000 puts several images of the same lesion in
     the dataset. Plain `StratifiedKFold` would place a rotated copy of a
     training image in the validation fold and inflate CV scores, so folds are
     built with `StratifiedGroupKFold` grouped on `lesion_id`.
  2. **Metrics.** Overall accuracy is uninterpretable against a 66.9% nevus
     majority, so the headline numbers are balanced accuracy (the ISIC 2018
     Task 3 official metric), macro-F1 and macro one-vs-rest AUC, always
     reported with per-class recall and the 7x7 confusion matrix.
  3. **Baseline.** The majority-class rate is computed and reported next to
     accuracy so the numbers can be read honestly.

The test set is touched exactly once per (lambda, classifier) pair. Model
selection must use the CV columns only.

Outputs:
    paper_pipeline/output/results7/results7_<lambda>_<suffix>_<timestamp>.csv
    paper_pipeline/output/results7/confusion_<lambda>_<clf>_<suffix>.csv
    paper_pipeline/output/results7/preds7_<lambda>_<clf>_<suffix>.npz
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
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline import qa
from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier
from paper_pipeline.pipeline.dataset import CLASSES7

LOG = logging.getLogger("train_eval7")

BOOKKEEPING_COLS = {"source_id", "aug_tag", "label", "lesion_id", "dx"}
DEFAULT_N_FEATURES = 360
DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_LAMBDA = "odd_harmonic"


def split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["lesion_id"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, groups, feature_cols


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray | None) -> float:
    """Macro one-vs-rest AUC, tolerant of classes missing from the test split."""
    if proba is None:
        return float("nan")
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        if len(present) == proba.shape[1]:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        # Restrict to classes actually present, renormalising their columns.
        sub = proba[:, present]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        remap = {c: i for i, c in enumerate(present)}
        y_remap = np.array([remap[v] for v in y_true])
        return float(roc_auc_score(y_remap, sub, multi_class="ovr", average="macro"))
    except Exception as e:  # pragma: no cover - defensive
        LOG.warning("macro OvR AUC failed: %s", e)
        return float("nan")


def cross_validate_grouped(
    name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    cv_splits: int, random_state: int,
) -> dict[str, float]:
    """Grouped, stratified CV on the TRAINING data only.

    Grouping on lesion_id keeps augmented copies of an image — and repeat
    photographs of a lesion — out of the validation fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    bal_accs, macro_f1s = [], []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        try:
            clf = build_classifier(name, input_dim=X.shape[1])
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            bal_accs.append(balanced_accuracy_score(y[va], pred))
            macro_f1s.append(f1_score(y[va], pred, average="macro", zero_division=0.0))
        except Exception as e:
            LOG.warning("  CV fold %d failed for %s: %s", fold, name, e)
    if not bal_accs:
        return {"cv_balanced_accuracy_mean": float("nan"), "cv_balanced_accuracy_std": float("nan"),
                "cv_macro_f1_mean": float("nan"), "cv_macro_f1_std": float("nan")}
    return {
        "cv_balanced_accuracy_mean": float(np.mean(bal_accs)),
        "cv_balanced_accuracy_std": float(np.std(bal_accs)),
        "cv_macro_f1_mean": float(np.mean(macro_f1s)),
        "cv_macro_f1_std": float(np.std(macro_f1s)),
    }


def evaluate_one_classifier(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cv_splits: int, random_state: int,
    results_dir: Path, lambda_name: str, suffix: str,
    skip_cv: bool,
) -> dict:
    LOG.info("  ---- %s ----", name)
    row: dict = {"classifier": name}

    if skip_cv:
        row.update({"cv_balanced_accuracy_mean": float("nan"), "cv_balanced_accuracy_std": float("nan"),
                    "cv_macro_f1_mean": float("nan"), "cv_macro_f1_std": float("nan")})
    else:
        t0 = time.time()
        row.update(cross_validate_grouped(
            name, X_train, y_train, groups_train, cv_splits, random_state))
        LOG.info("    CV(grouped): bal_acc=%.4f±%.4f  macro_f1=%.4f  (%.1fs)",
                 row["cv_balanced_accuracy_mean"], row["cv_balanced_accuracy_std"],
                 row["cv_macro_f1_mean"], time.time() - t0)

    clf = build_classifier(name, input_dim=X_train.shape[1])
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    # === Test set touched HERE — exactly once ===
    y_pred = clf.predict(X_test)
    try:
        proba = clf.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        proba = None

    labels = list(range(len(CLASSES7)))
    row.update({
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "test_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0.0),
        "test_weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0.0),
        "test_macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0.0),
        "test_macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0.0),
        "test_macro_ovr_auc": macro_ovr_auc(y_test, proba),
        "test_cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "fit_seconds": fit_seconds,
    })

    per_class_recall = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0.0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0.0)
    support = Counter(y_test.tolist())
    for i, dx in enumerate(CLASSES7):
        row[f"recall_{dx}"] = float(per_class_recall[i])
        row[f"f1_{dx}"] = float(per_class_f1[i])
        row[f"support_{dx}"] = int(support.get(i, 0))

    LOG.info("    TEST bal_acc=%.4f  macro_f1=%.4f  acc=%.4f  macro_auc=%.4f  kappa=%.4f (fit=%.1fs)",
             row["test_balanced_accuracy"], row["test_macro_f1"], row["test_accuracy"],
             row["test_macro_ovr_auc"], row["test_cohen_kappa"], fit_seconds)
    LOG.info("    per-class recall: %s",
             "  ".join(f"{dx}={row[f'recall_{dx}']:.3f}(n={row[f'support_{dx}']})"
                       for dx in CLASSES7))

    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    results_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES7],
                 columns=[f"pred_{c}" for c in CLASSES7]).to_csv(
        results_dir / f"confusion_{lambda_name}_{safe}_{suffix}.csv")
    np.savez(results_dir / f"preds7_{lambda_name}_{safe}_{suffix}.npz",
             y_test=y_test, y_pred=y_pred,
             y_proba=np.array([]) if proba is None else proba,
             classes=np.array(CLASSES7), classifier=name, lambda_name=lambda_name)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "results7")
    ap.add_argument("--lambdas", type=str, default=DEFAULT_LAMBDA)
    ap.add_argument("--suffix", default="lesion_equalize")
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    ap.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    ap.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    ap.add_argument("--classifiers", type=str, default=",".join(ACTIVE_CLASSIFIERS))
    ap.add_argument("--skip-cv", action="store_true",
                    help="Test-only pass (fast). Model selection still requires CV.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    chosen_lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    chosen_clfs = [s.strip() for s in args.classifiers.split(",") if s.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []

    for lam in chosen_lambdas:
        tr_path = args.features_dir / f"features_train_{lam}_{args.suffix}.pkl"
        te_path = args.features_dir / f"features_test_{lam}_{args.suffix}.pkl"
        if not tr_path.is_file() or not te_path.is_file():
            LOG.error("Missing pickles: %s / %s", tr_path.name, te_path.name)
            return 2

        LOG.info("================  λ = %s  (%s)  ================", lam, args.suffix)
        train_df = pd.read_pickle(tr_path)
        test_df = pd.read_pickle(te_path)

        # Leakage re-check before anything is fitted.
        img_overlap = set(train_df["source_id"]) & set(test_df["source_id"])
        les_overlap = set(train_df["lesion_id"]) & set(test_df["lesion_id"])
        if img_overlap:
            raise RuntimeError(f"IMAGE LEAKAGE: {len(img_overlap)} shared source_ids")
        if les_overlap:
            LOG.warning("LESION LEAKAGE: %d lesions on both sides "
                        "(expected only for the --split-unit image arm)", len(les_overlap))

        X_train, y_train, groups_train, feature_cols = split_xy(train_df)
        X_test, y_test, _, _ = split_xy(test_df)
        LOG.info("train %s   test %s   features=%d", X_train.shape, X_test.shape, len(feature_cols))
        LOG.info("train rows/class: %s", {CLASSES7[k]: v for k, v in sorted(Counter(y_train).items())})
        LOG.info("test  rows/class: %s", {CLASSES7[k]: v for k, v in sorted(Counter(y_test).items())})

        majority = Counter(y_test).most_common(1)[0]
        baseline = majority[1] / len(y_test)
        LOG.info("MAJORITY-CLASS BASELINE on test: %.4f (always predict %s). "
                 "Balanced-accuracy baseline: %.4f",
                 baseline, CLASSES7[majority[0]], 1.0 / len(CLASSES7))

        k = min(args.n_features, X_train.shape[1])
        LOG.info("Mutual-information selection: k=%d (fit on train only)", k)
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        scaler = RobustScaler()
        X_train_sel = scaler.fit_transform(X_train_sel)
        X_test_sel = scaler.transform(X_test_sel)

        for name in chosen_clfs:
            try:
                row = evaluate_one_classifier(
                    name, X_train_sel, y_train, groups_train, X_test_sel, y_test,
                    cv_splits=args.cv_splits, random_state=args.random_state,
                    results_dir=args.results_dir, lambda_name=lam, suffix=args.suffix,
                    skip_cv=args.skip_cv,
                )
            except Exception as e:
                LOG.error("  %s FAILED: %s", name, e)
                continue
            row["lambda"] = lam
            row["suffix"] = args.suffix
            row["majority_baseline"] = baseline
            row["n_features"] = k
            all_rows.append(row)

    if not all_rows:
        LOG.error("No classifier produced a result.")
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"results7_{args.suffix}_{timestamp}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    LOG.info("Wrote %s", out)

    qa.write_manifest({
        "suffix": args.suffix, "lambdas": chosen_lambdas, "classifiers": chosen_clfs,
        "n_features": args.n_features, "cv_splits": args.cv_splits,
        "random_state": args.random_state, "results_file": str(out),
    }, "train_eval7")

    LOG.info("\n===============  RANKED BY BALANCED ACCURACY  ===============")
    LOG.info("%-20s %10s %10s %10s %10s", "classifier", "bal_acc", "macro_f1", "acc", "macro_auc")
    for _, r in df.sort_values("test_balanced_accuracy", ascending=False).iterrows():
        LOG.info("%-20s %10.4f %10.4f %10.4f %10.4f",
                 r["classifier"], r["test_balanced_accuracy"], r["test_macro_f1"],
                 r["test_accuracy"], r["test_macro_ovr_auc"])
    LOG.info("(majority-class baseline: %.4f — accuracy below this is worse than "
             "always predicting nv)", all_rows[0]["majority_baseline"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
