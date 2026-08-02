"""
Cross-feature stacking: combine top base models from BOTH the frozen-CNN
hybrid features AND the fine-tuned-CNN hybrid features. A logistic-regression
meta-learner is fit on the union of out-of-fold predictions, then evaluated
once on the held-out test set.

Why this could help: frozen-CNN features give the highest AUC; fine-tuned-CNN
features give the highest accuracy. The two error modes are different, so the
ensemble has more chance of being right.

Outputs:
    paper_pipeline/output/results/cross_stack_<timestamp>.csv
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

from paper_pipeline.pipeline.classifiers import build_classifier

LOG = logging.getLogger("cross_stack")
BOOKKEEPING = {"source_id", "aug_tag", "label"}
N_FEATURES = 360
CV_SPLITS = 5
RANDOM_STATE = 42


def specificity(y_true, y_pred) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def prepare(lambda_name: str, suffix: str):
    train = pd.read_pickle(PROJECT_ROOT / "paper_pipeline" / "output" / "features"
                           / f"features_train_{lambda_name}_{suffix}.pkl")
    test = pd.read_pickle(PROJECT_ROOT / "paper_pipeline" / "output" / "features"
                          / f"features_test_{lambda_name}_{suffix}.pkl")
    feat_cols = [c for c in train.columns if c not in BOOKKEEPING]
    Xtr = np.nan_to_num(train[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    ytr = train["label"].to_numpy(np.int64)
    Xte = np.nan_to_num(test[feat_cols].to_numpy(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
    yte = test["label"].to_numpy(np.int64)
    sel = SelectKBest(mutual_info_classif, k=min(N_FEATURES, Xtr.shape[1]))
    Xtr_s = sel.fit_transform(Xtr, ytr)
    Xte_s = sel.transform(Xte)
    sc = RobustScaler()
    Xtr_s = sc.fit_transform(Xtr_s)
    Xte_s = sc.transform(Xte_s)
    return Xtr_s, ytr, Xte_s, yte


def oof_and_test_probs(clf_name: str, Xtr, ytr, Xte, cv: StratifiedKFold):
    oof = np.zeros(len(ytr))
    for tr_i, va_i in cv.split(Xtr, ytr):
        c = build_classifier(clf_name, input_dim=Xtr.shape[1])
        c.fit(Xtr[tr_i], ytr[tr_i])
        try:
            oof[va_i] = c.predict_proba(Xtr[va_i])[:, 1]
        except (AttributeError, NotImplementedError):
            oof[va_i] = c.predict(Xtr[va_i]).astype(float)
    c = build_classifier(clf_name, input_dim=Xtr.shape[1])
    c.fit(Xtr, ytr)
    try:
        test_p = c.predict_proba(Xte)[:, 1]
    except (AttributeError, NotImplementedError):
        test_p = c.predict(Xte).astype(float)
    return oof, test_p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="dft,odd_harmonic,low_pass")
    parser.add_argument(
        "--combos", default=
        "hybrid:MLP,hybrid:CatBoost,hybrid:LightGBM,hybrid:GradientBoosting,fthybrid:MLP,fthybrid:ExtraTrees",
        help="Comma-separated combos like 'suffix:classifier'",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    combos = []
    for c in args.combos.split(","):
        suf, clf = c.split(":")
        clf = clf.replace("GradientBoosting", "Gradient Boosting").replace("SVMRBF", "SVM (RBF)").replace("ExtraTrees", "Extra Trees").replace("LogisticRegression", "Logistic Regression").replace("DeepDNN", "Deep DNN")
        combos.append((suf, clf))
    LOG.info("Base combos: %s", combos)

    results = []
    for lam in [s.strip() for s in args.lambdas.split(",")]:
        LOG.info("================  λ = %s  ================", lam)
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        # For each suffix, prepare once
        prep_cache = {}
        for suf, _clf in combos:
            if suf not in prep_cache:
                LOG.info("  preparing features (%s) for %s …", suf, lam)
                try:
                    prep_cache[suf] = prepare(lam, suf)
                except FileNotFoundError as e:
                    LOG.warning("  missing pickle: %s", e)
                    prep_cache[suf] = None
        # Build matrices
        oof_cols = []
        test_cols = []
        names = []
        yte_ref = None
        ytr_ref = None
        for suf, clf in combos:
            if prep_cache.get(suf) is None:
                continue
            Xtr, ytr, Xte, yte = prep_cache[suf]
            if ytr_ref is None: ytr_ref = ytr
            if yte_ref is None: yte_ref = yte
            t0 = time.time()
            oof, test_p = oof_and_test_probs(clf, Xtr, ytr, Xte, cv)
            LOG.info("    %s/%s  OOF+test in %.1fs", suf, clf, time.time() - t0)
            oof_cols.append(oof)
            test_cols.append(test_p)
            names.append(f"{suf}:{clf}")
        if not oof_cols:
            continue
        OOF = np.column_stack(oof_cols)
        TST = np.column_stack(test_cols)
        meta = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)
        meta.fit(OOF, ytr_ref)
        y_proba = meta.predict_proba(TST)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        row = {
            "lambda": lam,
            "stack_components": " + ".join(names),
            "test_accuracy": accuracy_score(yte_ref, y_pred),
            "test_sensitivity": recall_score(yte_ref, y_pred, zero_division=0.0),
            "test_specificity": specificity(yte_ref, y_pred),
            "test_precision": precision_score(yte_ref, y_pred, zero_division=0.0),
            "test_f1": f1_score(yte_ref, y_pred, zero_division=0.0),
            "test_auc": roc_auc_score(yte_ref, y_proba),
        }
        LOG.info("  CROSS-STACK[%s]  acc=%.4f sens=%.4f spec=%.4f auc=%.4f",
                 lam, row["test_accuracy"], row["test_sensitivity"],
                 row["test_specificity"], row["test_auc"])
        results.append(row)

    df = pd.DataFrame(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "paper_pipeline" / "output" / "results" / f"cross_stack_{stamp}.csv"
    df.to_csv(out, index=False)
    LOG.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
