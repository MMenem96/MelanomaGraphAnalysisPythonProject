"""Deterministic k-sensitivity sweep for the headline cell (MLP, odd-harmonic).

Uses the same deterministic MI helper as phase_a_full_grid.py so the numbers
reproduce on re-run and can be cited cleanly in §4.4 of the paper.
"""
from __future__ import annotations
import logging, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR  = PROJECT_ROOT / "paper_pipeline" / "output" / "results"

LAMBDA       = "odd_harmonic"
RANDOM_STATE = 42
K_VALUES     = [100, 200, 360, 500, 800, 1200, 2000, 2509]   # 2509 = no selection
BOOKKEEPING  = {"source_id", "aug_tag", "label"}


def _mi_seeded(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_STATE,
                                n_neighbors=3, n_jobs=1)


def build_mlp():
    return MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001,
                         learning_rate="adaptive", learning_rate_init=0.001,
                         max_iter=500, early_stopping=True, validation_fraction=0.1,
                         random_state=RANDOM_STATE)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("k_sweep")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_pickle(FEATURES_DIR / f"features_train_{LAMBDA}_hybrid.pkl")
    test_df  = pd.read_pickle(FEATURES_DIR / f"features_test_{LAMBDA}_hybrid.pkl")
    cols = [c for c in train_df.columns if c not in BOOKKEEPING]
    Xtr = np.nan_to_num(train_df[cols].to_numpy(dtype=np.float64),
                        nan=0.0, posinf=1e10, neginf=-1e10)
    ytr = train_df["label"].to_numpy(dtype=np.int64)
    Xte = np.nan_to_num(test_df[cols].to_numpy(dtype=np.float64),
                        nan=0.0, posinf=1e10, neginf=-1e10)
    yte = test_df["label"].to_numpy(dtype=np.int64)
    log.info("Train shape: %s   Test shape: %s", Xtr.shape, Xte.shape)
    log.info("λ = %s, classifier = MLP", LAMBDA)

    rows = []
    for k in K_VALUES:
        t0 = time.time()
        if k >= Xtr.shape[1]:
            Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
            k_actual = Xtr.shape[1]
        else:
            sel = SelectKBest(_mi_seeded, k=k).fit(Xtr, ytr)
            Xtr_s = sel.transform(Xtr); Xte_s = sel.transform(Xte)
            k_actual = k
        sca = RobustScaler().fit(Xtr_s)
        Xtr_n = sca.transform(Xtr_s); Xte_n = sca.transform(Xte_s)

        clf = build_mlp().fit(Xtr_n, ytr)
        yp  = clf.predict(Xte_n)
        ypr = clf.predict_proba(Xte_n)[:, 1]
        tn, fp, fn, tp = confusion_matrix(yte, yp, labels=[0, 1]).ravel()
        acc  = accuracy_score(yte, yp)
        sens = recall_score(yte, yp, zero_division=0.0)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        auc  = roc_auc_score(yte, ypr)
        log.info("k=%-5d acc=%.4f sens=%.4f spec=%.4f auc=%.4f  (%.1fs)",
                 k_actual, acc, sens, spec, auc, time.time() - t0)
        rows.append({"k_requested": k, "k_actual": k_actual,
                      "accuracy": acc, "sensitivity": sens, "specificity": spec,
                      "auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    out = RESULTS_DIR / f"k_sweep_deterministic_{LAMBDA}_MLP_{ts}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("→ %s", out)


if __name__ == "__main__":
    main()
