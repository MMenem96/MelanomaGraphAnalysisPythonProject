"""
Save feature importances per (λ, classifier) for the tree-based and linear
models. Also reports the top-30 features by mutual information (the actual
selection scores used by SelectKBest in train_eval).

Outputs:
    paper_pipeline/output/figures/feat_imp_<lambda>_<classifier>.csv
    paper_pipeline/output/figures/mi_top30_<lambda>.csv
    paper_pipeline/output/figures/feat_imp_<lambda>_<classifier>.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
FIGURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "figures"

# Only these classifiers expose feature_importances_ or coef_.
IMPORTANCE_CAPABLE = ("CatBoost", "LightGBM", "Gradient Boosting", "Extra Trees", "Logistic Regression")

LOG = logging.getLogger("feature_importance")

BOOKKEEPING = {"source_id", "aug_tag", "label"}


def _get_importance(clf, feature_names: list[str]) -> np.ndarray | None:
    """Try the common sklearn-style importance accessors."""
    # CalibratedClassifierCV wraps an estimator
    if hasattr(clf, "calibrated_classifiers_"):
        try:
            base = clf.calibrated_classifiers_[0].estimator
            if hasattr(base, "coef_"):
                return np.abs(base.coef_).ravel()
        except Exception:
            return None
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_)).ravel()
    return None


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def run_for_lambda(lambda_name: str, n_features: int, top: int) -> None:
    train_pickle = FEATURES_DIR / f"features_train_{lambda_name}.pkl"
    if not train_pickle.is_file():
        LOG.warning("Missing %s — skip", train_pickle.name)
        return
    LOG.info("Loading %s", train_pickle.name)
    df = pd.read_pickle(train_pickle)
    feature_cols = [c for c in df.columns if c not in BOOKKEEPING]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    y = df["label"].to_numpy(dtype=np.int64)
    LOG.info("  X=%s y=%s", X.shape, y.shape)

    # Mutual-information scores (the same scores that drive SelectKBest)
    LOG.info("Computing mutual information…")
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    mi_df = pd.DataFrame({"feature": feature_cols, "mi": mi}).sort_values("mi", ascending=False)
    mi_top = mi_df.head(30).reset_index(drop=True)
    out_mi = FIGURES_DIR / f"mi_top30_{lambda_name}.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mi_top.to_csv(out_mi, index=False)
    LOG.info("  wrote %s", out_mi.name)

    # Apply selection + scaling exactly as train_eval does
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    X_sel = selector.fit_transform(X, y)
    selected_idx = selector.get_support(indices=True)
    selected_names = [feature_cols[i] for i in selected_idx]
    scaler = RobustScaler()
    X_sel = scaler.fit_transform(X_sel)
    LOG.info("  selected k=%d features; X_sel=%s", X_sel.shape[1], X_sel.shape)

    for clf_name in IMPORTANCE_CAPABLE:
        if clf_name not in ACTIVE_CLASSIFIERS:
            continue
        LOG.info("  fitting %s for importance extraction…", clf_name)
        clf = build_classifier(clf_name, input_dim=X_sel.shape[1])
        try:
            clf.fit(X_sel, y)
        except Exception as e:
            LOG.warning("    fit failed for %s: %s", clf_name, e)
            continue
        imp = _get_importance(clf, selected_names)
        if imp is None:
            LOG.warning("    no importance accessor on %s", clf_name)
            continue
        imp_df = pd.DataFrame({"feature": selected_names, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        out_csv = FIGURES_DIR / f"feat_imp_{lambda_name}_{_safe(clf_name)}.csv"
        imp_df.to_csv(out_csv, index=False)
        LOG.info("    wrote %s", out_csv.name)

        top_n = imp_df.head(top)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * len(top_n))))
        ax.barh(top_n["feature"][::-1], top_n["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{len(top_n)} features — {clf_name}  (λ = {lambda_name})")
        fig.tight_layout()
        out_png = FIGURES_DIR / f"feat_imp_{lambda_name}_{_safe(clf_name)}.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        LOG.info("    wrote %s", out_png.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", default="low_pass,high_pass,dft,odd_harmonic")
    parser.add_argument("--n-features", type=int, default=360)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    for lam in [s.strip() for s in args.lambdas.split(",") if s.strip()]:
        run_for_lambda(lam, args.n_features, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
