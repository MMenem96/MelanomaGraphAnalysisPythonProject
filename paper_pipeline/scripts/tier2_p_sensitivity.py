"""Tier-2 driver: Krawtchouk p-sensitivity sweep at the headline λ (DFT).

Resolves reviewer concern P1-4: "Krawtchouk parameter p=0.5 not justified."

Re-extracts MKT features at p ∈ {0.3, 0.5, 0.7} for the headline λ (DFT),
merges with the existing frozen ResNet-50 CNN features (CNN features do
NOT depend on p, so they are reused), trains the headline classifier
(MLP), and reports test-set metrics for each p.

Outputs (`paper_pipeline/output/results/`):
    tier2_p_sensitivity_<timestamp>.csv      one row per p
    tier2_p_features/                        new feature pickles per p
        features_train_dft_p030_hybrid.pkl
        features_test_dft_p030_hybrid.pkl
        ... (p=0.5 reuses existing pickle if present)
        features_train_dft_p070_hybrid.pkl
        features_test_dft_p070_hybrid.pkl

Run from project root:
    cd "/Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3"
    python paper_pipeline/scripts/tier2_p_sensitivity.py
    # Or restrict the sweep:
    python paper_pipeline/scripts/tier2_p_sensitivity.py --p-values 0.3,0.7

Compute cost: feature re-extraction at each new p value runs the full MKT
extractor across all 1,613 source images + 411 BCC h-flips + 411 BCC v-flips
(≈ 2,400 images total). Each p re-extraction takes roughly the same time as
the original full feature_extraction.py run.
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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after sys.path setup
from paper_pipeline.pipeline import feature_extraction as fx
from paper_pipeline.pipeline.feature_extraction import (
    _LambdaScopedExtractor, _process_sample_for_all_lambdas,
)
from paper_pipeline.pipeline.dataset import load_canonical_samples
from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn
from src.conventional_features import ConventionalFeatureExtractor


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
P_FEATURES_DIR = FEATURES_DIR / "tier2_p_features"

HEADLINE_LAMBDA = "odd_harmonic"
N_FEATURES = 360
RANDOM_STATE = 42
BOOKKEEPING_COLS = {"source_id", "aug_tag", "label"}


def _build_mlp() -> MLPClassifier:
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


def _split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in BOOKKEEPING_COLS]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X, y, cols


def _evaluate(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_features: int = N_FEATURES,
) -> dict:
    X_train, y_train, train_cols = _split_xy(train_df)
    X_test, y_test, _ = _split_xy(test_df)

    k = min(n_features, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k).fit(X_train, y_train)
    X_train_s = selector.transform(X_train)
    X_test_s = selector.transform(X_test)

    scaler = RobustScaler().fit(X_train_s)
    X_train_s = scaler.transform(X_train_s)
    X_test_s = scaler.transform(X_test_s)

    clf = _build_mlp()
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features_in": len(train_cols),
        "n_features_selected": k,
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": recall_score(y_test, y_pred, zero_division=0.0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_test, y_pred, zero_division=0.0),
        "f1": f1_score(y_test, y_pred, zero_division=0.0),
        "auc": roc_auc_score(y_test, y_proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _re_extract_mkt_at_p(p_value: float, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run MKT feature extraction at the given p, only for λ=DFT.

    Reuses everything else (preprocessing, augmentation order, dataset split).
    Returns (train_mkt_df, test_mkt_df) with columns: source_id, aug_tag, label, <MKT features...>.
    """
    log.info("  Monkey-patching feature_extraction.P_MKT = %s", p_value)
    original_p = fx.P_MKT
    fx.P_MKT = p_value

    try:
        # Build the λ-DFT extractor
        lam_fn = get_lambda_fn(HEADLINE_LAMBDA)
        extractor = _LambdaScopedExtractor(lambda_fn=lam_fn)
        base_extractor = ConventionalFeatureExtractor()
        lambda_extractors = {HEADLINE_LAMBDA: extractor}

        # Load the same canonical samples + augmentation policy used by the paper
        samples = load_canonical_samples()
        log.info("  Loaded %d canonical samples", len(samples))

        # Stratified split (same seed as the paper: random_state=42, test_size=0.2)
        # We need to mirror feature_extraction.py's split logic exactly.
        from sklearn.model_selection import train_test_split
        ids = np.array([s.image_id for s in samples])
        labels = np.array([s.label for s in samples])
        train_ids, test_ids = train_test_split(
            ids, test_size=0.2, stratify=labels, random_state=RANDOM_STATE,
        )
        train_set = set(train_ids); test_set = set(test_ids)
        train_samples = [s for s in samples if s.image_id in train_set]
        test_samples = [s for s in samples if s.image_id in test_set]
        log.info("  Split: train=%d test=%d", len(train_samples), len(test_samples))

        # Determine augmentation tags. For train: orig + h-flip + v-flip for BCC; orig only for BKL.
        # For test: orig only.
        def aug_tags_for(sample, is_train):
            if not is_train:
                return ("orig",)
            return ("orig", "h_flip", "v_flip") if sample.label == 1 else ("orig",)

        rows_train, rows_test = [], []
        t_start = time.time()
        log.info("  Extracting MKT features at p=%s for λ=%s (train+test)...",
                 p_value, HEADLINE_LAMBDA)
        for i, sample in enumerate(train_samples + test_samples):
            is_train = sample.image_id in train_set
            tags = aug_tags_for(sample, is_train)
            result = _process_sample_for_all_lambdas(
                sample, tags, base_extractor, lambda_extractors, save_qa=False,
            )
            lam_rows = result.get(HEADLINE_LAMBDA, [])
            for row in lam_rows:
                row_with_meta = dict(row)
                row_with_meta["source_id"] = sample.image_id
                row_with_meta["label"] = int(sample.label)
                if is_train:
                    rows_train.append(row_with_meta)
                else:
                    rows_test.append(row_with_meta)
            if (i + 1) % 100 == 0:
                log.info("    %d / %d images processed (%.1fs elapsed)",
                         i + 1, len(samples), time.time() - t_start)

        log.info("  Extraction done in %.1fs", time.time() - t_start)
        train_df = pd.DataFrame(rows_train)
        test_df  = pd.DataFrame(rows_test)
        return train_df, test_df

    finally:
        # Restore the original constant even if extraction failed
        fx.P_MKT = original_p
        log.info("  Restored feature_extraction.P_MKT = %s", fx.P_MKT)


def _merge_with_cnn(mkt_df: pd.DataFrame, cnn_pickle: Path,
                    log: logging.Logger) -> pd.DataFrame:
    """Inner-join the per-(source_id, aug_tag) MKT features with the existing
    frozen-CNN features.

    The CNN feature pickle is the same one used by the headline hybrid run
    — it does NOT depend on p (frozen ResNet-50 on the preprocessed image).
    """
    log.info("  Merging with CNN features: %s", cnn_pickle.name)
    cnn_df = pd.read_pickle(cnn_pickle)
    join_keys = ["source_id"]
    if "aug_tag" in cnn_df.columns and "aug_tag" in mkt_df.columns:
        join_keys.append("aug_tag")
    merged = pd.merge(mkt_df, cnn_df, on=join_keys, how="inner",
                      suffixes=("", "_cnn"))
    # Reconcile duplicate label column if present
    if "label_cnn" in merged.columns:
        if (merged["label"] == merged["label_cnn"]).all():
            merged = merged.drop(columns=["label_cnn"])
        else:
            log.warning("Label mismatch between MKT and CNN feature pickles!")
    log.info("  Merged shape: %s", merged.shape)
    return merged


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-values", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated p values to sweep (default: 0.3,0.5,0.7)")
    parser.add_argument("--reuse-p05", action="store_true",
                        help="If set, skip re-extraction at p=0.5 and use the "
                             "existing features_train_dft_hybrid.pkl. Default off "
                             "for stable reproducibility.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("tier2")

    p_values = [float(p) for p in args.p_values.split(",")]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    P_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Tier-2 p-sensitivity sweep at λ=%s, classifier=MLP", HEADLINE_LAMBDA)
    log.info("p values: %s", p_values)
    log.info("=" * 70)

    # Identify the CNN feature pickle. Convention from the practical project:
    # the hybrid pickle = MKT+handcrafted + frozen-CNN features, joined by
    # source_id (and aug_tag for the train side). We can extract the CNN-only
    # columns from the existing hybrid pickle.
    existing_hybrid_train = FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_hybrid.pkl"
    existing_hybrid_test  = FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_hybrid.pkl"
    if not existing_hybrid_train.exists() or not existing_hybrid_test.exists():
        log.error("Hybrid feature pickles not found. Cannot proceed.")
        sys.exit(1)

    log.info("Splitting CNN columns out of existing hybrid pickle...")
    hybrid_train = pd.read_pickle(existing_hybrid_train)
    hybrid_test = pd.read_pickle(existing_hybrid_test)
    cnn_cols = [c for c in hybrid_train.columns
                if "resnet" in c.lower() or "_cnn" in c.lower()
                or "deep_feat" in c.lower() or c.startswith("cnn_")
                or c.startswith("emb_")]
    if not cnn_cols:
        log.error("Could not identify CNN columns in the hybrid pickle. "
                  "Inspect column names and update the heuristic in this script.")
        sys.exit(1)
    log.info("  CNN columns identified: %d (sample: %s)", len(cnn_cols), cnn_cols[:3])
    cnn_keep = ["source_id", "aug_tag", "label"] + cnn_cols
    cnn_train_only = hybrid_train[[c for c in cnn_keep if c in hybrid_train.columns]]
    cnn_test_only  = hybrid_test[[c for c in cnn_keep if c in hybrid_test.columns]]
    cnn_train_path = P_FEATURES_DIR / "cnn_features_train.pkl"
    cnn_test_path  = P_FEATURES_DIR / "cnn_features_test.pkl"
    cnn_train_only.to_pickle(cnn_train_path)
    cnn_test_only.to_pickle(cnn_test_path)

    summary_rows = []
    for p in p_values:
        log.info("")
        log.info("-" * 70)
        log.info("  p = %s", p)
        log.info("-" * 70)

        p_tag = f"p{int(p*100):03d}"   # e.g. p030, p050, p070
        train_pickle = P_FEATURES_DIR / f"features_train_{HEADLINE_LAMBDA}_{p_tag}_hybrid.pkl"
        test_pickle  = P_FEATURES_DIR / f"features_test_{HEADLINE_LAMBDA}_{p_tag}_hybrid.pkl"

        if args.reuse_p05 and abs(p - 0.5) < 1e-9 and existing_hybrid_train.exists():
            log.info("  Reusing existing features_*_{}_hybrid.pkl for p=0.5".format(HEADLINE_LAMBDA))
            train_df, test_df = hybrid_train, hybrid_test
        else:
            mkt_train, mkt_test = _re_extract_mkt_at_p(p, log)
            train_df = _merge_with_cnn(mkt_train, cnn_train_path, log)
            test_df  = _merge_with_cnn(mkt_test,  cnn_test_path,  log)
            log.info("  Saving %s and %s", train_pickle.name, test_pickle.name)
            train_df.to_pickle(train_pickle)
            test_df.to_pickle(test_pickle)

        log.info("  Training MLP on hybrid features at p=%s...", p)
        metrics = _evaluate(train_df, test_df)
        log.info("  acc=%.4f  sens=%.4f  spec=%.4f  prec=%.4f  f1=%.4f  auc=%.4f",
                 metrics["accuracy"], metrics["sensitivity"],
                 metrics["specificity"], metrics["precision"],
                 metrics["f1"], metrics["auc"])
        summary_rows.append({"p": p, **metrics})

    # ---------- Save summary ----------
    summary_csv = RESULTS_DIR / f"tier2_p_sensitivity_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    log.info("")
    log.info("Summary → %s", summary_csv)


if __name__ == "__main__":
    main()
