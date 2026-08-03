"""
Combine every feature family into one representation for the 7-class arm.

    handcrafted + MKT        461   (geometric, colour, texture, Krawtchouk)
    frozen ResNet-50       2,048   (ImageNet, never adapted to dermoscopy)
    fine-tuned EfficientNet 1,280  (adapted to these lesions)
    patient metadata           19  (age, sex, localization)
    -------------------------------
    total                  3,808

Frozen and fine-tuned deep features are kept *together* rather than one
replacing the other: they encode different things (generic ImageNet texture vs
lesion-specific structure) and mutual-information selection can then choose
between them instead of us guessing.

Row alignment is asserted on (source_id, aug_tag) before anything is merged —
all inputs derive from the same base pickles in the same order.

Usage:
    python paper_pipeline/pipeline/combine7.py --suffix lesion_equalize
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("combine7")
BOOKKEEPING = {"source_id", "aug_tag", "label", "lesion_id", "dx"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--suffix", default="lesion_equalize")
    ap.add_argument("--lambdas", default="odd_harmonic")
    ap.add_argument("--out-tag", default="all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    lam = args.lambdas.split(",")[0].strip()

    for split in ("train", "test"):
        base_p = args.features_dir / f"features_{split}_{lam}_{args.suffix}_hybrid_meta.pkl"
        ft_p = args.features_dir / f"features_{split}_{lam}_{args.suffix}_ft.pkl"
        if not base_p.is_file() or not ft_p.is_file():
            LOG.error("missing input: %s / %s", base_p.name, ft_p.name)
            return 2

        base = pd.read_pickle(base_p)          # 461 + 2048 + 19 metadata
        ft = pd.read_pickle(ft_p)              # 461 + 1280 fine-tuned

        if len(base) != len(ft):
            LOG.error("%s row mismatch: %d vs %d", split, len(base), len(ft))
            return 1
        aligned = ((base["source_id"].to_numpy() == ft["source_id"].to_numpy()).all()
                   and (base["aug_tag"].to_numpy() == ft["aug_tag"].to_numpy()).all())
        if not aligned:
            LOG.error("%s: row alignment broken between the two feature sets", split)
            return 1

        ft_cols = [c for c in ft.columns if c.startswith("ftcnn_")]
        merged = pd.concat([base.reset_index(drop=True),
                            ft[ft_cols].reset_index(drop=True)], axis=1)
        n_feat = merged.shape[1] - len(BOOKKEEPING)
        out = args.features_dir / f"features_{split}_{lam}_{args.suffix}_{args.out_tag}.pkl"
        merged.to_pickle(out)
        LOG.info("%s: %d + %d fine-tuned = %d features  →  %s",
                 split, base.shape[1] - len(BOOKKEEPING), len(ft_cols), n_feat, out.name)

    LOG.info("Done. Next: train_eval7.py --suffix %s_%s", args.suffix, args.out_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
