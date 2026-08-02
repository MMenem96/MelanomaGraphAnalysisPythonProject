"""
Train-only image-level augmentation.

The augmentation is applied at **feature-extraction time**, on the training
subset only, after the stratified train/test split has already happened on
image IDs. This eliminates the train/test leakage of the previous design
where flipped copies could land on opposite sides of the split.

Each training image yields three feature rows:
    1. original
    2. horizontal flip   ( cv2.flip(img, 1) )
    3. vertical flip     ( cv2.flip(img, 0) )

Test images are NEVER flipped.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

# Augmentation tags appear in the saved feature pickle so leakage can be re-verified.
TAG_ORIGINAL = "orig"
TAG_HFLIP = "h_flip"
TAG_VFLIP = "v_flip"
TAG_ROT_P15 = "rot_p15"
TAG_ROT_M15 = "rot_m15"
TRAIN_TAGS: tuple[str, ...] = (TAG_ORIGINAL, TAG_HFLIP, TAG_VFLIP)
TEST_TAGS: tuple[str, ...] = (TAG_ORIGINAL,)


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate around centre, preserve size. White (255) fill for canvas."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )


def variants(image: np.ndarray, tags: Iterable[str]) -> list[tuple[str, np.ndarray]]:
    """
    Produce (tag, image-variant) pairs.

    For training set: tags ⊆ {orig, h_flip, v_flip, rot_p15, rot_m15}.
    For test set:     tags = TEST_TAGS   → 1 variant.

    Mask transforms accompany image transforms via `mask_variants`.
    """
    out: list[tuple[str, np.ndarray]] = []
    for t in tags:
        if t == TAG_ORIGINAL:
            out.append((t, image))
        elif t == TAG_HFLIP:
            out.append((t, cv2.flip(image, 1)))
        elif t == TAG_VFLIP:
            out.append((t, cv2.flip(image, 0)))
        elif t == TAG_ROT_P15:
            out.append((t, _rotate(image, +15.0)))
        elif t == TAG_ROT_M15:
            out.append((t, _rotate(image, -15.0)))
        else:
            raise ValueError(f"Unknown augmentation tag {t!r}")
    return out


def mask_variants(mask: np.ndarray, tags: Iterable[str]) -> list[tuple[str, np.ndarray]]:
    """Flip the binary mask the same way as the image so features stay aligned."""
    return variants(mask, tags)
