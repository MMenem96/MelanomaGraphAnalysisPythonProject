"""
Image preprocessing wrapper.

Encapsulates the active preprocessing path used in the original pipeline:
  1. Convert to grayscale for hair detection.
  2. Combined Black-Hat + Top-Hat morphology to detect hair.
  3. Telea inpainting (cv2.INPAINT_TELEA, radius 1) over the detected mask.
  4. Gaussian blur for residual-noise smoothing.

Hair removal IS enabled here — this is the path the paper describes. The
disabled `hair_detected = False` line in src/preprocessing.py only affects
the unused graph-pipeline path; the feature-extraction route used in the
paper goes through `SkinLesionProcessor.apply_inpainting` which already
uses `cv2.INPAINT_TELEA`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the library class. The model weights are loaded once when the
# module is imported by the orchestrator.
from src.segmentation.skin_lesion_processor import SkinLesionProcessor  # noqa: E402


_processor_singleton: SkinLesionProcessor | None = None


def get_processor() -> SkinLesionProcessor:
    """Lazy singleton — model weights load once per process."""
    global _processor_singleton
    if _processor_singleton is None:
        _processor_singleton = SkinLesionProcessor()
    return _processor_singleton


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply the canonical preprocessing chain to a single RGB image.

    Args:
        image_rgb: H×W×3 uint8 array, BGR or RGB (cv2 default is BGR).

    Returns:
        H×W×3 uint8 array after hair removal + Gaussian blur.
    """
    proc = get_processor()
    grayscale = proc.convert_to_grayscale(image_rgb)
    combined_hair_mask, _blackhat, _tophat = proc.apply_combined_hair_detection(grayscale)
    inpainted = proc.apply_inpainting(image_rgb, combined_hair_mask)
    smoothed = proc.apply_gaussian_blur(inpainted)
    return smoothed
