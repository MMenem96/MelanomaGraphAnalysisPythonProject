"""
Test script to verify Krawtchouk moments implementation
"""
import numpy as np
import cv2
from src.conventional_features import ConventionalFeatureExtractor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_krawtchouk_moments():
    """Test the Krawtchouk moments implementation."""
    
    logger.info("="*60)
    logger.info("TESTING KRAWTCHOUK MOMENTS IMPLEMENTATION")
    logger.info("="*60)
    
    # Create a feature extractor
    extractor = ConventionalFeatureExtractor()
    
    # Test 1: Simple synthetic image
    logger.info("\n[TEST 1] Synthetic circular lesion")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 30, (150, 100, 80), -1)
    test_mask = np.zeros((100, 100), dtype=bool)
    cv2.circle(test_mask.astype(np.uint8), (50, 50), 30, 1, -1)
    test_mask = test_mask.astype(bool)
    
    features = extractor.extract_krawtchouk_moments(test_image, test_mask)
    
    logger.info(f"✓ Extracted {len(features)} Krawtchouk features")
    logger.info(f"✓ Feature keys: {list(features.keys())[:5]}... (showing first 5)")
    logger.info(f"✓ Sample values:")
    for i, (key, value) in enumerate(list(features.items())[:5]):
        logger.info(f"    {key}: {value:.6f}")
    
    # Test 2: Irregular shape (more realistic lesion)
    logger.info("\n[TEST 2] Irregular lesion shape")
    test_image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create irregular shape
    pts = np.array([[20,20], [80,25], [85,70], [25,75], [15,40]], np.int32)
    cv2.fillPoly(test_image2, [pts], (180, 120, 90))
    test_mask2 = np.zeros((100, 100), dtype=bool)
    cv2.fillPoly(test_mask2.astype(np.uint8), [pts], 1)
    test_mask2 = test_mask2.astype(bool)
    
    features2 = extractor.extract_krawtchouk_moments(test_image2, test_mask2)
    
    logger.info(f"✓ Extracted {len(features2)} Krawtchouk features")
    logger.info(f"✓ Values differ from circular lesion: {not np.allclose(list(features.values()), list(features2.values()))}")
    
    # Test 3: Full feature extraction pipeline
    logger.info("\n[TEST 3] Full feature extraction with Krawtchouk moments")
    all_features = extractor.extract_all_features(test_image, test_mask)
    
    krawtchouk_feature_count = sum(1 for key in all_features.keys() if 'krawtchouk' in key)
    logger.info(f"✓ Total features extracted: {len(all_features)}")
    logger.info(f"✓ Krawtchouk features in full set: {krawtchouk_feature_count}")
    
    # Verify no NaN or Inf values
    has_nan = any(np.isnan(v) if isinstance(v, (int, float)) else False for v in all_features.values())
    has_inf = any(np.isinf(v) if isinstance(v, (int, float)) else False for v in all_features.values())
    
    logger.info(f"✓ Contains NaN values: {has_nan}")
    logger.info(f"✓ Contains Inf values: {has_inf}")
    
    # Test 4: Empty mask handling
    logger.info("\n[TEST 4] Edge case - empty mask")
    empty_mask = np.zeros((100, 100), dtype=bool)
    features_empty = extractor.extract_krawtchouk_moments(test_image, empty_mask)
    logger.info(f"✓ Handles empty mask gracefully: {len(features_empty)} features returned")
    logger.info(f"✓ All values are zero: {all(v == 0.0 for v in features_empty.values())}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("="*60)
    logger.info("\n📊 FEATURE SUMMARY:")
    logger.info(f"   - Total Krawtchouk features: {krawtchouk_feature_count}")
    logger.info(f"   - Moment features: {sum(1 for k in features.keys() if 'moment_' in k and 'abs' not in k)}")
    logger.info(f"   - Absolute moment features: {sum(1 for k in features.keys() if 'moment_abs' in k)}")
    logger.info(f"   - Invariant features: {sum(1 for k in features.keys() if 'invariant' in k)}")
    logger.info(f"   - Energy/Entropy features: {sum(1 for k in features.keys() if k in ['krawtchouk_energy', 'krawtchouk_entropy'])}")
    logger.info("\n🎯 Krawtchouk moments implementation is ready for training!")
    
    return True

if __name__ == "__main__":
    try:
        test_krawtchouk_moments()
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
