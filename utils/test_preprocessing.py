"""
Advanced Dermoscopic Image Preprocessing Pipeline
Optimized for BCC vs SK classification with robust ROI detection
"""

import cv2
import numpy as np
import os
import argparse
import logging
from typing import Tuple, Optional

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_image_characteristics(image: np.ndarray, logger) -> dict:
    """
    Analyze image characteristics to determine best preprocessing approach.
    
    Args:
        image: Input BGR image
        logger: Logger instance
        
    Returns:
        Dictionary with image analysis results
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze color distribution
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    l_channel = lab[:,:,0]
    a_channel = lab[:,:,1]
    b_channel = lab[:,:,2]
    
    analysis = {
        'dominant_hue': np.median(h_channel),
        'saturation_range': np.std(s_channel),
        'brightness_range': np.std(v_channel),
        'contrast_level': np.std(gray),
        'color_variance': np.std(a_channel) + np.std(b_channel),
        'has_dark_regions': np.percentile(v_channel, 10) < 50,
        'has_bright_regions': np.percentile(v_channel, 90) > 200,
        'is_low_contrast': np.std(gray) < 30
    }
    
    logger.info(f"Image analysis - Contrast: {analysis['contrast_level']:.1f}, "
               f"Color variance: {analysis['color_variance']:.1f}, "
               f"Hue: {analysis['dominant_hue']:.0f}")
    
    return analysis

def remove_hair_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Remove hair artifacts using morphological operations and inpainting.
    
    Args:
        image: Input BGR image
        
    Returns:
        Image with hair artifacts removed
    """
    # Convert to grayscale for hair detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create kernel for hair detection (thin linear structures)
    kernel_hair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 9))
    
    # Detect dark linear structures (hairs)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hair)
    
    # Threshold to create hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly to ensure complete coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    
    # Inpaint hair regions
    result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
    
    return result

def enhance_contrast_adaptive(image: np.ndarray, analysis: dict) -> np.ndarray:
    """
    Apply adaptive contrast enhancement based on image characteristics.
    
    Args:
        image: Input BGR image
        analysis: Image analysis results
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space for better contrast control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Adaptive CLAHE parameters based on image characteristics
    if analysis['is_low_contrast']:
        clip_limit = 3.0
        tile_size = (6, 6)
    elif analysis['contrast_level'] > 50:
        clip_limit = 1.5
        tile_size = (8, 8)
    else:
        clip_limit = 2.0
        tile_size = (4, 4)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_channel = clahe.apply(l_channel)
    
    # Reconstruct image
    lab[:,:,0] = l_channel
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_lesion_region_multi_method(image: np.ndarray, analysis: dict, logger) -> Tuple[np.ndarray, str]:
    """
    Detect lesion region using multiple methods and select the best one.
    
    Args:
        image: Input BGR image
        analysis: Image analysis results
        logger: Logger instance
        
    Returns:
        Tuple of (mask, method_used)
    """
    h, w = image.shape[:2]
    
    # Method 1: Center crop (always reliable)
    center_mask = create_center_crop_mask(image, coverage=0.75)
    
    # Method 2: Color-based segmentation for well-contrasted lesions
    color_mask = None
    if analysis['contrast_level'] > 25 and analysis['color_variance'] > 15:
        try:
            color_mask = segment_by_color_advanced(image, analysis)
            
            # Validate color segmentation
            coverage = np.sum(color_mask > 0) / (h * w)
            if coverage < 0.02 or coverage > 0.7:  # Too small or too large
                color_mask = None
        except:
            color_mask = None
    
    # Method 3: Edge-based detection for dark lesions
    edge_mask = None
    if analysis['has_dark_regions'] and analysis['contrast_level'] > 30:
        try:
            edge_mask = segment_by_edges(image)
            
            # Validate edge segmentation
            coverage = np.sum(edge_mask > 0) / (h * w)
            if coverage < 0.01 or coverage > 0.6:
                edge_mask = None
        except:
            edge_mask = None
    
    # Select best method
    if color_mask is not None:
        logger.info("Using color-based lesion segmentation")
        return color_mask, "color"
    elif edge_mask is not None:
        logger.info("Using edge-based lesion segmentation")
        return edge_mask, "edge"
    else:
        logger.info("Using reliable center-crop segmentation")
        return center_mask, "center_crop"

def create_center_crop_mask(image: np.ndarray, coverage: float = 0.75) -> np.ndarray:
    """Create a center crop mask."""
    h, w = image.shape[:2]
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = int(h * coverage), int(w * coverage)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = max(0, center_h - crop_h // 2)
    y2 = min(h, center_h + crop_h // 2)
    x1 = max(0, center_w - crop_w // 2)
    x2 = min(w, center_w + crop_w // 2)
    
    mask[y1:y2, x1:x2] = 255
    return mask

def segment_by_color_advanced(image: np.ndarray, analysis: dict) -> np.ndarray:
    """Advanced color-based segmentation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h, w = image.shape[:2]
    
    # LAB channel analysis
    a_channel = lab[:,:,1]
    b_channel = lab[:,:,2]
    
    # Adaptive thresholding based on dominant hue
    if analysis['dominant_hue'] < 30 or analysis['dominant_hue'] > 150:  # Red/Brown regions
        # Target reddish/brownish areas
        a_thresh = np.percentile(a_channel, 60)
        _, mask_a = cv2.threshold(a_channel, int(a_thresh), 255, cv2.THRESH_BINARY)
        
        b_thresh = np.percentile(b_channel, 55)
        _, mask_b = cv2.threshold(b_channel, int(b_thresh), 255, cv2.THRESH_BINARY)
        
        color_mask = cv2.bitwise_or(mask_a, mask_b)
    else:  # Other colors
        # Use saturation and value
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]
        
        s_thresh = np.percentile(s_channel, 40)
        _, mask_s = cv2.threshold(s_channel, int(s_thresh), 255, cv2.THRESH_BINARY)
        
        v_thresh = np.percentile(v_channel, 30)
        _, mask_v = cv2.threshold(v_channel, int(v_thresh), 255, cv2.THRESH_BINARY_INV)
        
        color_mask = cv2.bitwise_and(mask_s, mask_v)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    # Keep largest connected component
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(color_mask)
        cv2.fillPoly(mask, [largest_contour], 255)
        return mask
    
    return color_mask

def segment_by_edges(image: np.ndarray) -> np.ndarray:
    """Edge-based segmentation for dark lesions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold for edge detection
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # Find contours and keep largest
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [largest_contour], 255)
        return mask
    
    return edges

def apply_roi_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply ROI mask to image."""
    if len(image.shape) == 3:
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(image, mask_3d)
    else:
        return cv2.bitwise_and(image, mask)

def comprehensive_preprocessing_pipeline(image_path: str, output_dir: str = "test_preprocessing_images"):
    """
    Comprehensive preprocessing pipeline for dermoscopic images.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for processed images
        
    Returns:
        bool: Success status
    """
    logger = setup_logging()
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return False
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Step 1: Resize for consistent processing
        target_height = 512
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image to {target_width}x{target_height}")
        
        # Save original
        original_path = os.path.join(output_dir, f"{filename}_1_original.jpg")
        cv2.imwrite(original_path, image)
        
        # Step 2: Analyze image characteristics
        analysis = analyze_image_characteristics(image, logger)
        
        # Step 3: Remove hair artifacts if detected
        hair_removed = remove_hair_artifacts(image)
        hair_removed_path = os.path.join(output_dir, f"{filename}_2_hair_removed.jpg")
        cv2.imwrite(hair_removed_path, hair_removed)
        logger.info("Applied hair artifact removal")
        
        # Step 4: Adaptive contrast enhancement
        enhanced = enhance_contrast_adaptive(hair_removed, analysis)
        enhanced_path = os.path.join(output_dir, f"{filename}_3_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        logger.info("Applied adaptive contrast enhancement")
        
        # Step 5: Lesion region detection
        lesion_mask, method = detect_lesion_region_multi_method(enhanced, analysis, logger)
        
        # Step 6: Apply ROI mask
        final_image = apply_roi_mask(enhanced, lesion_mask)
        final_path = os.path.join(output_dir, f"{filename}_4_final_processed.jpg")
        cv2.imwrite(final_path, final_image)
        
        # Step 7: Create visualization
        visualization = enhanced.copy()
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        viz_path = os.path.join(output_dir, f"{filename}_5_roi_visualization.jpg")
        cv2.imwrite(viz_path, visualization)
        
        logger.info(f"Preprocessing completed using {method} method")
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return False

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Advanced dermoscopic image preprocessing")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output_dir", default="test_preprocessing_images", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return
    
    success = comprehensive_preprocessing_pipeline(args.image_path, args.output_dir)
    
    if success:
        print(f"Preprocessing completed successfully. Check {args.output_dir} for results.")
    else:
        print("Preprocessing failed. Check logs for details.")

if __name__ == "__main__":
    main()





"""
Idea that built.

Key Intelligence Features:
1. Image Analysis (Lines 21-60):

Analyzes contrast, color variance, dominant hue
Determines optimal processing approach for each image
2. Multi-Method ROI Detection (Lines 129-181):

Color-based segmentation for high-contrast lesions
Edge-based detection for dark lesions
Center-crop fallback for challenging cases
Automatically selects best method
3. Adaptive Enhancement (Lines 93-127):

Adjusts CLAHE parameters based on image characteristics
Low contrast → stronger enhancement
High contrast → gentler processing
4. Hair Artifact Removal (Lines 62-91):

Detects and removes hair using morphological operations
Uses inpainting to fill removed areas naturally
5. Intelligent Processing Pipeline (Lines 281-369):

Orchestrates entire workflow
Saves intermediate steps for analysis
Creates visualization with ROI boundaries
How It Works:
The system automatically analyzes each dermoscopic image and chooses the optimal processing method:

High contrast + good color variance → Uses color-based detection
Dark regions + good contrast → Uses edge-based detection
Low contrast or challenging features → Uses reliable center-crop
This gives you consistent, high-quality preprocessing across your entire BCC vs SK dataset, handling all the diverse lesion types you showed me.

The preprocessing pipeline is now ready for integration into your main training workflow for optimal feature extraction and classification performance.

"""
