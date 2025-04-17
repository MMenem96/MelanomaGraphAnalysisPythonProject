import cv2
import numpy as np
from skimage import color
import logging

class ImageValidator:
    def __init__(self):
        """Initialize the image validator with logging."""
        self.logger = logging.getLogger(__name__)
        
    def validate_skin_image(self, image):
        """
        Validate if an image is likely a skin lesion image.
        
        Args:
            image: The input image in numpy array format
            
        Returns:
            tuple: (is_valid, reason)
                - is_valid: Boolean indicating if image is valid
                - reason: String explaining why image is invalid (if applicable)
        """
        try:
            # 1. Check if image is not empty
            if image is None or image.size == 0:
                return False, "Empty image"
                
            # 2. Check if image has reasonable dimensions
            if image.shape[0] < 50 or image.shape[1] < 50:
                return False, "Image too small (less than 50x50 pixels)"
                
            # 3. Check if image has colors (not grayscale)
            if len(image.shape) < 3 or image.shape[2] < 3:
                return False, "Image must be color (RGB)"
                
            # 4. Check for skin color presence
            is_skin, skin_reason = self._check_skin_color_presence(image)
            if not is_skin:
                return False, skin_reason
                
            # 5. Check for lesion characteristics
            has_lesion, lesion_reason = self._check_lesion_characteristics(image)
            if not has_lesion:
                return False, lesion_reason
                
            # All validations passed
            return True, "Valid skin lesion image"
            
        except Exception as e:
            self.logger.error(f"Error validating image: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def _check_skin_color_presence(self, image):
        """
        Check if the image contains significant amount of skin-colored pixels.
        """
        try:
            # Convert image to HSV color space for better skin detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define skin color range in HSV
            # These ranges are approximations and may need adjustment
            lower_skin_hsv1 = np.array([0, 20, 70])
            upper_skin_hsv1 = np.array([20, 150, 255])
            
            lower_skin_hsv2 = np.array([170, 20, 70])
            upper_skin_hsv2 = np.array([180, 150, 255])
            
            # Create masks for skin detection
            mask1 = cv2.inRange(hsv_image, lower_skin_hsv1, upper_skin_hsv1)
            mask2 = cv2.inRange(hsv_image, lower_skin_hsv2, upper_skin_hsv2)
            skin_mask = cv2.bitwise_or(mask1, mask2)
            
            # Calculate percentage of skin pixels
            skin_percentage = (np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])) * 100
            
            # Require at least 30% of the image to be skin-colored
            if skin_percentage < 30:
                return False, f"Insufficient skin color detected ({skin_percentage:.1f}%)"
                
            return True, "Sufficient skin color detected"
            
        except Exception as e:
            self.logger.error(f"Error checking skin color: {str(e)}")
            return False, f"Skin color check error: {str(e)}"
    
    def _check_lesion_characteristics(self, image):
        """
        Check if the image contains characteristics of a skin lesion.
        """
        try:
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply blurring to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply thresholding to find dark regions (potential lesions)
            # Use Otsu's method for automatic thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size to find potential lesions
            img_area = image.shape[0] * image.shape[1]
            min_lesion_area = img_area * 0.05  # Lesion should be at least 5% of image
            max_lesion_area = img_area * 0.9   # Lesion shouldn't be more than 90% of image
            
            lesion_contours = [c for c in contours if min_lesion_area < cv2.contourArea(c) < max_lesion_area]
            
            # Check if we found at least one lesion
            if not lesion_contours:
                return False, "No potential lesion found in the image"
                
            # Check for color variation within the largest contour (lesions usually have some color variation)
            largest_contour = max(lesion_contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            
            # Apply mask to the original image and calculate color variance
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            
            # Calculate color variation within the lesion
            non_zero_pixels = masked_img[np.where(mask > 0)]
            if non_zero_pixels.size > 0:
                color_std = np.std(non_zero_pixels, axis=0)
                color_variation = np.mean(color_std)
                
                # Threshold for color variation (lesions typically have some color variation)
                if color_variation < 5.0:
                    return False, "Insufficient color variation in the potential lesion"
            
            return True, "Lesion characteristics detected"
            
        except Exception as e:
            self.logger.error(f"Error checking lesion characteristics: {str(e)}")
            return False, f"Lesion check error: {str(e)}"