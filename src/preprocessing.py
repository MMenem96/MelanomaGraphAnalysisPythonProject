import numpy as np
from PIL import Image
from skimage import color, exposure, morphology, segmentation, filters
from scipy import ndimage
import logging
import os
import cv2
import time

class ImagePreprocessor:
    def __init__(self):
        """Initialize preprocessor with parameters from the paper."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.logger = logging.getLogger(__name__)
        self.target_size = (750, 750)  # Standard size as per paper [52]
        self.clahe_clip_limit = 0.03  # CLAHE parameter from paper
        self.gaussian_sigma = 0.4  # Reduced Gaussian filter sigma for better balance between noise reduction and feature preservation
        
        # Artifact removal parameters
        self.hair_removal_enabled = True
        self.ruler_removal_enabled = True
        self.bubble_removal_enabled = True
        self.artifact_removal_debug = False  # Set to True to save intermediate artifact detection results

    def load_image(self, image_path):
        """Load and validate image."""
        try:
            # Validate file extension
            _, ext = os.path.splitext(image_path)
            if ext.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {ext}")
            
            # Load image
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Step 1: Resample to 750x750 as specified in paper [52]
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            self.logger.info(f"Image resampled to {img.size} pixels")

            return np.array(img)

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Error loading image: {str(e)}")

    def preprocess(self, image):
        """Preprocess image according to paper specifications."""
        try:
            # Ensure input is valid
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")

            # Handle different input formats
            if len(image.shape) == 2:  # Grayscale
                image = color.gray2rgb(image)
            elif image.shape[2] == 4:  # RGBA
                image = image[:,:,:3]
            elif len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")

            # Step 2: Convert to float and normalize to [0,1] range
            image = image.astype(float) / 255.0
            self.logger.info(f"Image normalized to range [0,1]: min={image.min():.3f}, max={image.max():.3f}")
            
            # Step 3: Apply artifact removal before filtering
            if self.hair_removal_enabled or self.ruler_removal_enabled:
                image = self.remove_artifacts(image)
            
            # Step 4: Apply Gaussian filter for noise reduction
            gaussian_filtered = np.zeros_like(image)
            for i in range(image.shape[2]):
                gaussian_filtered[:,:,i] = cv2.GaussianBlur(image[:,:,i], (5, 5), self.gaussian_sigma)
            image = gaussian_filtered
            self.logger.info(f"Applied Gaussian filter with sigma={self.gaussian_sigma}")

            # Step 5: Apply CLAHE contrast enhancement with paper-specified parameters
            image = exposure.equalize_adapthist(image, clip_limit=self.clahe_clip_limit)
            self.logger.info(f"Applied CLAHE with clip_limit={self.clahe_clip_limit}")

            # Step 5: Convert to LAB color space as specified in paper
            image_lab = color.rgb2lab(image)

            # Step 6: Normalize LAB values to [0,1] range as per paper
            # L channel: [0, 100]
            # a channel: [-128, 127]
            # b channel: [-128, 127]
            l_chan = (image_lab[:,:,0]) / 100.0  # L channel normalization
            a_chan = (image_lab[:,:,1] + 128) / 255.0  # a channel normalization
            b_chan = (image_lab[:,:,2] + 128) / 255.0  # b channel normalization

            # Log channel statistics
            self.logger.info(f"LAB Normalization stats:")
            self.logger.info(f"L channel: min={l_chan.min():.3f}, max={l_chan.max():.3f}")
            self.logger.info(f"a channel: min={a_chan.min():.3f}, max={a_chan.max():.3f}")
            self.logger.info(f"b channel: min={b_chan.min():.3f}, max={b_chan.max():.3f}")

            # Stack normalized channels
            image_lab_normalized = np.dstack((l_chan, a_chan, b_chan))

            return image_lab_normalized

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise ValueError(f"Error during preprocessing: {str(e)}")

    def remove_artifacts(self, image):
        """
        Remove hair, ruler, and bubble artifacts from dermoscopic images.
        
        Args:
            image: Input image as numpy array in range [0,1]
            
        Returns:
            Cleaned image with artifacts removed
        """
        try:
            artifacts_removed_count = 0
            original_image = image.copy()
            
            # Convert to uint8 for OpenCV operations
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Hair removal
            if self.hair_removal_enabled:
                image_uint8, hair_detected = self.remove_hair_artifacts(image_uint8)
                if hair_detected:
                    artifacts_removed_count += 1
                    # self.logger.info("Hair artifacts detected and removed")
            
            # Ruler removal
            if self.ruler_removal_enabled:
                image_uint8, ruler_detected = self.remove_ruler_artifacts(image_uint8)
                if ruler_detected:
                    artifacts_removed_count += 1
                    # self.logger.info("Ruler artifacts detected and removed")
            
            # Bubble removal
            if self.bubble_removal_enabled:
                image_uint8, bubble_detected = self.remove_bubble_artifacts(image_uint8)
                if bubble_detected:
                    artifacts_removed_count += 1
                    # self.logger.info("Bubble artifacts detected and removed")
            
            # Convert back to float [0,1]
            cleaned_image = image_uint8.astype(float) / 255.0
            
            # if artifacts_removed_count > 0:
            #     self.logger.info(f"Total artifacts removed: {artifacts_removed_count}")
            # else:
            #     self.logger.info("No artifacts detected")
                
            return cleaned_image
            
        except Exception as e:
            self.logger.warning(f"Error in artifact removal: {str(e)}. Using original image.")
            return image

    def remove_hair_artifacts(self, image):
        """
        Remove hair artifacts using morphological operations and inpainting.
        
        Args:
            image: Input image as uint8 numpy array
            
        Returns:
            tuple: (processed_image, hair_detected_flag)
        """
        try:
            # Convert to grayscale for hair detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Create kernel for black-hat filtering (detects dark linear structures)
            kernel_size = 11  # Reduced from 17 to be more selective
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Apply black-hat transform to detect dark linear structures (hair)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold to create binary mask of hair regions
            _, hair_mask = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up the mask
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel_small)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Check if significant hair artifacts were detected
            hair_pixels = np.sum(hair_mask > 0)
            total_pixels = hair_mask.shape[0] * hair_mask.shape[1]
            hair_ratio = hair_pixels / total_pixels
            
            hair_detected = False  # Temporarily disable hair removal to preserve natural texture
            
            if hair_detected:
                # Apply inpainting to remove detected hair
                if len(image.shape) == 3:
                    # For color images, apply inpainting to each channel
                    result = image.copy()
                    for channel in range(3):
                        result[:,:,channel] = cv2.inpaint(image[:,:,channel], hair_mask, 3, cv2.INPAINT_TELEA)
                else:
                    # For grayscale images
                    result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
                
                # Optional: Save debug image
                if self.artifact_removal_debug:
                    self._save_debug_image(hair_mask, "hair_mask")
                    
                return result, True
            else:
                return image, False
                
        except Exception as e:
            self.logger.warning(f"Error in hair removal: {str(e)}")
            return image, False

    def remove_ruler_artifacts(self, image):
        """
        Remove ruler and measurement artifacts using edge detection and geometric analysis.
        
        Args:
            image: Input image as uint8 numpy array
            
        Returns:
            tuple: (processed_image, ruler_detected_flag)
        """
        try:
            # Convert to grayscale for edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # Edge detection using Canny
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=100, maxLineGap=10)
            
            ruler_detected = False
            ruler_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            if lines is not None and len(lines) > 0:
                # Analyze detected lines for ruler characteristics
                long_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # Check for long straight lines (potential rulers)
                    if length > 200:  # Minimum length for ruler consideration
                        # Check if line is mostly horizontal or vertical
                        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                        if angle < 10 or angle > 170 or (80 < angle < 100):  # Horizontal or vertical
                            long_lines.append(line[0])
                
                # If we found potential ruler lines, create mask
                if len(long_lines) > 0:
                    ruler_detected = True
                    
                    for line in long_lines:
                        x1, y1, x2, y2 = line
                        # Create thick line mask (rulers have width)
                        cv2.line(ruler_mask, (x1, y1), (x2, y2), 255, thickness=15)
                    
                    # Morphological operations to expand ruler mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    ruler_mask = cv2.morphologyEx(ruler_mask, cv2.MORPH_DILATE, kernel)
            
            # Additional check for text/numbers (common on rulers)
            # Look for high-contrast rectangular regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Text-sized regions
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check for text-like rectangles
                    if 0.2 < aspect_ratio < 5.0:  # Text aspect ratios
                        cv2.rectangle(ruler_mask, (x-5, y-5), (x+w+5, y+h+5), 255, -1)
                        ruler_detected = True
            
            if ruler_detected:
                # Apply inpainting to remove detected ruler artifacts
                if len(image.shape) == 3:
                    result = image.copy()
                    for channel in range(3):
                        result[:,:,channel] = cv2.inpaint(image[:,:,channel], ruler_mask, 5, cv2.INPAINT_TELEA)
                else:
                    result = cv2.inpaint(image, ruler_mask, 5, cv2.INPAINT_TELEA)
                
                # Optional: Save debug image
                if self.artifact_removal_debug:
                    self._save_debug_image(ruler_mask, "ruler_mask")
                    
                return result, True
            else:
                return image, False
                
        except Exception as e:
            self.logger.warning(f"Error in ruler removal: {str(e)}")
            return image, False

    def remove_bubble_artifacts(self, image):
        """
        Remove bubble artifacts using circular Hough transform detection.
        
        Args:
            image: Input image as uint8 numpy array
            
        Returns:
            tuple: (processed_image, bubble_detected_flag)
        """
        try:
            bubble_detected = False
            result = image.copy()
            
            # Convert to grayscale for bubble detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Create bubble mask
            bubble_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            # Apply Gaussian blur to reduce noise for circle detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect circles using HoughCircles
            # Parameters tuned for dermoscopic bubble detection
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,                    # Inverse ratio of accumulator resolution
                minDist=30,              # Minimum distance between circle centers
                param1=50,               # Upper threshold for edge detection
                param2=30,               # Accumulator threshold for center detection
                minRadius=5,             # Minimum circle radius
                maxRadius=100            # Maximum circle radius
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Validate circle is within image bounds
                    if (x-r >= 0 and y-r >= 0 and 
                        x+r < image.shape[1] and y+r < image.shape[0]):
                        
                        # Check if this region looks like a bubble
                        # Bubbles are typically bright, circular regions
                        roi = gray[y-r:y+r, x-r:x+r]
                        if roi.size > 0:
                            mean_intensity = np.mean(roi)
                            # Bubbles are usually brighter than surrounding skin
                            if mean_intensity > 140:  # Bright regions
                                # Add to bubble mask with some padding
                                cv2.circle(bubble_mask, (x, y), r+5, 255, -1)
                                bubble_detected = True
            
            # Additional detection: Look for bright oval/circular regions
            # using morphological operations
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Find contours of bright regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Bubble-sized areas
                    # Check if contour is roughly circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # Values close to 1.0 indicate circular shapes
                        if circularity > 0.6:  # Reasonably circular
                            cv2.fillPoly(bubble_mask, [contour], 255)
                            bubble_detected = True
            
            if bubble_detected:
                # Apply inpainting to remove detected bubbles
                if len(image.shape) == 3:
                    result = image.copy()
                    for channel in range(3):
                        result[:,:,channel] = cv2.inpaint(image[:,:,channel], bubble_mask, 3, cv2.INPAINT_TELEA)
                else:
                    result = cv2.inpaint(image, bubble_mask, 3, cv2.INPAINT_TELEA)
                
                # Optional: Save debug image
                if self.artifact_removal_debug:
                    self._save_debug_image(bubble_mask, "bubble_mask")
                    
                return result, True
            else:
                return image, False
                
        except Exception as e:
            self.logger.warning(f"Error in bubble removal: {str(e)}")
            return image, False

    def _save_debug_image(self, mask, artifact_type):
        """Save debug images for artifact detection analysis."""
        try:
            debug_dir = "artifact_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"{debug_dir}/{artifact_type}_{timestamp}.png"
            cv2.imwrite(filename, mask)
            self.logger.debug(f"Saved debug image: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Could not save debug image: {str(e)}")