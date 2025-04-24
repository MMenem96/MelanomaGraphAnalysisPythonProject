import cv2
import numpy as np
from skimage import color, feature, measure, morphology, segmentation
from scipy import ndimage
import logging

class DermoscopicFeatureDetector:
    """
    Specialized feature detector for dermoscopic patterns that are particularly
    relevant for distinguishing BCC (Basal Cell Carcinoma) from SK (Seborrheic Keratosis).
    
    Key dermoscopic features include:
    - Blue-white veil detection (more common in BCC)
    - Leaf-like structures (characteristic of BCC)
    - Milia-like cysts (characteristic of SK)
    - Arborizing vessels (highly specific for BCC)
    - Brain-like appearance (typical of SK)
    """
    
    def __init__(self):
        """Initialize the dermoscopic feature detector."""
        self.logger = logging.getLogger(__name__)
        
    def detect_all_features(self, image, mask=None):
        """
        Extract dermoscopic features relevant for BCC vs SK differentiation.
        
        Args:
            image: RGB image array
            mask: Boolean mask of the lesion (True for lesion pixels)
            
        Returns:
            Dictionary of dermoscopic features
        """
        try:
            features = {}
            
            # If mask is not provided, use the entire image
            if mask is None:
                mask = np.ones(image.shape[:2], dtype=bool)
                
            # Extract blue-white veil features
            bw_features = self.detect_blue_white_veil(image, mask)
            features.update(bw_features)
            
            # Extract arborizing vessel features
            vessel_features = self.detect_arborizing_vessels(image, mask)
            features.update(vessel_features)
            
            # Extract milia-like cyst features (common in SK)
            milia_features = self.detect_milia_like_cysts(image, mask)
            features.update(milia_features)
            
            # Extract leaf-like structure features (characteristic of BCC)
            leaf_features = self.detect_leaf_like_structures(image, mask)
            features.update(leaf_features)
            
            # Extract brain-like appearance features (typical of SK)
            brain_features = self.detect_brain_like_appearance(image, mask)
            features.update(brain_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting dermoscopic features: {str(e)}")
            return {}
            
    def detect_blue_white_veil(self, image, mask):
        """
        Detect blue-white veil structures which are more common in BCC.
        
        Method: Use color thresholding in LAB space to detect bluish areas,
        then calculate their proportion and distribution.
        """
        try:
            features = {}
            
            # Convert to LAB color space for better color discrimination
            lab_image = color.rgb2lab(image[:,:,:3])
            
            # Blue-white veil shows as lowered a* values (greenish) and lowered b* values (bluish)
            a_channel = lab_image[:,:,1]
            b_channel = lab_image[:,:,2]
            
            # Thresholds determined empirically for blue-white veil detection
            blue_white_mask = (a_channel < -5) & (b_channel < -15) & mask
            
            # Calculate proportion of blue-white veil within the lesion
            if np.sum(mask) > 0:
                features['blue_white_veil_proportion'] = float(np.sum(blue_white_mask) / np.sum(mask))
            else:
                features['blue_white_veil_proportion'] = 0.0
                
            # Calculate distribution properties (clustering of blue-white areas)
            if np.sum(blue_white_mask) > 0:
                # Label connected components
                labeled_mask, num_labels = ndimage.label(blue_white_mask)
                
                if num_labels > 0:
                    # Size statistics of blue-white regions
                    region_sizes = ndimage.sum(blue_white_mask, labeled_mask, range(1, num_labels+1))
                    features['blue_white_region_count'] = float(num_labels)
                    features['blue_white_max_region_size'] = float(np.max(region_sizes) if len(region_sizes) > 0 else 0)
                    features['blue_white_mean_region_size'] = float(np.mean(region_sizes) if len(region_sizes) > 0 else 0)
                else:
                    features['blue_white_region_count'] = 0.0
                    features['blue_white_max_region_size'] = 0.0
                    features['blue_white_mean_region_size'] = 0.0
            else:
                features['blue_white_region_count'] = 0.0
                features['blue_white_max_region_size'] = 0.0
                features['blue_white_mean_region_size'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting blue-white veil: {str(e)}")
            return {
                'blue_white_veil_proportion': 0.0,
                'blue_white_region_count': 0.0,
                'blue_white_max_region_size': 0.0,
                'blue_white_mean_region_size': 0.0
            }
    
    def detect_arborizing_vessels(self, image, mask):
        """
        Detect arborizing vessels which are highly specific for BCC.
        
        Method: Use multiscale vesselness filter to enhance blood vessels,
        then analyze their branching patterns.
        """
        try:
            features = {}
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            # Ensure values are in [0, 1] range
            if np.max(gray) > 1.0:
                gray = gray / 255.0
                
            # Apply contrast enhancement to better visualize vessels
            gray_enhanced = (gray * 255).astype(np.uint8)
            gray_enhanced = cv2.equalizeHist(gray_enhanced)
            gray_enhanced = gray_enhanced.astype(float) / 255.0
            
            # Create masked version
            masked_gray = gray_enhanced.copy()
            masked_gray[~mask] = 0

            # Apply Frangi filter to enhance vessel-like structures at different scales
            # This is a simplified version as the actual Frangi filter implementation is complex
            vessel_mask = np.zeros_like(masked_gray, dtype=bool)
            
            # Use multiple scales of Hessian determinant to detect vessels of different sizes
            for sigma in [1, 2, 3]:
                # Calculate Gaussian gradients
                gx = ndimage.gaussian_filter(masked_gray, sigma, order=(0, 1))
                gy = ndimage.gaussian_filter(masked_gray, sigma, order=(1, 0))
                
                # Calculate Hessian elements
                gxx = ndimage.gaussian_filter(gx, sigma, order=(0, 1))
                gyy = ndimage.gaussian_filter(gy, sigma, order=(1, 0))
                gxy = ndimage.gaussian_filter(gx, sigma, order=(1, 0))
                
                # Simple vesselness measure based on Hessian determinant and trace
                # More sophisticated approaches exist but this captures the essence
                det = gxx * gyy - gxy**2
                trace = gxx + gyy
                
                # Vessel-like structures have negative determinant
                # and high absolute eigenvalues (measured via trace)
                vessel_sigma = (det < -0.0001) & (np.abs(trace) > 0.001) & mask
                vessel_mask = vessel_mask | vessel_sigma
            
            # Calculate vessel features
            if np.sum(mask) > 0:
                features['vessel_proportion'] = float(np.sum(vessel_mask) / np.sum(mask))
            else:
                features['vessel_proportion'] = 0.0
                
            # Analyze vessel branching pattern
            if np.sum(vessel_mask) > 0:
                # Skeletonize the vessel mask to get vessel centerlines
                vessel_skeleton = morphology.skeletonize(vessel_mask)
                
                # Find branch points (pixels with more than 2 neighbors)
                skel_padded = np.pad(vessel_skeleton, pad_width=1, mode='constant', constant_values=0)
                branch_points = np.zeros_like(vessel_skeleton, dtype=bool)
                
                for i in range(1, skel_padded.shape[0]-1):
                    for j in range(1, skel_padded.shape[1]-1):
                        if skel_padded[i, j]:
                            # Count neighbors
                            neighbors = np.sum(skel_padded[i-1:i+2, j-1:j+2]) - 1
                            if neighbors > 2:
                                branch_points[i-1, j-1] = True
                
                # Count branch points
                num_branches = np.sum(branch_points)
                
                features['vessel_branch_count'] = float(num_branches)
                
                if np.sum(vessel_skeleton) > 0:
                    features['vessel_branch_density'] = float(num_branches / np.sum(vessel_skeleton))
                else:
                    features['vessel_branch_density'] = 0.0
            else:
                features['vessel_branch_count'] = 0.0
                features['vessel_branch_density'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting arborizing vessels: {str(e)}")
            return {
                'vessel_proportion': 0.0,
                'vessel_branch_count': 0.0,
                'vessel_branch_density': 0.0
            }
    
    def detect_milia_like_cysts(self, image, mask):
        """
        Detect milia-like cysts which are characteristic of SK.
        
        Method: Use blob detection to find small, bright, rounded structures.
        """
        try:
            features = {}
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            # Ensure values are in [0, 1] range
            if np.max(gray) > 1.0:
                gray = gray / 255.0
                
            # Create masked version
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # Enhance contrast for better cyst detection
            masked_gray_enhanced = (masked_gray * 255).astype(np.uint8)
            masked_gray_enhanced = cv2.equalizeHist(masked_gray_enhanced)
            masked_gray_enhanced = masked_gray_enhanced.astype(float) / 255.0
            
            # Detect blobs (milia-like cysts appear as small, bright blobs)
            # Convert to uint8 for OpenCV operations
            gray_uint8 = (masked_gray_enhanced * 255).astype(np.uint8)
            
            # Set up blob detector parameters
            params = cv2.SimpleBlobDetector_Params()
            
            # Filter by color (milia-like cysts are bright)
            params.filterByColor = True
            params.blobColor = 255
            
            # Filter by area
            params.filterByArea = True
            params.minArea = 5
            params.maxArea = 100
            
            # Filter by circularity (milia-like cysts are circular)
            params.filterByCircularity = True
            params.minCircularity = 0.7
            
            # Filter by convexity (milia-like cysts are convex)
            params.filterByConvexity = True
            params.minConvexity = 0.8
            
            # Filter by inertia (milia-like cysts are round)
            params.filterByInertia = True
            params.minInertiaRatio = 0.6
            
            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)
            
            # Detect blobs
            keypoints = detector.detect(gray_uint8)
            
            # Count cysts and calculate features
            features['milia_cyst_count'] = float(len(keypoints))
            
            if np.sum(mask) > 0:
                area = np.sum(mask)
                features['milia_cyst_density'] = float(len(keypoints) / (area / 1000))  # per 1000 pixels
            else:
                features['milia_cyst_density'] = 0.0
                
            # Calculate average size of cysts if any are detected
            if len(keypoints) > 0:
                avg_size = np.mean([k.size for k in keypoints])
                features['milia_cyst_avg_size'] = float(avg_size)
            else:
                features['milia_cyst_avg_size'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting milia-like cysts: {str(e)}")
            return {
                'milia_cyst_count': 0.0,
                'milia_cyst_density': 0.0,
                'milia_cyst_avg_size': 0.0
            }
    
    def detect_leaf_like_structures(self, image, mask):
        """
        Detect leaf-like structures which are characteristic of BCC.
        
        Method: Edge detection, followed by shape analysis for elongated structures 
        with characteristic leaf-like patterns.
        """
        try:
            features = {}
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            # Ensure values are in [0, 1] range
            if np.max(gray) > 1.0:
                gray = gray / 255.0
                
            # Create masked version
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # Apply edge detection
            edges = feature.canny(masked_gray, sigma=1.5, low_threshold=0.05, high_threshold=0.2)
            
            # Dilate edges to connect nearby structures
            edges_dilated = morphology.binary_dilation(edges, morphology.disk(1))
            
            # Find contours
            contours = measure.find_contours(edges_dilated, 0.5)
            
            # Filter contours to find elongated structures (potential leaf-like structures)
            leaf_like_count = 0
            elongation_values = []
            
            for contour in contours:
                if len(contour) >= 10:  # Minimum length to be considered
                    # Calculate contour properties
                    x, y = contour[:, 1], contour[:, 0]
                    
                    # Bounding rectangle
                    x_min, x_max = np.min(x), np.max(x)
                    y_min, y_max = np.min(y), np.max(y)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width > 0 and height > 0:
                        # Calculate elongation ratio
                        elongation = max(width, height) / min(width, height)
                        
                        # Leaf-like structures are typically elongated
                        if elongation > 2.0:
                            elongation_values.append(elongation)
                            leaf_like_count += 1
            
            features['leaf_like_structure_count'] = float(leaf_like_count)
            
            if np.sum(mask) > 0:
                area = np.sum(mask)
                features['leaf_like_structure_density'] = float(leaf_like_count / (area / 1000))  # per 1000 pixels
            else:
                features['leaf_like_structure_density'] = 0.0
                
            # Calculate average elongation if any leaf-like structures were found
            if elongation_values:
                features['leaf_like_structure_avg_elongation'] = float(np.mean(elongation_values))
            else:
                features['leaf_like_structure_avg_elongation'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting leaf-like structures: {str(e)}")
            return {
                'leaf_like_structure_count': 0.0,
                'leaf_like_structure_density': 0.0,
                'leaf_like_structure_avg_elongation': 0.0
            }
    
    def detect_brain_like_appearance(self, image, mask):
        """
        Detect brain-like appearance which is typical of SK.
        
        Method: Analyze texture patterns for the characteristic fissured, "brain-like" appearance
        using lacunarity and multiscale entropy.
        """
        try:
            features = {}
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            # Ensure values are in [0, 1] range
            if np.max(gray) > 1.0:
                gray = gray / 255.0
                
            # Create masked version
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # Apply Gaussian blur to reduce noise
            masked_gray_smooth = ndimage.gaussian_filter(masked_gray, sigma=1.0)
            
            # Apply texture filtering to detect fissures
            # Brain-like appearance has characteristic fissured texture
            entropy_img = feature.multiscale_basic_features(
                masked_gray_smooth, 
                intensity=False, 
                edges=False, 
                texture=True,
                sigma_min=1,
                sigma_max=3
            )
            
            # Extract the texture entropy channel
            if entropy_img.shape[2] >= 3:
                texture_entropy = entropy_img[:,:,2]
            else:
                texture_entropy = entropy_img[:,:,0]
            
            # Threshold to find high entropy regions (fissures)
            threshold = np.percentile(texture_entropy[mask], 80)
            high_entropy_mask = (texture_entropy > threshold) & mask
            
            # Calculate proportion of high entropy regions
            if np.sum(mask) > 0:
                features['brain_like_fissure_proportion'] = float(np.sum(high_entropy_mask) / np.sum(mask))
            else:
                features['brain_like_fissure_proportion'] = 0.0
                
            # Calculate lacunarity - a measure of how patterns fill space, relates to "brain-like" appearance
            if np.sum(mask) > 0:
                # Approximate lacunarity using box counting at multiple scales
                lacunarity_values = []
                for box_size in [5, 10, 15]:
                    # Count boxes with high entropy regions
                    boxes = np.zeros((masked_gray.shape[0] // box_size, masked_gray.shape[1] // box_size))
                    
                    for i in range(boxes.shape[0]):
                        for j in range(boxes.shape[1]):
                            # Box region
                            box_region = high_entropy_mask[
                                i*box_size:min((i+1)*box_size, high_entropy_mask.shape[0]),
                                j*box_size:min((j+1)*box_size, high_entropy_mask.shape[1])
                            ]
                            
                            if np.any(box_region):
                                boxes[i, j] = np.mean(box_region)
                    
                    # Calculate second moment for non-zero boxes
                    non_zero_boxes = boxes[boxes > 0]
                    if len(non_zero_boxes) > 1:
                        mean_val = np.mean(non_zero_boxes)
                        second_moment = np.mean(non_zero_boxes**2)
                        
                        # Lacunarity is (second moment) / (mean)^2
                        if mean_val > 0:
                            lacunarity = second_moment / (mean_val**2)
                            lacunarity_values.append(lacunarity)
                
                # Average lacunarity across scales
                if lacunarity_values:
                    features['brain_like_lacunarity'] = float(np.mean(lacunarity_values))
                else:
                    features['brain_like_lacunarity'] = 1.0  # Default value indicating no pattern
            else:
                features['brain_like_lacunarity'] = 1.0
                
            # Brain-like patterns also have characteristic edges
            edges = feature.canny(masked_gray, sigma=1.0)
            
            # Calculate edge density
            if np.sum(mask) > 0:
                features['brain_like_edge_density'] = float(np.sum(edges & mask) / np.sum(mask))
            else:
                features['brain_like_edge_density'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting brain-like appearance: {str(e)}")
            return {
                'brain_like_fissure_proportion': 0.0,
                'brain_like_lacunarity': 1.0,
                'brain_like_edge_density': 0.0
            }