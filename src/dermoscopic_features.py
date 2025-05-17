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
            # L channel for lightness is also important as the veil is often lighter
            l_channel = lab_image[:,:,0]
            a_channel = lab_image[:,:,1]
            b_channel = lab_image[:,:,2]
            
            # Enhanced thresholds for more accurate blue-white veil detection
            # Refined after analysis of BCC samples
            blue_white_mask = (a_channel < -6) & (b_channel < -12) & (l_channel > 65) & mask
            
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
                    
                    # Add new feature: standard deviation of region sizes (helps detect irregular patterns)
                    features['blue_white_std_region_size'] = float(np.std(region_sizes) if len(region_sizes) > 0 else 0)
                    
                    # Add new feature: compactness of the largest region
                    if len(region_sizes) > 0:
                        largest_label = np.argmax(region_sizes) + 1
                        largest_region = labeled_mask == largest_label
                        perimeter = measure.perimeter(largest_region)
                        area = np.sum(largest_region)
                        if area > 0 and perimeter > 0:
                            compactness = (perimeter ** 2) / (4 * np.pi * area)
                            features['blue_white_largest_compactness'] = float(compactness)
                        else:
                            features['blue_white_largest_compactness'] = 1.0
                    else:
                        features['blue_white_largest_compactness'] = 1.0
                else:
                    features['blue_white_region_count'] = 0.0
                    features['blue_white_max_region_size'] = 0.0
                    features['blue_white_mean_region_size'] = 0.0
                    features['blue_white_std_region_size'] = 0.0
                    features['blue_white_largest_compactness'] = 1.0
            else:
                features['blue_white_region_count'] = 0.0
                features['blue_white_max_region_size'] = 0.0
                features['blue_white_mean_region_size'] = 0.0
                features['blue_white_std_region_size'] = 0.0
                features['blue_white_largest_compactness'] = 1.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting blue-white veil: {str(e)}")
            return {
                'blue_white_veil_proportion': 0.0,
                'blue_white_region_count': 0.0,
                'blue_white_max_region_size': 0.0,
                'blue_white_mean_region_size': 0.0,
                'blue_white_std_region_size': 0.0,
                'blue_white_largest_compactness': 1.0
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
                # Calculate Gaussian derivatives using a different approach
                # First apply Gaussian smoothing
                smoothed = ndimage.gaussian_filter(masked_gray, sigma)
                
                # Then compute finite differences for derivatives
                # For x derivative (gradient in x direction)
                gx = np.zeros_like(smoothed)
                gx[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2
                
                # For y derivative (gradient in y direction)
                gy = np.zeros_like(smoothed)
                gy[1:-1, :] = (smoothed[2:, :] - smoothed[:-2, :]) / 2
                
                # Second derivatives
                # For xx derivative
                gxx = np.zeros_like(smoothed)
                gxx[:, 1:-1] = (smoothed[:, 2:] - 2*smoothed[:, 1:-1] + smoothed[:, :-2])
                
                # For yy derivative
                gyy = np.zeros_like(smoothed)
                gyy[1:-1, :] = (smoothed[2:, :] - 2*smoothed[1:-1, :] + smoothed[:-2, :])
                
                # For xy derivative (mixed)
                gxy = np.zeros_like(smoothed)
                gxy[1:-1, 1:-1] = (smoothed[2:, 2:] - smoothed[2:, :-2] - smoothed[:-2, 2:] + smoothed[:-2, :-2]) / 4
                
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
            
            # Use a simpler approach for blob detection to avoid OpenCV version issues
            # Detect bright spots using a combination of thresholding and connected component analysis
            
            # Threshold the image to identify bright spots (possible milia-like cysts)
            thresh_val = 200  # High threshold to detect only bright spots
            _, binary = cv2.threshold(gray_uint8, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up the binary image
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components (blobs)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # Filter blobs by size and shape (similar to what SimpleBlobDetector would do)
            keypoints = []
            
            # Skip the first component (background)
            for i in range(1, num_labels):
                # Get blob statistics
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate circularity approximation
                perimeter = 2 * (width + height)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Filter by size and shape
                if 5 <= area <= 100 and circularity > 0.6:
                    # Create a keypoint-like object (just need x, y coordinates and size)
                    x, y = centroids[i]
                    keypoints.append((x, y, np.sqrt(area)))
            
            # Now we have our filtered keypoints similar to what SimpleBlobDetector would give
            # We already have keypoints, so we don't need to call detector.detect()
            
            # Count cysts and calculate features
            features['milia_cyst_count'] = float(len(keypoints))
            
            if np.sum(mask) > 0:
                area = np.sum(mask)
                features['milia_cyst_density'] = float(len(keypoints) / (area / 1000))  # per 1000 pixels
            else:
                features['milia_cyst_density'] = 0.0
                
            # Calculate average size of cysts if any are detected
            if len(keypoints) > 0:
                # Extract size from our custom keypoints (third element in each tuple)
                sizes = [kp[2] for kp in keypoints]
                features['milia_cyst_avg_size'] = float(np.mean(sizes))
                
                # Add new feature: standard deviation of cyst sizes (helps distinguish SK patterns)
                features['milia_cyst_size_std'] = float(np.std(sizes)) if len(sizes) > 1 else 0.0
                
                # Add new feature: largest cyst size (SK often has larger cysts)
                features['milia_cyst_max_size'] = float(np.max(sizes))
                
                # Add new feature: spatial distribution of cysts (clustering metric)
                if len(keypoints) > 1:
                    # Get all keypoint centers
                    centers = np.array([(kp[0], kp[1]) for kp in keypoints])
                    
                    # Calculate pairwise distances between all cysts
                    from scipy.spatial.distance import pdist
                    distances = pdist(centers)
                    
                    # Mean distance between cysts - lower value indicates clustering
                    features['milia_cyst_mean_distance'] = float(np.mean(distances))
                    
                    # Coefficient of variation of distances - higher value indicates uneven distribution
                    features['milia_cyst_distance_cv'] = float(np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0.0
                else:
                    features['milia_cyst_mean_distance'] = 0.0
                    features['milia_cyst_distance_cv'] = 0.0
            else:
                features['milia_cyst_avg_size'] = 0.0
                features['milia_cyst_size_std'] = 0.0
                features['milia_cyst_max_size'] = 0.0
                features['milia_cyst_mean_distance'] = 0.0
                features['milia_cyst_distance_cv'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting milia-like cysts: {str(e)}")
            return {
                'milia_cyst_count': 0.0,
                'milia_cyst_density': 0.0,
                'milia_cyst_avg_size': 0.0,
                'milia_cyst_size_std': 0.0,
                'milia_cyst_max_size': 0.0,
                'milia_cyst_mean_distance': 0.0,
                'milia_cyst_distance_cv': 0.0
            }
    
    def detect_leaf_like_structures(self, image, mask):
        """
        Detect leaf-like structures which are characteristic of BCC.
        
        Enhanced method: Multi-scale edge detection, followed by detailed shape analysis for 
        elongated structures with characteristic leaf-like patterns.
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
            
            # Apply contrast enhancement for better edge detection
            p2, p98 = np.percentile(masked_gray[mask] if np.any(mask) else masked_gray, (2, 98))
            if p98 > p2:  # Avoid division by zero
                masked_gray = np.clip((masked_gray - p2) / (p98 - p2), 0, 1)
                masked_gray[~mask] = 0
            
            # Apply multi-scale edge detection (better for capturing different sized features)
            edges_combined = np.zeros_like(masked_gray, dtype=bool)
            
            for sigma in [1.0, 1.5, 2.0]:  # Multiple scales to capture different sized structures
                edges = feature.canny(
                    masked_gray, 
                    sigma=sigma, 
                    low_threshold=0.05, 
                    high_threshold=0.2
                )
                edges_combined = edges_combined | edges
            
            # Dilate edges to connect nearby structures
            edges_dilated = morphology.binary_dilation(edges_combined, morphology.disk(1))
            
            # Find contours
            contours = measure.find_contours(edges_dilated, 0.5)
            
            # Filter contours to find elongated structures (potential leaf-like structures)
            leaf_like_count = 0
            elongation_values = []
            curvature_values = []
            
            for contour in contours:
                if len(contour) >= 15:  # Increased minimum length for better quality
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
                        
                        # Calculate contour perimeter and area
                        perimeter = len(contour)
                        
                        # Convert to integer coordinates for mask creation
                        coords = np.vstack([y, x]).T.astype(np.int32)
                        coords = coords[coords[:, 0] >= 0]
                        coords = coords[coords[:, 1] >= 0]
                        coords = coords[coords[:, 0] < masked_gray.shape[0]]
                        coords = coords[coords[:, 1] < masked_gray.shape[1]]
                        
                        if len(coords) > 0:
                            # Create a mask for this contour
                            contour_mask = np.zeros_like(masked_gray, dtype=bool)
                            contour_mask[coords[:, 0], coords[:, 1]] = True
                            
                            # Calculate filled area
                            filled_contour = ndimage.binary_fill_holes(contour_mask)
                            area = np.sum(filled_contour)
                            
                            # Calculate solidity (area / convex hull area)
                            if area > 0:
                                # Leaf-like structures often have a specific shape
                                # Consider both elongation and "leafiness" (area to perimeter ratio)
                                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                                
                                # Calculate angles along the contour for curvature estimation
                                if len(contour) > 20:
                                    angles = []
                                    for i in range(1, len(contour) - 1):
                                        v1 = contour[i] - contour[i-1]
                                        v2 = contour[i+1] - contour[i]
                                        # Compute angle between vectors
                                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                                            v1 = v1 / np.linalg.norm(v1)
                                            v2 = v2 / np.linalg.norm(v2)
                                            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                                            angle = np.arccos(dot)
                                            angles.append(angle)
                                    
                                    if angles:
                                        curvature = np.std(angles)
                                        curvature_values.append(curvature)
                                
                                # Leaf-like structures criteria:
                                # 1. Elongated (high elongation ratio)
                                # 2. Not too circular (low-medium circularity)
                                # 3. Has a certain size
                                if (elongation > 2.5 and circularity < 0.7 and area > 30) or \
                                   (elongation > 2.0 and circularity < 0.5 and area > 50):
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
                features['leaf_like_structure_max_elongation'] = float(np.max(elongation_values))
            else:
                features['leaf_like_structure_avg_elongation'] = 0.0
                features['leaf_like_structure_max_elongation'] = 0.0
                
            # Add curvature features (important for leaf shape characterization)
            if curvature_values:
                features['leaf_like_structure_avg_curvature'] = float(np.mean(curvature_values))
                features['leaf_like_structure_max_curvature'] = float(np.max(curvature_values))
            else:
                features['leaf_like_structure_avg_curvature'] = 0.0
                features['leaf_like_structure_max_curvature'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting leaf-like structures: {str(e)}")
            return {
                'leaf_like_structure_count': 0.0,
                'leaf_like_structure_density': 0.0,
                'leaf_like_structure_avg_elongation': 0.0,
                'leaf_like_structure_max_elongation': 0.0,
                'leaf_like_structure_avg_curvature': 0.0,
                'leaf_like_structure_max_curvature': 0.0
            }
    
    def detect_brain_like_appearance(self, image, mask):
        """
        Detect brain-like appearance which is typical of SK.
        
        Enhanced method: Analyze texture patterns for the characteristic fissured, "brain-like" appearance
        using improved lacunarity metrics, multiscale entropy, and fissure pattern analysis.
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
            
            # Apply contrast enhancement to better visualize the "brain-like" structures
            # SK often shows fissured structures with better contrast
            if np.any(mask):
                p2, p98 = np.percentile(masked_gray[mask], (2, 98))
                if p98 > p2:  # Avoid division by zero
                    enhanced_gray = np.clip((masked_gray - p2) / (p98 - p2), 0, 1)
                    enhanced_gray[~mask] = 0
                else:
                    enhanced_gray = masked_gray
            else:
                enhanced_gray = masked_gray
            
            # Apply Gaussian blur to reduce noise but preserve fissure structures
            masked_gray_smooth = ndimage.gaussian_filter(enhanced_gray, sigma=0.8)  # Reduced sigma to preserve finer details
            
            # Apply texture filtering to detect fissures with improved sensitivity
            # Brain-like appearance has characteristic fissured texture
            entropy_img = feature.multiscale_basic_features(
                masked_gray_smooth, 
                intensity=False, 
                edges=False, 
                texture=True,
                sigma_min=1,
                sigma_max=4,  # Increased to capture more scales
                num_sigma=4   # More scales for better detail
            )
            
            # Extract the texture entropy channel
            if entropy_img.shape[2] >= 3:
                texture_entropy = entropy_img[:,:,2]
            else:
                texture_entropy = entropy_img[:,:,0]
            
            # Threshold to find high entropy regions (fissures) with adaptive threshold
            if np.any(mask):
                threshold = np.percentile(texture_entropy[mask], 75)  # Lower threshold to capture more fissures
                high_entropy_mask = (texture_entropy > threshold) & mask
            else:
                high_entropy_mask = np.zeros_like(texture_entropy, dtype=bool)
            
            # Calculate proportion of high entropy regions
            if np.sum(mask) > 0:
                features['brain_like_fissure_proportion'] = float(np.sum(high_entropy_mask) / np.sum(mask))
            else:
                features['brain_like_fissure_proportion'] = 0.0
            
            # Enhance fissure detection using edge processing
            # Detect edges at multiple scales to capture fine and coarse structures
            edges_combined = np.zeros_like(enhanced_gray, dtype=bool)
            
            for sigma in [0.8, 1.5, 2.5]:
                edges = feature.canny(
                    enhanced_gray, 
                    sigma=sigma, 
                    low_threshold=0.05, 
                    high_threshold=0.15  # Lower threshold to capture more fissures
                )
                edges_combined = edges_combined | edges
            
            # Skeletonize the edges to get centerlines of fissures
            skeletonized_edges = morphology.skeletonize(edges_combined)
            
            # Count branch points (junctions in the fissure network)
            # This is a key characteristic of "brain-like" appearance
            branch_points = 0
            if np.any(skeletonized_edges):
                skel_padded = np.pad(skeletonized_edges, pad_width=1, mode='constant', constant_values=0)
                
                for i in range(1, skel_padded.shape[0]-1):
                    for j in range(1, skel_padded.shape[1]-1):
                        if skel_padded[i, j]:
                            # Count neighbors
                            neighbors = np.sum(skel_padded[i-1:i+2, j-1:j+2]) - 1
                            if neighbors > 2:  # More than 2 neighbors = branch point
                                branch_points += 1
            
            # Calculate branch point density (highly indicative of brain-like patterns)
            if np.sum(mask) > 0:
                features['brain_like_branch_density'] = float(branch_points / np.sum(mask) * 1000)  # per 1000 pixels
            else:
                features['brain_like_branch_density'] = 0.0
                
            # Calculate lacunarity - a measure of how patterns fill space, relates to "brain-like" appearance
            if np.sum(mask) > 0:
                # Approximate lacunarity using box counting at multiple scales
                lacunarity_values = []
                for box_size in [5, 10, 15, 20]:  # Added larger scale for better pattern characterization
                    # Count boxes with high entropy regions
                    boxes = np.zeros((masked_gray.shape[0] // box_size, masked_gray.shape[1] // box_size))
                    
                    for i in range(boxes.shape[0]):
                        for j in range(boxes.shape[1]):
                            # Box region
                            i_end = min((i+1)*box_size, high_entropy_mask.shape[0])
                            j_end = min((j+1)*box_size, high_entropy_mask.shape[1])
                            
                            if i*box_size < i_end and j*box_size < j_end:  # Ensure valid indices
                                box_region = high_entropy_mask[i*box_size:i_end, j*box_size:j_end]
                                
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
                
                # Calculate lacunarity features across scales
                if lacunarity_values:
                    features['brain_like_lacunarity'] = float(np.mean(lacunarity_values))
                    # Add variance of lacunarity across scales (helps distinguish SK patterns)
                    features['brain_like_lacunarity_std'] = float(np.std(lacunarity_values)) if len(lacunarity_values) > 1 else 0.0
                else:
                    features['brain_like_lacunarity'] = 1.0  # Default value indicating no pattern
                    features['brain_like_lacunarity_std'] = 0.0
            else:
                features['brain_like_lacunarity'] = 1.0
                features['brain_like_lacunarity_std'] = 0.0
            
            # Calculate texture features using local binary patterns (LBP)
            # These are excellent for capturing fine texture differences between BCC and SK
            try:
                from skimage.feature import local_binary_pattern
                
                # Calculate LBP with parameters tuned for dermoscopic features
                radius = 2
                n_points = 8 * radius
                lbp = local_binary_pattern(masked_gray_smooth, n_points, radius, method='uniform')
                
                # Mask the LBP image
                lbp_masked = lbp.copy()
                lbp_masked[~mask] = 0
                
                # Calculate histogram of LBP values (only over the mask region)
                if np.sum(mask) > 0:
                    hist, _ = np.histogram(lbp_masked[mask], bins=n_points+2, range=(0, n_points+2), density=True)
                    
                    # Extract statistical features from the LBP histogram
                    features['brain_like_lbp_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
                    features['brain_like_lbp_uniformity'] = float(np.sum(hist**2))
                    features['brain_like_lbp_max'] = float(np.max(hist))
                else:
                    features['brain_like_lbp_entropy'] = 0.0
                    features['brain_like_lbp_uniformity'] = 0.0
                    features['brain_like_lbp_max'] = 0.0
            except Exception as tex_err:
                self.logger.warning(f"Could not calculate LBP features: {str(tex_err)}")
                features['brain_like_lbp_entropy'] = 0.0
                features['brain_like_lbp_uniformity'] = 0.0
                features['brain_like_lbp_max'] = 0.0
            
            # Calculate edge density and fissure characteristics
            if np.sum(mask) > 0:
                features['brain_like_edge_density'] = float(np.sum(edges_combined & mask) / np.sum(mask))
                
                # Calculate mean gradient magnitude within high entropy regions
                # (SK typically has stronger gradients along fissure edges)
                grad_x = ndimage.sobel(masked_gray_smooth, axis=1)
                grad_y = ndimage.sobel(masked_gray_smooth, axis=0)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                if np.sum(high_entropy_mask) > 0:
                    features['brain_like_gradient_mean'] = float(np.mean(grad_mag[high_entropy_mask]))
                else:
                    features['brain_like_gradient_mean'] = 0.0
            else:
                features['brain_like_edge_density'] = 0.0
                features['brain_like_gradient_mean'] = 0.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting brain-like appearance: {str(e)}")
            return {
                'brain_like_fissure_proportion': 0.0,
                'brain_like_lacunarity': 1.0,
                'brain_like_lacunarity_std': 0.0,
                'brain_like_branch_density': 0.0,
                'brain_like_edge_density': 0.0,
                'brain_like_gradient_mean': 0.0,
                'brain_like_lbp_entropy': 0.0,
                'brain_like_lbp_uniformity': 0.0,
                'brain_like_lbp_max': 0.0
            }