import numpy as np
import logging
from skimage import color, feature, measure
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import convex_hull_image
from scipy import ndimage
import pywt
import cv2

class ConventionalFeatureExtractor:
    def __init__(self):
        """Initialize the conventional feature extractor."""
        self.logger = logging.getLogger(__name__)
        
    def extract_all_features(self, image, mask=None):
        """Extract all conventional features from the image."""
        try:
            features = {}
            feature_counts = {
                'geometric': 0,
                'color': 0,
                'texture': 0
            }
            
            # If mask is not provided, use the entire image
            if mask is None:
                mask = np.ones(image.shape[:2], dtype=bool)
                
            # Extract geometric features
            geometric_features = self.extract_geometric_features(mask)
            features.update(geometric_features)
            feature_counts['geometric'] = len(geometric_features)
            
            # Extract color features from different color spaces
            color_features = self.extract_color_features(image, mask)
            features.update(color_features)
            feature_counts['color'] = len(color_features)
            
            # Extract texture features
            texture_features = self.extract_texture_features(image, mask)
            features.update(texture_features)
            feature_counts['texture'] = len(texture_features)
            
            # Log feature counts
            total_features = len(features)
            self.logger.info(f"Extracted {total_features} conventional features: " 
                            f"geometric={feature_counts['geometric']}, "
                            f"color={feature_counts['color']}, "
                            f"texture={feature_counts['texture']}")
            
            if total_features < 300:
                self.logger.warning(f"Conventional feature count ({total_features}) is less than expected (~394)")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting conventional features: {str(e)}")
            return {}
            
    def extract_geometric_features(self, mask):
        """Extract geometric features describing shape and border."""
        try:
            features = {}
            
            # Basic region properties
            regionprops = measure.regionprops(mask.astype(int))
            if not regionprops:
                return {}
                
            regionprops = regionprops[0]
            
            # Area and perimeter
            features['area'] = regionprops.area
            features['perimeter'] = regionprops.perimeter
            
            # Compactness (circularity)
            features['compactness'] = (4 * np.pi * regionprops.area) / (regionprops.perimeter ** 2) if regionprops.perimeter > 0 else 0
            
            # Asymmetry
            features['eccentricity'] = regionprops.eccentricity
            features['extent'] = regionprops.extent
            
            # Border irregularity
            # Calculate border irregularity using fractal dimension approach
            contours = measure.find_contours(mask, 0.5)
            if contours:
                boundary = contours[0]
                features['boundary_length'] = len(boundary)
            else:
                features['boundary_length'] = 0
            
            # Calculate convex hull and convexity
            hull = convex_hull_image(mask)
            hull_perimeter = measure.perimeter(hull)
            features['convexity'] = hull_perimeter / regionprops.perimeter if regionprops.perimeter > 0 else 1
            
            # Asymmetry measurement based on moments
            moments = measure.moments(mask)
            features['hu_moments'] = measure.moments_hu(moments).tolist()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting geometric features: {str(e)}")
            return {}
            
    def extract_color_features(self, image, mask):
        """Extract color features from multiple color spaces."""
        try:
            features = {}
            
            # Make sure image has at least 3 channels
            if len(image.shape) < 3:
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] < 3:
                image = np.stack([image[:,:,0]] * 3, axis=2)
                
            # Define color spaces to analyze
            # Original RGB
            rgb_image = image[:,:,:3]
            
            # Convert to other color spaces
            hsv_image = color.rgb2hsv(rgb_image)
            lab_image = color.rgb2lab(rgb_image)
            
            # For each color space, extract statistical features
            color_spaces = {
                'rgb': rgb_image,
                'hsv': hsv_image,
                'lab': lab_image
            }
            
            for space_name, space_image in color_spaces.items():
                for channel in range(space_image.shape[2]):
                    channel_data = space_image[:,:,channel][mask]
                    if len(channel_data) == 0:
                        continue
                        
                    prefix = f"{space_name}_{channel}"
                    
                    # Basic statistics
                    features[f"{prefix}_mean"] = float(np.mean(channel_data))
                    features[f"{prefix}_std"] = float(np.std(channel_data))
                    features[f"{prefix}_min"] = float(np.min(channel_data))
                    features[f"{prefix}_max"] = float(np.max(channel_data))
                    
                    # Higher order statistics
                    features[f"{prefix}_skewness"] = float(ndimage.mean(
                        (channel_data - np.mean(channel_data))**3
                    ) / (np.std(channel_data)**3) if np.std(channel_data) > 0 else 0)
                    
                    features[f"{prefix}_kurtosis"] = float(ndimage.mean(
                        (channel_data - np.mean(channel_data))**4
                    ) / (np.std(channel_data)**4) if np.std(channel_data) > 0 else 0)
                    
                    # Entropy
                    if np.max(channel_data) > np.min(channel_data):
                        hist, _ = np.histogram(channel_data, bins=256, density=True)
                        hist = hist[hist > 0]  # Remove zeros
                        features[f"{prefix}_entropy"] = float(-np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0)
                    else:
                        features[f"{prefix}_entropy"] = 0.0
            
            # Color variation features
            for space_name, space_image in color_spaces.items():
                # Color variance within the lesion
                if np.sum(mask) > 0:
                    masked_colors = space_image[mask]
                    color_var = np.sum(np.var(masked_colors, axis=0))
                    features[f"{space_name}_color_variance"] = float(color_var)
                else:
                    features[f"{space_name}_color_variance"] = 0.0
                
                # Color distribution histogram features
                for channel in range(space_image.shape[2]):
                    channel_data = space_image[:,:,channel][mask]
                    if len(channel_data) > 0 and np.max(channel_data) > np.min(channel_data):
                        hist, _ = np.histogram(channel_data, bins=8, density=True)
                        for i, count in enumerate(hist):
                            features[f"{space_name}_{channel}_hist_{i}"] = float(count)
                    else:
                        for i in range(8):
                            features[f"{space_name}_{channel}_hist_{i}"] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting color features: {str(e)}")
            return {}
            
    def extract_texture_features(self, image, mask):
        """Extract texture features using GLCM, LBP, and wavelets."""
        try:
            features = {}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            # Ensure values are in [0, 1] range
            if np.max(gray) > 1.0:
                gray = gray / 255.0
                
            # Create masked version for analysis
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # Quantize to 8 levels for GLCM (prevent memory issues)
            gray_quantized = np.round(gray * 7).astype(np.uint8)
            
            # Keep track of feature counts for logging
            glcm_count = 0
            lbp_count = 0
            wavelet_count = 0
            other_texture_count = 0
            
            # GLCM features - Expanded with more distances, angles, and properties
            distances = [1, 2, 3, 4]  # Added more distances
            angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]  # More angles
            
            for d in distances:
                for a_idx, angle in enumerate(angles):
                    try:
                        # Compute GLCM
                        glcm = graycomatrix(gray_quantized, 
                                           distances=[d], 
                                           angles=[angle], 
                                           levels=8,
                                           symmetric=True, 
                                           normed=True)
                        
                        # Compute GLCM properties - Added more properties
                        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 
                                'correlation', 'ASM', 'contrast', 'entropy']
                        
                        for prop in props:
                            try:
                                value = graycoprops(glcm, prop)[0, 0]
                                features[f'glcm_{prop}_d{d}_a{a_idx}'] = float(value)
                                glcm_count += 1
                            except:
                                # Entropy is not a built-in property, calculate it manually
                                if prop == 'entropy':
                                    glcm_flat = glcm[:,:,0,0].flatten()
                                    glcm_flat = glcm_flat[glcm_flat > 0]  # Remove zeros
                                    entropy = -np.sum(glcm_flat * np.log2(glcm_flat)) if len(glcm_flat) > 0 else 0
                                    features[f'glcm_entropy_d{d}_a{a_idx}'] = float(entropy)
                                    glcm_count += 1
                    except Exception as e:
                        # If GLCM fails, set default values
                        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 
                                'correlation', 'ASM', 'contrast', 'entropy']
                        for prop in props:
                            features[f'glcm_{prop}_d{d}_a{a_idx}'] = 0.0
                            glcm_count += 1
            
            self.logger.debug(f"Extracted {glcm_count} GLCM features")
            
            # Local Binary Patterns - Enhanced with multiple radii and methods
            try:
                # Multiple radii for multi-scale analysis
                for radius in [1, 2, 3]:
                    n_points = 8 * radius
                    # Convert to uint8 to avoid warning with local_binary_pattern
                    gray_uint8 = (gray * 255).astype(np.uint8)
                    
                    # Different LBP methods for diverse feature sets
                    for method in ['uniform', 'default', 'ror']:
                        method_suffix = f"r{radius}_{method}"
                        lbp = feature.local_binary_pattern(gray_uint8, n_points, radius, method=method)
                        lbp_masked = lbp[mask]
                        
                        if len(lbp_masked) > 0:
                            # LBP histogram with appropriate bin count for the method
                            if method == 'uniform':
                                num_bins = n_points + 2
                            elif method == 'default':
                                num_bins = 2**n_points
                                if num_bins > 128:  # Limit bin count to prevent memory issues
                                    num_bins = 128
                            else:  # 'ror' method
                                num_bins = n_points + 1
                                
                            # Create histogram and store as features
                            hist, _ = np.histogram(lbp_masked, bins=num_bins, density=True)
                            for i, val in enumerate(hist[:min(30, len(hist))]):  # Limit to first 30 bins max
                                features[f'lbp_hist_{method_suffix}_{i}'] = float(val)
                                lbp_count += 1
                            
                            # Extract additional statistics from LBP
                            features[f'lbp_mean_{method_suffix}'] = float(np.mean(lbp_masked))
                            features[f'lbp_std_{method_suffix}'] = float(np.std(lbp_masked))
                            features[f'lbp_entropy_{method_suffix}'] = float(
                                -np.sum((hist * np.log2(hist + 1e-10))) if len(hist) > 0 else 0)
                            lbp_count += 3
                        else:
                            # Default values if mask is empty
                            num_bins = 30  # Consistent limit
                            for i in range(num_bins):
                                features[f'lbp_hist_{method_suffix}_{i}'] = 0.0
                                lbp_count += 1
                                
                            features[f'lbp_mean_{method_suffix}'] = 0.0
                            features[f'lbp_std_{method_suffix}'] = 0.0
                            features[f'lbp_entropy_{method_suffix}'] = 0.0
                            lbp_count += 3
                            
                self.logger.debug(f"Extracted {lbp_count} LBP features")
            except Exception as e:
                self.logger.warning(f"LBP computation failed: {str(e)}")
                # Set default values for all LBP features
                for radius in [1, 2, 3]:
                    for method in ['uniform', 'default', 'ror']:
                        method_suffix = f"r{radius}_{method}"
                        for i in range(30):  # Consistent limit
                            features[f'lbp_hist_{method_suffix}_{i}'] = 0.0
                            lbp_count += 1
                        
                        features[f'lbp_mean_{method_suffix}'] = 0.0
                        features[f'lbp_std_{method_suffix}'] = 0.0
                        features[f'lbp_entropy_{method_suffix}'] = 0.0
                        lbp_count += 3
                        
                self.logger.debug(f"Added {lbp_count} default LBP features")
            
            # Wavelet features - Enhanced with multiple wavelet families and more statistics
            try:
                # Use multiple wavelet families for richer feature extraction
                wavelet_families = ['haar', 'db1', 'db2', 'sym2', 'coif1']
                wavelet_levels = 3  # More decomposition levels
                
                for family in wavelet_families:
                    try:
                        # Apply 2D wavelet transform with higher level
                        coeffs = pywt.wavedec2(masked_gray, family, level=wavelet_levels)
                        
                        # Extract statistical features from each coefficient matrix
                        for level, coeff_data in enumerate(coeffs):
                            if level == 0:  # Approximation
                                cA = coeff_data
                                # More statistical features for approximation coefficients
                                stats = {
                                    'mean': float(np.mean(np.abs(cA))),
                                    'std': float(np.std(cA)),
                                    'energy': float(np.sum(cA**2)),
                                    'entropy': float(-np.sum(np.abs(cA)**2 * np.log2(np.abs(cA)**2 + 1e-10))),
                                    'kurtosis': float(np.mean((cA-np.mean(cA))**4) / (np.std(cA)**4 + 1e-10)),
                                    'skewness': float(np.mean((cA-np.mean(cA))**3) / (np.std(cA)**3 + 1e-10))
                                }
                                
                                for stat_name, stat_value in stats.items():
                                    features[f'wavelet_{family}_approx_{stat_name}'] = stat_value
                                    wavelet_count += 1
                                    
                            else:  # Details
                                for i, c_type in enumerate(['horizontal', 'vertical', 'diagonal']):
                                    c = coeff_data[i]
                                    # More statistical features for detail coefficients
                                    stats = {
                                        'mean': float(np.mean(np.abs(c))),
                                        'std': float(np.std(c)),
                                        'energy': float(np.sum(c**2)),
                                        'entropy': float(-np.sum(np.abs(c)**2 * np.log2(np.abs(c)**2 + 1e-10))),
                                        'max': float(np.max(np.abs(c)))
                                    }
                                    
                                    for stat_name, stat_value in stats.items():
                                        features[f'wavelet_{family}_{c_type}_{level}_{stat_name}'] = stat_value
                                        wavelet_count += 1
                    except:
                        # Skip this wavelet family if it fails
                        self.logger.warning(f"Wavelet family {family} failed, skipping")
                        continue
                
                self.logger.debug(f"Extracted {wavelet_count} wavelet features")
                
            except Exception as e:
                self.logger.warning(f"Wavelet computation failed: {str(e)}")
                # Default wavelet features if transform fails
                for family in ['haar', 'db1', 'db2', 'sym2', 'coif1']:
                    # Approximation coefficients
                    for stat in ['mean', 'std', 'energy', 'entropy', 'kurtosis', 'skewness']:
                        features[f'wavelet_{family}_approx_{stat}'] = 0.0
                        wavelet_count += 1
                    
                    # Detail coefficients
                    for level in range(1, 4):  # 3 levels
                        for c_type in ['horizontal', 'vertical', 'diagonal']:
                            for stat in ['mean', 'std', 'energy', 'entropy', 'max']:
                                features[f'wavelet_{family}_{c_type}_{level}_{stat}'] = 0.0
                                wavelet_count += 1
                
                self.logger.debug(f"Added {wavelet_count} default wavelet features")
            
            # Advanced gradient and edge features with additional methods
            try:
                # Compute gradient with multiple kernel sizes
                for ksize in [3, 5, 7]:
                    # Sobel gradients
                    sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=ksize)
                    sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=ksize)
                    
                    magnitude = np.sqrt(sobelx**2 + sobely**2)
                    direction = np.arctan2(sobely, sobelx)
                    
                    # Extract statistical features from gradient information
                    features[f'gradient_mag_k{ksize}_mean'] = float(np.mean(magnitude[mask]))
                    features[f'gradient_mag_k{ksize}_std'] = float(np.std(magnitude[mask]))
                    features[f'gradient_mag_k{ksize}_max'] = float(np.max(magnitude[mask]))
                    features[f'gradient_mag_k{ksize}_energy'] = float(np.sum(magnitude[mask]**2))
                    
                    features[f'gradient_dir_k{ksize}_mean'] = float(np.mean(direction[mask]))
                    features[f'gradient_dir_k{ksize}_std'] = float(np.std(direction[mask]))
                    
                    # Gradient histogram with more bins for finer detail
                    hist, _ = np.histogram(direction[mask], bins=16, range=(-np.pi, np.pi), density=True)
                    for i, val in enumerate(hist):
                        features[f'gradient_dir_k{ksize}_hist_{i}'] = float(val)
                    
                    other_texture_count += 22  # Count the features added in this loop
                
                # Add edge detection features using different methods
                # Canny edges
                gray_uint8 = (masked_gray * 255).astype(np.uint8)
                for thresh in [50, 100, 150]:
                    edges = cv2.Canny(gray_uint8, thresh/2, thresh)
                    edge_ratio = np.sum(edges > 0) / np.sum(mask)
                    features[f'canny_edge_ratio_t{thresh}'] = float(edge_ratio)
                    other_texture_count += 1
                
                # Laplacian edge detector
                laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
                features['laplacian_mean'] = float(np.mean(np.abs(laplacian[mask])))
                features['laplacian_std'] = float(np.std(laplacian[mask]))
                features['laplacian_max'] = float(np.max(np.abs(laplacian[mask])))
                other_texture_count += 3
                
                # Scharr filters for edge detection (more accurate than Sobel)
                scharrx = cv2.Scharr(masked_gray, cv2.CV_64F, 1, 0)
                scharry = cv2.Scharr(masked_gray, cv2.CV_64F, 0, 1)
                scharr_mag = np.sqrt(scharrx**2 + scharry**2)
                features['scharr_mean'] = float(np.mean(scharr_mag[mask]))
                features['scharr_std'] = float(np.std(scharr_mag[mask]))
                features['scharr_energy'] = float(np.sum(scharr_mag[mask]**2))
                other_texture_count += 3
                
                self.logger.debug(f"Extracted {other_texture_count} gradient and edge features")
                
            except Exception as e:
                self.logger.warning(f"Gradient computation failed: {str(e)}")
                
                # Default gradient features for all kernel sizes
                for ksize in [3, 5, 7]:
                    features[f'gradient_mag_k{ksize}_mean'] = 0.0
                    features[f'gradient_mag_k{ksize}_std'] = 0.0
                    features[f'gradient_mag_k{ksize}_max'] = 0.0
                    features[f'gradient_mag_k{ksize}_energy'] = 0.0
                    
                    features[f'gradient_dir_k{ksize}_mean'] = 0.0
                    features[f'gradient_dir_k{ksize}_std'] = 0.0
                    
                    for i in range(16):
                        features[f'gradient_dir_k{ksize}_hist_{i}'] = 0.0
                
                # Default edge detection features
                for thresh in [50, 100, 150]:
                    features[f'canny_edge_ratio_t{thresh}'] = 0.0
                
                features['laplacian_mean'] = 0.0
                features['laplacian_std'] = 0.0
                features['laplacian_max'] = 0.0
                
                features['scharr_mean'] = 0.0
                features['scharr_std'] = 0.0
                features['scharr_energy'] = 0.0
                
                other_texture_count = 29 + 3*22  # Total count of default features
                self.logger.debug(f"Added {other_texture_count} default gradient and edge features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {str(e)}")
            return {}