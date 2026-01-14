import numpy as np
import logging
from skimage import color, feature, measure, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import convex_hull_image, disk
from skimage.segmentation import find_boundaries
from scipy import ndimage, stats
from scipy.special import comb
import pywt
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os

class ConventionalFeatureExtractor:
    def __init__(self):
        """Initialize the conventional feature extractor."""
        self.logger = logging.getLogger(__name__)
        
    def extract_all_features(self, image, mask=None):
        """Extract all conventional features from the image."""
        try:
            features = {}
            
            # If mask is not provided, use the entire image
            if mask is None:
                mask = np.ones(image.shape[:2], dtype=bool)
                
            # Extract geometric features
            geometric_features = self.extract_geometric_features(mask)
            features.update(geometric_features)
            
            # Extract color features from different color spaces
            color_features = self.extract_color_features(image, mask)
            features.update(color_features)
            
            # Extract texture features
            texture_features = self.extract_texture_features(image, mask)
            features.update(texture_features)
            
            # Extract Krawtchouk moments for shape and texture characterization
            krawtchouk_features = self.extract_krawtchouk_moments(image, mask)
            features.update(krawtchouk_features)

            abcde_features = self.extract_abcde_features(image, mask)
            features.update(abcde_features)
            
            # Enhanced color features
            enhanced_color_features = self.extract_enhanced_color_features(image, mask)
            features.update(enhanced_color_features)
            
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
            
            # Enhanced border analysis
            try:
                from skimage.segmentation import find_boundaries
                from scipy.spatial.distance import pdist
                
                # Find border pixels
                border = find_boundaries(mask, mode='inner')
                border_coords = np.column_stack(np.where(border))
                
                if len(border_coords) > 10:  # Need sufficient border points
                    # Multi-scale border irregularity
                    distances = pdist(border_coords)
                    features['border_distance_mean'] = float(np.mean(distances))
                    features['border_distance_std'] = float(np.std(distances))
                    features['border_distance_range'] = float(np.max(distances) - np.min(distances))
                    
                    # Local border variation (measures smoothness)
                    if len(border_coords) > 3:
                        # Calculate curvature along border
                        border_smooth = measure.approximate_polygon(border_coords, tolerance=2)
                        if len(border_smooth) > 3:
                            features['border_smoothness'] = float(len(border_smooth) / len(border_coords))
                        else:
                            features['border_smoothness'] = 1.0
                    else:
                        features['border_smoothness'] = 1.0
                else:
                    # Default values for insufficient border points
                    features['border_distance_mean'] = 0.0
                    features['border_distance_std'] = 0.0
                    features['border_distance_range'] = 0.0
                    features['border_smoothness'] = 1.0
                    
            except Exception as e:
                self.logger.warning(f"Error in enhanced border analysis: {str(e)}")
                features['border_distance_mean'] = 0.0
                features['border_distance_std'] = 0.0
                features['border_distance_range'] = 0.0
                features['border_smoothness'] = 1.0
            
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
        """Extract texture features using GLCM, LBP, wavelets, and Gabor filters."""
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
            
            # Contrast enhancement to improve feature extraction
            masked_gray_enhanced = filters.rank.enhance_contrast(
                (masked_gray * 255).astype(np.uint8), 
                footprint=np.ones((3, 3)),
                mask=mask.astype(np.uint8)
            ).astype(float) / 255.0
            
            # Quantize to 8 levels for GLCM (prevent memory issues)
            gray_quantized = np.round(gray * 7).astype(np.uint8)
            
            # GLCM features
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
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
                        
                        # Compute GLCM properties
                        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                        for prop in props:
                            value = graycoprops(glcm, prop)[0, 0]
                            features[f'glcm_{prop}_d{d}_a{a_idx}'] = float(value)
                    except:
                        # If GLCM fails, set default values
                        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                        for prop in props:
                            features[f'glcm_{prop}_d{d}_a{a_idx}'] = 0.0
            
            # Local Binary Patterns
            try:
                radius = 3
                n_points = 8 * radius
                # Convert to uint8 to avoid warning with local_binary_pattern
                gray_uint8 = (gray * 255).astype(np.uint8)
                lbp = feature.local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
                lbp_masked = lbp[mask]
                
                if len(lbp_masked) > 0:
                    # LBP histogram
                    hist, _ = np.histogram(lbp_masked, bins=n_points+2, density=True)
                    for i, val in enumerate(hist):
                        features[f'lbp_hist_{i}'] = float(val)
                else:
                    for i in range(n_points+2):
                        features[f'lbp_hist_{i}'] = 0.0
            except:
                for i in range(n_points+2 if 'n_points' in locals() else 26):
                    features[f'lbp_hist_{i}'] = 0.0
            
            # Wavelet features
            try:
                # Apply 2D wavelet transform
                coeffs = pywt.wavedec2(masked_gray, 'db1', level=2)
                
                # Extract statistical features from each coefficient matrix
                for level, coeff_data in enumerate(coeffs):
                    if level == 0:  # Approximation
                        cA = coeff_data
                        features[f'wavelet_approx_mean'] = float(np.mean(np.abs(cA)))
                        features[f'wavelet_approx_std'] = float(np.std(cA))
                        features[f'wavelet_approx_energy'] = float(np.sum(cA**2))
                    else:  # Details
                        for i, c_type in enumerate(['horizontal', 'vertical', 'diagonal']):
                            c = coeff_data[i]
                            features[f'wavelet_{c_type}_{level}_mean'] = float(np.mean(np.abs(c)))
                            features[f'wavelet_{c_type}_{level}_std'] = float(np.std(c))
                            features[f'wavelet_{c_type}_{level}_energy'] = float(np.sum(c**2))
            except:
                # Default wavelet features if transform fails
                prefixes = ['wavelet_approx'] + [f'wavelet_{d}_{l}' for l in [1, 2] for d in ['horizontal', 'vertical', 'diagonal']]
                for prefix in prefixes:
                    for stat in ['mean', 'std', 'energy']:
                        features[f'{prefix}_{stat}'] = 0.0
            
            # Gradient features
            try:
                # Compute gradient magnitude and direction
                sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
                
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                direction = np.arctan2(sobely, sobelx)
                
                # Extract statistical features from gradient information
                features['gradient_mag_mean'] = float(np.mean(magnitude[mask]))
                features['gradient_mag_std'] = float(np.std(magnitude[mask]))
                features['gradient_dir_mean'] = float(np.mean(direction[mask]))
                features['gradient_dir_std'] = float(np.std(direction[mask]))
                
                # Gradient histogram (binned by direction)
                hist, _ = np.histogram(direction[mask], bins=8, range=(-np.pi, np.pi), density=True)
                for i, val in enumerate(hist):
                    features[f'gradient_dir_hist_{i}'] = float(val)
            except:
                features['gradient_mag_mean'] = 0.0
                features['gradient_mag_std'] = 0.0
                features['gradient_dir_mean'] = 0.0
                features['gradient_dir_std'] = 0.0
                for i in range(8):
                    features[f'gradient_dir_hist_{i}'] = 0.0
            
            # NEW: Multi-scale Gabor filter features - highly effective for skin lesion texture
            try:
                # Define Gabor filter parameters optimized for BCC vs SK detection
                # These parameters were selected based on empirical studies in dermatology
                theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations
                sigma_values = [3, 5]  # Standard deviations
                frequency_values = [0.1, 0.2, 0.3]  # Frequencies
                
                # Compute Gabor features for each combination of parameters
                gabor_features = {}
                for theta in theta_values:
                    for sigma in sigma_values:
                        for frequency in frequency_values:
                            # Generate Gabor filter and apply it
                            gabor_kernel = cv2.getGaborKernel(
                                ksize=(21, 21),
                                sigma=sigma,
                                theta=theta,
                                lambd=1/frequency,
                                gamma=0.5,
                                psi=0
                            )
                            
                            # Apply filter
                            filtered_img = cv2.filter2D(masked_gray_enhanced, cv2.CV_64F, gabor_kernel)
                            filtered_masked = filtered_img[mask]
                            
                            if len(filtered_masked) > 0:
                                # Calculate statistical features from filter response
                                prefix = f"gabor_t{int(theta*180/np.pi)}_s{sigma}_f{int(frequency*100)}"
                                gabor_features[f"{prefix}_mean"] = float(np.mean(np.abs(filtered_masked)))
                                gabor_features[f"{prefix}_std"] = float(np.std(filtered_masked))
                                gabor_features[f"{prefix}_energy"] = float(np.sum(filtered_masked**2))
                                gabor_features[f"{prefix}_entropy"] = float(stats.entropy(
                                    np.histogram(filtered_masked, bins=10)[0] + 1e-10
                                ))
                                
                                # Feature capturing the presence of specific patterns (e.g., blood vessels in BCC)
                                # High values (99th percentile) indicate strong presence of the pattern
                                if len(filtered_masked) >= 100:  # Need enough pixels for percentile
                                    gabor_features[f"{prefix}_p99"] = float(np.percentile(filtered_masked, 99))
                                else:
                                    gabor_features[f"{prefix}_p99"] = float(np.max(filtered_masked) if len(filtered_masked) > 0 else 0)
                            else:
                                # Default values if no masked regions
                                prefix = f"gabor_t{int(theta*180/np.pi)}_s{sigma}_f{int(frequency*100)}"
                                gabor_features[f"{prefix}_mean"] = 0.0
                                gabor_features[f"{prefix}_std"] = 0.0
                                gabor_features[f"{prefix}_energy"] = 0.0
                                gabor_features[f"{prefix}_entropy"] = 0.0
                                gabor_features[f"{prefix}_p99"] = 0.0
                
                # Add Gabor features to main feature set
                features.update(gabor_features)
                
            except Exception as e:
                self.logger.warning(f"Error computing Gabor features: {str(e)}")
                # Create default Gabor features on error
                for theta in [0, 45, 90, 135]:
                    for sigma in [3, 5]:
                        for frequency in [10, 20, 30]:
                            prefix = f"gabor_t{theta}_s{sigma}_f{frequency}"
                            features[f"{prefix}_mean"] = 0.0
                            features[f"{prefix}_std"] = 0.0
                            features[f"{prefix}_energy"] = 0.0
                            features[f"{prefix}_entropy"] = 0.0
                            features[f"{prefix}_p99"] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {str(e)}")
            return {}

    def _krawtchouk_polynomial(self, x, n, p, N):
        """
        Calculate Krawtchouk polynomial value.
        
        The weighted Krawtchouk polynomial is defined as:
        K_n(x; p, N) = sum_{k=0}^{n} a_k * C(x, k) * C(N-x, n-k)
        
        where a_k = (-1)^k * C(n, k) * p^(n-k) * (1-p)^k
        
        Args:
            x: Point at which to evaluate polynomial (0 <= x <= N)
            n: Order of the polynomial (0 <= n <= N)
            p: Parameter controlling shape (0 < p < 1)
            N: Size of the discrete interval
            
        Returns:
            float: Value of the Krawtchouk polynomial
        """
        if n > N or x > N or x < 0:
            return 0.0
        
        try:
            result = 0.0
            for k in range(n + 1):
                if x >= k and (N - x) >= (n - k):
                    # Calculate binomial coefficients
                    binom_x = comb(x, k, exact=True)
                    binom_N = comb(N - x, n - k, exact=True)
                    
                    # Calculate coefficient a_k
                    a_k = ((-1) ** k) * comb(n, k, exact=True) * (p ** (n - k)) * ((1 - p) ** k)
                    
                    result += a_k * binom_x * binom_N
            
            return result
        except:
            return 0.0

    def _krawtchouk_weight(self, x, p, N):
        """
        Calculate weighting function for Krawtchouk polynomials.
        
        w(x; p, N) = C(N, x) * p^x * (1-p)^(N-x)
        
        Args:
            x: Position
            p: Parameter
            N: Size of interval
            
        Returns:
            float: Weight value
        """
        try:
            if x < 0 or x > N:
                return 0.0
            weight = comb(N, x, exact=True) * (p ** x) * ((1 - p) ** (N - x))
            return weight
        except:
            return 0.0

    def extract_krawtchouk_moments(self, image, mask, max_order=4, p1=0.5, p2=0.5, normalize_size=64):
        """
        Extract Krawtchouk moments optimized for skin lesion texture and shape analysis.
        
        Krawtchouk moments are orthogonal discrete moments that provide excellent 
        discrimination for texture patterns in medical images. They are particularly 
        effective for:
        - Detecting subtle texture differences (BCC pearling vs SK keratin plugs)
        - Capturing border irregularities
        - Shape asymmetry analysis
        - Rotation and scale invariant feature extraction
        
        Args:
            image: Input image (RGB or grayscale)
            mask: Binary mask defining the lesion region
            max_order: Maximum moment order (default: 4)
                      - Higher orders capture finer details but may overfit
                      - Recommended: 3-5 for skin lesions
            p1: Parameter for x-direction polynomials (default: 0.5)
                - 0.5 provides symmetric weighting
            p2: Parameter for y-direction polynomials (default: 0.5)
                - 0.5 provides symmetric weighting
            normalize_size: Resize region to this size for computational efficiency (default: 64)
        
        Returns:
            dict: Dictionary of Krawtchouk moment features
        """
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
            
            # Extract lesion region using mask
            if np.sum(mask) == 0:
                # If mask is empty, return zero features
                self.logger.warning("Empty mask provided for Krawtchouk moments")
                for n in range(max_order + 1):
                    for m in range(max_order + 1):
                        if n + m <= max_order:
                            features[f'krawtchouk_moment_{n}_{m}'] = 0.0
                            features[f'krawtchouk_moment_abs_{n}_{m}'] = 0.0
                return features
            
            # Get bounding box of the lesion
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                self.logger.warning("Invalid mask for Krawtchouk moments")
                for n in range(max_order + 1):
                    for m in range(max_order + 1):
                        if n + m <= max_order:
                            features[f'krawtchouk_moment_{n}_{m}'] = 0.0
                            features[f'krawtchouk_moment_abs_{n}_{m}'] = 0.0
                return features
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Extract and crop the lesion region
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Apply mask to region
            lesion_region[~lesion_mask] = 0
            
            # Resize for computational efficiency while preserving aspect ratio
            original_shape = lesion_region.shape
            if max(original_shape) > normalize_size:
                scale = normalize_size / max(original_shape)
                new_height = int(original_shape[0] * scale)
                new_width = int(original_shape[1] * scale)
                lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                          interpolation=cv2.INTER_LINEAR)
                lesion_mask_resized = cv2.resize(lesion_mask.astype(np.uint8), 
                                                (new_width, new_height), 
                                                interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                lesion_mask_resized = lesion_mask
            
            N1, N2 = lesion_region.shape  # Image dimensions
            
            # Pre-compute Krawtchouk polynomials for efficiency
            K_x = np.zeros((max_order + 1, N2))  # Polynomials in x-direction
            K_y = np.zeros((max_order + 1, N1))  # Polynomials in y-direction
            
            for n in range(max_order + 1):
                for x in range(N2):
                    K_x[n, x] = self._krawtchouk_polynomial(x, n, p1, N2 - 1)
                for y in range(N1):
                    K_y[n, y] = self._krawtchouk_polynomial(y, n, p2, N1 - 1)
            
            # Compute Krawtchouk moments
            for n in range(max_order + 1):
                for m in range(max_order + 1):
                    if n + m > max_order:
                        continue
                    
                    # Calculate moment Q_nm
                    moment = 0.0
                    for y in range(N1):
                        for x in range(N2):
                            if lesion_mask_resized[y, x]:
                                moment += lesion_region[y, x] * K_x[n, x] * K_y[m, y]
                    
                    # Normalize by region area
                    moment = moment / np.sum(lesion_mask_resized) if np.sum(lesion_mask_resized) > 0 else 0.0
                    
                    # Store both raw and absolute moments
                    features[f'krawtchouk_moment_{n}_{m}'] = float(moment)
                    features[f'krawtchouk_moment_abs_{n}_{m}'] = float(np.abs(moment))
            
            # Compute normalized invariant moments (rotation invariant)
            # These are particularly useful for lesions that may appear at different orientations
            try:
                # Central moment (0,0) for normalization
                Q00 = features['krawtchouk_moment_0_0']
                
                if Q00 > 1e-10:  # Avoid division by zero
                    # Lower-order invariant moments
                    features['krawtchouk_invariant_1'] = float(
                        (features['krawtchouk_moment_abs_2_0'] + features['krawtchouk_moment_abs_0_2']) / (Q00 ** 2)
                    )
                    
                    features['krawtchouk_invariant_2'] = float(
                        ((features['krawtchouk_moment_2_0'] - features['krawtchouk_moment_0_2']) ** 2 + 
                         4 * features['krawtchouk_moment_1_1'] ** 2) / (Q00 ** 4)
                    )
                    
                    features['krawtchouk_invariant_3'] = float(
                        (features['krawtchouk_moment_abs_3_0'] + features['krawtchouk_moment_abs_1_2']) / (Q00 ** 2.5)
                    )
                    
                    features['krawtchouk_invariant_4'] = float(
                        (features['krawtchouk_moment_abs_0_3'] + features['krawtchouk_moment_abs_2_1']) / (Q00 ** 2.5)
                    )
                else:
                    features['krawtchouk_invariant_1'] = 0.0
                    features['krawtchouk_invariant_2'] = 0.0
                    features['krawtchouk_invariant_3'] = 0.0
                    features['krawtchouk_invariant_4'] = 0.0
            except Exception as e:
                self.logger.warning(f"Error computing Krawtchouk invariants: {str(e)}")
                features['krawtchouk_invariant_1'] = 0.0
                features['krawtchouk_invariant_2'] = 0.0
                features['krawtchouk_invariant_3'] = 0.0
                features['krawtchouk_invariant_4'] = 0.0
            
            # Compute energy and entropy from moments (texture descriptors)
            try:
                moment_values = [features[f'krawtchouk_moment_abs_{n}_{m}'] 
                               for n in range(max_order + 1) 
                               for m in range(max_order + 1) 
                               if n + m <= max_order]
                
                features['krawtchouk_energy'] = float(np.sum(np.array(moment_values) ** 2))
                
                # Normalize for entropy calculation
                moment_values_norm = np.array(moment_values)
                moment_sum = np.sum(moment_values_norm)
                if moment_sum > 1e-10:
                    moment_probs = moment_values_norm / moment_sum
                    moment_probs = moment_probs[moment_probs > 1e-10]  # Remove zeros
                    features['krawtchouk_entropy'] = float(-np.sum(moment_probs * np.log2(moment_probs)))
                else:
                    features['krawtchouk_entropy'] = 0.0
            except Exception as e:
                self.logger.warning(f"Error computing Krawtchouk energy/entropy: {str(e)}")
                features['krawtchouk_energy'] = 0.0
                features['krawtchouk_entropy'] = 0.0
            
            self.logger.debug(f"Extracted {len(features)} Krawtchouk moment features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting Krawtchouk moments: {str(e)}")
            # Return default zero features on error
            default_features = {}
            for n in range(max_order + 1):
                for m in range(max_order + 1):
                    if n + m <= max_order:
                        default_features[f'krawtchouk_moment_{n}_{m}'] = 0.0
                        default_features[f'krawtchouk_moment_abs_{n}_{m}'] = 0.0
            default_features['krawtchouk_invariant_1'] = 0.0
            default_features['krawtchouk_invariant_2'] = 0.0
            default_features['krawtchouk_invariant_3'] = 0.0
            default_features['krawtchouk_invariant_4'] = 0.0
            default_features['krawtchouk_energy'] = 0.0
            default_features['krawtchouk_entropy'] = 0.0
            return default_features

    def extract_abcde_features(self, image, mask):
        """Extract ABCDE rule features - critical for dermatological classification."""
        try:
            features = {}
            
            # A - ASYMMETRY (Enhanced)
            # Calculate asymmetry in multiple directions
            center_y, center_x = ndimage.center_of_mass(mask)
            
            # Horizontal asymmetry
            left_half = mask[:, :int(center_x)]
            right_half = np.fliplr(mask[:, int(center_x):])
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_resized = left_half[:, :min_width]
                right_resized = right_half[:, :min_width]
                horizontal_asymmetry = np.sum(np.abs(left_resized.astype(int) - right_resized.astype(int))) / np.sum(mask)
                features['asymmetry_horizontal'] = float(horizontal_asymmetry)
            else:
                features['asymmetry_horizontal'] = 0.0
            
            # Vertical asymmetry
            top_half = mask[:int(center_y), :]
            bottom_half = np.flipud(mask[int(center_y):, :])
            min_height = min(top_half.shape[0], bottom_half.shape[0])
            if min_height > 0:
                top_resized = top_half[:min_height, :]
                bottom_resized = bottom_half[:min_height, :]
                vertical_asymmetry = np.sum(np.abs(top_resized.astype(int) - bottom_resized.astype(int))) / np.sum(mask)
                features['asymmetry_vertical'] = float(vertical_asymmetry)
            else:
                features['asymmetry_vertical'] = 0.0
            
            # B - BORDER IRREGULARITY (Enhanced)
            from skimage.segmentation import find_boundaries
            border = find_boundaries(mask, mode='inner')
            border_coords = np.column_stack(np.where(border))
            
            if len(border_coords) > 10:
                # Calculate border fractal dimension
                features['border_fractal_dimension'] = self.calculate_fractal_dimension(border_coords)
                
                # Border curvature analysis
                if len(border_coords) > 20:
                    features['border_curvature_variance'] = self.calculate_curvature_variance(border_coords)
                else:
                    features['border_curvature_variance'] = 0.0
            else:
                features['border_fractal_dimension'] = 1.0
                features['border_curvature_variance'] = 0.0
            
            # C - COLOR VARIATION (Enhanced) - STABLE VERSION
            if len(image.shape) == 3 and image.shape[2] >= 3:
                rgb_image = image[:,:,:3]
                hsv_image = color.rgb2hsv(rgb_image)
                
                # STABLE: Use variance-based color count instead of K-means
                masked_rgb = rgb_image[mask]
                if len(masked_rgb) > 10:
                    # Calculate color diversity based on variance
                    color_var = np.var(masked_rgb, axis=0)
                    total_color_variance = np.sum(color_var)
                    
                    # Empirical mapping: higher variance = more colors
                    if total_color_variance < 0.01:
                        dominant_colors = 1
                    elif total_color_variance < 0.05:
                        dominant_colors = 2
                    elif total_color_variance < 0.15:
                        dominant_colors = 3
                    elif total_color_variance < 0.30:
                        dominant_colors = 4
                    else:
                        dominant_colors = min(6, int(total_color_variance * 10))
                    
                    features['dominant_colors_count'] = dominant_colors
                else:
                    features['dominant_colors_count'] = 1
                
                # These are stable - keep them
                features['color_entropy_rgb'] = self.calculate_color_entropy(rgb_image, mask)
                features['color_entropy_hsv'] = self.calculate_color_entropy(hsv_image, mask)
                features['color_uniformity'] = self.calculate_color_uniformity(rgb_image, mask)
            
            # D - DIAMETER-related features
            regionprops = measure.regionprops(mask.astype(int))
            if regionprops:
                props = regionprops[0]
                features['equivalent_diameter'] = float(props.equivalent_diameter)
                features['major_axis_length'] = float(props.major_axis_length)
                features['minor_axis_length'] = float(props.minor_axis_length)
                features['axis_ratio'] = float(props.major_axis_length / props.minor_axis_length) if props.minor_axis_length > 0 else 1.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in ABCDE features: {str(e)}")
            return {}

    def calculate_fractal_dimension(self, coords):
        """Calculate fractal dimension of border."""
        try:
            # Box counting method
            scales = np.logspace(0.5, 2.5, num=10, dtype=int)
            counts = []
            
            for scale in scales:
                # Create grid
                grid_size = scale
                x_bins = np.arange(coords[:, 1].min(), coords[:, 1].max() + grid_size, grid_size)
                y_bins = np.arange(coords[:, 0].min(), coords[:, 0].max() + grid_size, grid_size)
                
                # Count occupied boxes
                hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[y_bins, x_bins])
                counts.append(np.count_nonzero(hist))
            
            # Fit line to log-log plot
            if len(counts) > 1 and all(c > 0 for c in counts):
                coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
                return float(-coeffs[0])
            else:
                return 1.0
        except:
            return 1.0

    def calculate_curvature_variance(self, coords):
        """Calculate variance in border curvature."""
        try:
            if len(coords) < 5:
                return 0.0
            
            # Calculate curvature at each point
            curvatures = []
            for i in range(2, len(coords) - 2):
                # Use 5-point stencil for curvature calculation
                p1, p2, p3, p4, p5 = coords[i-2:i+3]
                
                # Calculate derivatives
                dx1 = p3[1] - p1[1]
                dy1 = p3[0] - p1[0]
                dx2 = p5[1] - p3[1]
                dy2 = p5[0] - p3[0]
                
                # Calculate curvature
                denom = (dx1**2 + dy1**2)**1.5
                if denom > 0:
                    curvature = abs(dx1 * dy2 - dy1 * dx2) / denom
                    curvatures.append(curvature)
            
            return float(np.var(curvatures)) if curvatures else 0.0
        except:
            return 0.0

    def count_dominant_colors(self, image, mask):
        """Count number of dominant colors in the lesion."""
        try:
            from sklearn.cluster import KMeans
            
            masked_pixels = image[mask].reshape(-1, 3)
            if len(masked_pixels) < 10:
                return 1
            
            # Use k-means to find color clusters
            max_k = min(8, len(masked_pixels))
            best_k = 1
            
            for k in range(2, max_k + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(masked_pixels)
                    
                    # Calculate within-cluster sum of squares
                    wcss = kmeans.inertia_
                    
                    # Simple elbow method approximation
                    if k == 2:
                        prev_wcss = wcss
                        best_k = 2
                    else:
                        improvement = (prev_wcss - wcss) / prev_wcss
                        if improvement < 0.1:  # If improvement is small, stop
                            break
                        best_k = k
                        prev_wcss = wcss
                except:
                    break
            
            return best_k
        except:
            return 1

    def calculate_color_entropy(self, image, mask):
        """Calculate color entropy."""
        try:
            masked_pixels = image[mask]
            if len(masked_pixels) == 0:
                return 0.0
            
            # Convert to quantized color space
            quantized = (masked_pixels * 15).astype(int)  # 16 levels per channel
            
            # Create color histogram
            colors, counts = np.unique(quantized.reshape(-1, quantized.shape[-1]), axis=0, return_counts=True)
            
            # Calculate entropy
            probabilities = counts / np.sum(counts)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return float(entropy)
        except:
            return 0.0

    def calculate_color_uniformity(self, image, mask):
        """Calculate color uniformity (inverse of color variation)."""
        try:
            masked_pixels = image[mask]
            if len(masked_pixels) == 0:
                return 1.0
            
            # Calculate coefficient of variation for each channel
            cv_values = []
            for channel in range(image.shape[2]):
                channel_data = masked_pixels[:, channel]
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                cv = std_val / mean_val if mean_val > 0 else 0
                cv_values.append(cv)
            
            # Uniformity is inverse of average coefficient of variation
            avg_cv = np.mean(cv_values)
            uniformity = 1.0 / (1.0 + avg_cv)
            
            return float(uniformity)
        except:
            return 1.0
           
    def extract_enhanced_color_features(self, image, mask):
            """Extract enhanced color features for better BCC vs SK discrimination."""
            try:
                features = {}
                
                if len(image.shape) == 3 and image.shape[2] >= 3:
                    rgb_image = image[:,:,:3]
                    hsv_image = color.rgb2hsv(rgb_image)
                    lab_image = color.rgb2lab(rgb_image)
                
                    # Color moments (more robust than basic statistics)
                    for space_name, space_image in [('rgb', rgb_image), ('hsv', hsv_image), ('lab', lab_image)]:
                        for channel in range(space_image.shape[2]):
                            channel_data = space_image[:,:,channel][mask]
                            
                            if len(channel_data) > 0:
                                # Central moments
                                mean_val = np.mean(channel_data)
                                features[f'{space_name}_{channel}_moment_1'] = float(mean_val)
                                features[f'{space_name}_{channel}_moment_2'] = float(np.mean((channel_data - mean_val)**2))
                                features[f'{space_name}_{channel}_moment_3'] = float(np.mean((channel_data - mean_val)**3))
                                features[f'{space_name}_{channel}_moment_4'] = float(np.mean((channel_data - mean_val)**4))
                                
                                # Percentile-based features (robust to outliers)
                                features[f'{space_name}_{channel}_p10'] = float(np.percentile(channel_data, 10))
                                features[f'{space_name}_{channel}_p25'] = float(np.percentile(channel_data, 25))
                                features[f'{space_name}_{channel}_p75'] = float(np.percentile(channel_data, 75))
                                features[f'{space_name}_{channel}_p90'] = float(np.percentile(channel_data, 90))
                                features[f'{space_name}_{channel}_iqr'] = float(np.percentile(channel_data, 75) - np.percentile(channel_data, 25))
                    
                    # Color ratios (important for distinguishing BCC vs SK)
                    rgb_masked = rgb_image[mask]
                    if len(rgb_masked) > 0:
                        r_mean, g_mean, b_mean = np.mean(rgb_masked, axis=0)
                        
                        # Traditional color ratios
                        total_intensity = r_mean + g_mean + b_mean
                        if total_intensity > 0:
                            features['red_ratio'] = float(r_mean / total_intensity)
                            features['green_ratio'] = float(g_mean / total_intensity)
                            features['blue_ratio'] = float(b_mean / total_intensity)
                        
                        # Specific ratios for dermatology
                        features['rg_ratio'] = float(r_mean / g_mean) if g_mean > 0 else 1.0
                        features['rb_ratio'] = float(r_mean / b_mean) if b_mean > 0 else 1.0
                        features['gb_ratio'] = float(g_mean / b_mean) if b_mean > 0 else 1.0
                
                return features
                
            except Exception as e:
                self.logger.error(f"Error in enhanced color features: {str(e)}")
                return {}
    
    def visualize_features_for_publication(self, image_path, mask=None, save_path=None, dpi=300):
        """
        Create publication-quality visualization of extracted features.
        
        Generates a comprehensive multi-panel figure showing:
        1. Original image with mask overlay
        2. Geometric features (asymmetry axes, convex hull, border)
        3. Color distribution heatmaps (HSV)
        4. Texture features (Gabor responses, LBP)
        5. Border analysis and gradient visualization
        6. ABCDE clinical features summary
        7. Key feature values bar chart
        
        Args:
            image_path: Path to the image file
            mask: Binary mask (auto-generated if None)
            save_path: Path to save the figure (optional, auto-generated if None)
            dpi: Resolution for publication (default: 300)
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        try:
            # Load image
            if not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if mask is None:
                # Auto-generate mask using Otsu's thresholding
                gray = color.rgb2gray(image)
                threshold = filters.threshold_otsu(gray)
                mask = gray < threshold
                self.logger.info("Auto-generated mask using Otsu's method")
            
            # Extract all features
            self.logger.info("Extracting features for visualization...")
            features = self.extract_all_features(image, mask)
            
            # Create figure with GridSpec layout
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
            
            # ===== 1. Original image with mask overlay =====
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(image)
            ax1.contour(mask, colors='yellow', linewidths=2, alpha=0.8)
            ax1.set_title('(a) Original Image with Lesion Mask', fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            # ===== 2. Geometric features visualization =====
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(image)
            
            # Draw geometric features
            regionprops = measure.regionprops(mask.astype(int))
            if regionprops:
                props = regionprops[0]
                y0, x0 = props.centroid
                orientation = props.orientation
                
                # Major axis
                x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length
                y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length
                x2 = x0 - np.cos(orientation) * 0.5 * props.major_axis_length
                y2 = y0 + np.sin(orientation) * 0.5 * props.major_axis_length
                ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2.5, label='Major axis')
                
                # Minor axis
                x1_min = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                y1_min = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length
                x2_min = x0 + np.sin(orientation) * 0.5 * props.minor_axis_length
                y2_min = y0 + np.cos(orientation) * 0.5 * props.minor_axis_length
                ax2.plot([x1_min, x2_min], [y1_min, y2_min], 'b-', linewidth=2.5, label='Minor axis')
                
                # Centroid
                ax2.plot(x0, y0, 'go', markersize=10, label='Centroid', markeredgecolor='white', markeredgewidth=1)
                
                # Convex hull
                hull = convex_hull_image(mask)
                ax2.contour(hull, colors='lime', linewidths=2, linestyles='dashed', alpha=0.8)
                
            ax2.set_title('(b) Geometric Features\n(A: Asymmetry Analysis)', fontsize=11, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax2.axis('off')
            
            # ===== 3. Border irregularity =====
            ax3 = fig.add_subplot(gs[0, 2])
            border = find_boundaries(mask, mode='inner')
            border_overlay = image.copy()
            border_overlay[border] = [255, 0, 0]
            ax3.imshow(border_overlay)
            
            compactness_val = features.get("compactness", 0)
            smoothness_val = features.get("border_smoothness", 0)
            ax3.set_title(f'(c) Border Irregularity (B)\nCompactness: {compactness_val:.3f}\nSmoothness: {smoothness_val:.3f}', 
                         fontsize=11, fontweight='bold')
            ax3.axis('off')
            
            # ===== 4. Color distribution (Hue channel) =====
            ax4 = fig.add_subplot(gs[0, 3])
            hsv_image = color.rgb2hsv(image)
            h_channel = hsv_image[:,:,0].copy()
            h_channel[~mask] = 0
            im4 = ax4.imshow(h_channel, cmap='hsv', vmin=0, vmax=1)
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.ax.tick_params(labelsize=8)
            
            color_count = features.get("dominant_colors_count", 1)
            color_entropy = features.get("color_entropy_rgb", 0)
            ax4.set_title(f'(d) Hue Distribution (C: Color)\nDominant Colors: {color_count}\nEntropy: {color_entropy:.3f}', 
                         fontsize=11, fontweight='bold')
            ax4.axis('off')
            
            # ===== 5-8. Gabor filter responses (4 orientations) =====
            gray_img = color.rgb2gray(image)
            gabor_responses = []
            orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            orientation_names = ['0°', '45°', '90°', '135°']
            
            for theta in orientations:
                gabor_kernel = cv2.getGaborKernel((31, 31), sigma=4, theta=theta, 
                                                 lambd=10, gamma=0.5, psi=0)
                filtered = cv2.filter2D(gray_img, cv2.CV_64F, gabor_kernel)
                filtered_masked = filtered.copy()
                filtered_masked[~mask] = 0
                gabor_responses.append(filtered_masked)
            
            for idx, (response, orientation_name) in enumerate(zip(gabor_responses, orientation_names)):
                ax = fig.add_subplot(gs[1, idx])
                im = ax.imshow(response, cmap='viridis')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                ax.set_title(f'(e-{idx+1}) Gabor Filter {orientation_name}', fontsize=10, fontweight='bold')
                ax.axis('off')
            
            # ===== 9. Local Binary Pattern (LBP) =====
            ax9 = fig.add_subplot(gs[2, 0])
            gray_uint8 = (color.rgb2gray(image) * 255).astype(np.uint8)
            lbp = local_binary_pattern(gray_uint8, P=8*3, R=3, method='uniform')
            lbp_masked = lbp.copy()
            lbp_masked[~mask] = 0
            im9 = ax9.imshow(lbp_masked, cmap='gray')
            cbar9 = plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
            cbar9.ax.tick_params(labelsize=8)
            ax9.set_title('(f) Local Binary Pattern\n(Texture Analysis)', fontsize=11, fontweight='bold')
            ax9.axis('off')
            
            # ===== 10. Gradient magnitude (Edge strength) =====
            ax10 = fig.add_subplot(gs[2, 1])
            gray_norm = color.rgb2gray(image)
            sobelx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            magnitude_masked = magnitude.copy()
            magnitude_masked[~mask] = 0
            im10 = ax10.imshow(magnitude_masked, cmap='hot')
            cbar10 = plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
            cbar10.ax.tick_params(labelsize=8)
            
            edge_mean = features.get("edge_strength_mean", 0)
            ax10.set_title(f'(g) Edge Strength\n(Border Gradient)\nMean: {edge_mean:.3f}', 
                          fontsize=11, fontweight='bold')
            ax10.axis('off')
            
            # ===== 11. ABCDE Clinical Features Summary =====
            ax11 = fig.add_subplot(gs[2, 2])
            ax11.axis('off')
            
            abcde_text = f"""ABCDE Rule Features:

A - Asymmetry:
  • Horizontal: {features.get('asymmetry_horizontal', 0):.4f}
  • Vertical: {features.get('asymmetry_vertical', 0):.4f}
  • Diagonal: {features.get('asymmetry_diagonal', 0):.4f}

B - Border:
  • Compactness: {features.get('compactness', 0):.4f}
  • Fractal Dim: {features.get('border_fractal_dimension', 1.0):.4f}
  • Smoothness: {features.get('border_smoothness', 0):.4f}

C - Color:
  • Dominant colors: {features.get('dominant_colors_count', 1)}
  • RGB entropy: {features.get('color_entropy_rgb', 0):.4f}
  • Std deviation: {features.get('color_std', 0):.2f}

D - Diameter:
  • Equivalent: {features.get('equivalent_diameter', 0):.1f} px
  • Major axis: {features.get('major_axis_length', 0):.1f} px
  • Axis ratio: {features.get('axis_ratio', 1):.4f}

E - Evolving (Texture):
  • GLCM contrast: {features.get('glcm_contrast_d1_a0', 0):.4f}
  • LBP uniformity: {features.get('lbp_uniformity', 0):.4f}
            """
            ax11.text(0.05, 0.5, abcde_text, fontsize=9, family='monospace',
                    verticalalignment='center', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1))
            ax11.set_title('(h) ABCDE Clinical Features', fontsize=11, fontweight='bold')
            
            # ===== 12. Key Feature Values Bar Chart =====
            ax12 = fig.add_subplot(gs[2, 3])
            
            # Select most discriminative features for visualization
            feature_display = {
                'Compactness': features.get('compactness', 0),
                'Asymmetry (H)': features.get('asymmetry_horizontal', 0),
                'Fractal Dim': features.get('border_fractal_dimension', 1.0) - 1.0,  # Normalize
                'Color Count': features.get('dominant_colors_count', 1) / 10.0,  # Scale
                'Color Entropy': features.get('color_entropy_rgb', 0) / 8.0,  # Normalize
                'GLCM Contrast': min(features.get('glcm_contrast_d1_a0', 0) / 100.0, 1.0),  # Clip
                'Gabor Energy': min(features.get('gabor_t0_s3_f10_mean', 0) / 50.0, 1.0),  # Clip
                'Axis Ratio': features.get('axis_ratio', 1),
            }
            
            feature_names = list(feature_display.keys())
            feature_values = list(feature_display.values())
            
            # Color bars based on value (green=low, yellow=medium, red=high)
            colors_bar = []
            for val in feature_values:
                if val < 0.33:
                    colors_bar.append('#2ecc71')  # Green
                elif val < 0.67:
                    colors_bar.append('#f39c12')  # Orange
                else:
                    colors_bar.append('#e74c3c')  # Red
            
            y_pos = np.arange(len(feature_names))
            bars = ax12.barh(y_pos, feature_values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax12.set_yticks(y_pos)
            ax12.set_yticklabels(feature_names, fontsize=9)
            ax12.set_xlabel('Normalized Feature Value', fontsize=10)
            ax12.set_xlim(0, 1.0)
            ax12.set_title('(i) Key Feature Values\n(Normalized)', fontsize=11, fontweight='bold')
            ax12.grid(axis='x', alpha=0.3, linestyle='--')
            ax12.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, feature_values)):
                ax12.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8)
            
            # Add main title
            image_name = Path(image_path).stem
            fig.suptitle(f'Comprehensive Feature Extraction Analysis for Skin Lesion Classification\nImage: {image_name}', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            # Auto-generate save path if not provided
            if save_path is None:
                output_dir = Path('output/features')
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f'{image_name}_feature_visualization.png'
            
            # Save figure
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"✅ Feature visualization saved to: {save_path}")
            
            plt.close(fig)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Main execution for testing
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create extractor
    extractor = ConventionalFeatureExtractor()
    
    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Optional: mask path
        mask_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask_img > 127
            
        # Generate visualization
        print(f"\n{'='*60}")
        print(f"Generating Feature Visualization")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        if mask_path:
            print(f"Mask: {mask_path}")
        print(f"{'='*60}\n")
        
        fig = extractor.visualize_features_for_publication(
            image_path=image_path,
            mask=mask,
            dpi=300
        )
        
        if fig:
            print("\n✅ Visualization completed successfully!")
        else:
            print("\n❌ Visualization failed!")
    else:
        # Demo with sample images
        print("\n" + "="*60)
        print("Feature Visualization Tool")
        print("="*60)
        print("\nUsage:")
        print("  python conventional_features.py <image_path> [mask_path]")
        print("\nExample:")
        print("  python conventional_features.py data/bcc_segmented/ISIC_0000001.jpg")
        print("  python conventional_features.py data/sk_segmented/ISIC_0000001.jpg mask.png")
        print("="*60)
        
        # Try to find sample images
        sample_paths = [
            'data/bcc_segmented',
            'data/sk_segmented',
            'data/bcc',
            'data/sk'
        ]
        
        for sample_dir in sample_paths:
            if os.path.exists(sample_dir):
                import glob
                images = glob.glob(f"{sample_dir}/*.jpg") + glob.glob(f"{sample_dir}/*.png")
                if images:
                    print(f"\n✨ Found sample images in {sample_dir}/")
                    print(f"Running demo with: {images[0]}\n")
                    
                    fig = extractor.visualize_features_for_publication(
                        image_path=images[0],
                        dpi=300
                    )
                    
                    if fig:
                        print("\n✅ Demo visualization completed!")
                    break