import numpy as np
import logging
from skimage import color, feature, measure, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import convex_hull_image
from scipy import ndimage, stats
import pywt
import cv2
from scipy.special import comb
from scipy.special import gammaln
from scipy import special

class ConventionalFeatureExtractor:
    def __init__(self):
        """Initialize the conventional feature extractor."""
        self.logger = logging.getLogger(__name__)
        
    def extract_all_features(self, image, mask=None):
        try:
            features = {}
            
            if mask is None:
                mask = np.ones(image.shape[:2], dtype=bool)

            # Extract texture features
            texture_features = self.extract_texture_features(image, mask)
            features.update(texture_features)

            # Extract color features from different color spaces
            color_features = self.extract_color_features(image, mask)
            features.update(color_features)

            # Extract MDFKT features
            mdfkt_features = self.extract_mdfkt_features(image, mask, N=64, p=0.5)
            features.update(mdfkt_features)
            
            """

            # Extract geometric features
            geometric_features = self.extract_geometric_features(mask)
            features.update(geometric_features)

            # Enhanced color features
            enhanced_color_features = self.extract_enhanced_color_features(image, mask)
            features.update(enhanced_color_features)

            # Extract ABCDE features
            abcde_features = self.extract_abcde_features(image, mask)
            features.update(abcde_features)
            """
            
            
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



    def extract_mdfkt_features(self, image, mask, N=64, p=0.5):
        try:
            features = {}
            
            # Ensure image is in correct format
            if len(image.shape) < 3:
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] < 3:
                image = np.stack([image[:,:,0]] * 3, axis=2)
            
            rgb_image = image[:,:,:3]
            
            # Normalize to [0, 1] if needed
            if np.max(rgb_image) > 1.0:
                rgb_image = rgb_image / 255.0
            
            # Check if mask is valid
            if np.sum(mask) == 0:
                self.logger.warning("Empty mask provided for MDFKT features")
                return self._get_default_mdfkt_features()
            
            # Get bounding box of the lesion
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                self.logger.warning("Invalid mask for MDFKT features")
                return self._get_default_mdfkt_features()
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Extract lesion region from RGB image
            lesion_rgb = rgb_image[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Zero out background (consistent with texture features)
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            for channel_idx in range(3):
                lesion_rgb[:, :, channel_idx][~lesion_mask] = 0
            
            # Initialize MDFKT components
            K0 = self._compute_K0_matrix_mdfkt(N, p)
            Lambda = self._compute_lambda(N)
            
            # Verify K0 matrix validity
            if not np.all(np.isfinite(K0)):
                self.logger.error("K0 matrix contains non-finite values")
                return self._get_default_mdfkt_features()
            
            # Process each RGB channel
            channels = ['R', 'G', 'B']
            
            for idx, channel_name in enumerate(channels):
                # Extract channel from the cropped lesion (already zeroed)
                channel = lesion_rgb[:, :, idx]
                
                # Resize channel to N x N for MDFKT
                channel_resized = cv2.resize(channel, (N, N), interpolation=cv2.INTER_LINEAR)
                
                # Check for valid input
                if not np.all(np.isfinite(channel_resized)):
                    self.logger.warning(f"Non-finite values in {channel_name} channel, cleaning")
                    channel_resized = np.nan_to_num(channel_resized, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Apply 2D MDFKT (supervisor's approach)
                Y = self._apply_2D_MDFKT(K0, Lambda, channel_resized)
                
                # Verify transform output
                if not np.all(np.isfinite(Y)):
                    self.logger.warning(f"Non-finite values in {channel_name} MDFKT output, cleaning")
                    Y = np.nan_to_num(Y, nan=0.0+0j, posinf=1e10+0j, neginf=-1e10+0j)
                
                # Extract features from MDFKT coefficients
                channel_features = self._extract_mdfkt_channel_features(Y, channel_name)
                features.update(channel_features)
            
            # Add cross-channel MDFKT features
            cross_features = self._extract_mdfkt_cross_channel_features(features, channels)
            features.update(cross_features)
            
            self.logger.debug(f"Extracted {len(features)} MDFKT features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting MDFKT features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_mdfkt_features()

    def _krawtchouk_poly(self, n, x, N, p):
        """
        Compute raw Krawtchouk polynomial.
        K_n(x; p, N) = sum_{j=0}^{n} (-1)^j * C(x,j) * C(N-x, n-j) * (p/(1-p))^j
        """
        try:
            if p <= 0 or p >= 1:
                return 0.0
            
            if x < 0 or x > N or n < 0 or n > N:
                return 0.0
            
            s = 0.0
            for j in range(n + 1):
                try:
                    # Use exact=False to prevent overflow
                    c1 = comb(x, j, exact=False)
                    c2 = comb(N - x, n - j, exact=False)
                    
                    # Calculate power term safely
                    if j > 0:
                        power_term = (p / (1 - p)) ** j
                        # Clamp extreme values
                        if power_term > 1e10:
                            power_term = 1e10
                        elif power_term < 1e-10:
                            power_term = 1e-10
                    else:
                        power_term = 1.0
                    
                    term = ((-1) ** j) * c1 * c2 * power_term
                    
                    if np.isfinite(term):
                        s += term
                        
                except (OverflowError, ValueError):
                    continue
            
            return s
            
        except Exception as e:
            self.logger.warning(f"Error in Krawtchouk polynomial: {str(e)}")
            return 0.0

    def _normalized_krawtchouk(self, n, x, N, p):
        """
        Compute normalized Krawtchouk polynomial,
        Ensures orthonormality of the basis functions.
        """
        try:
            # Weight function: w(x) = C(N,x) * p^x * (1-p)^(N-x)
            w_x = comb(N, x, exact=False) * (p ** x) * ((1 - p) ** (N - x))
            
            # Normalization factor: sqrt(C(N,n) * p^n * (1-p)^(N-n))
            norm = np.sqrt(comb(N, n, exact=False) * (p ** n) * ((1 - p) ** (N - n)))
            
            # Compute raw Krawtchouk polynomial
            K_n_x = self._krawtchouk_poly(n, x, N, p)
            
            # Return normalized value
            if norm > 1e-10 and w_x > 0:
                return K_n_x * np.sqrt(w_x) / norm
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error in normalized Krawtchouk: {str(e)}")
            return 0.0

    def _compute_K0_matrix_mdfkt(self, N, p):
        """Compute orthonormal Krawtchouk matrix K0 for MDFKT."""
        try:
            K = np.zeros((N, N), dtype=float)
            for n in range(N):
                for x in range(N):
                    K[n, x] = self._normalized_krawtchouk(n, x, N - 1, p)
            
            if not np.all(np.isfinite(K)):
                self.logger.warning("Non-finite values in K matrix")
                K = np.nan_to_num(K, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Orthonormalize via QR decomposition
            Q, R = np.linalg.qr(K.T)
            K0 = Q.T
            
            if not np.all(np.isfinite(K0)):
                self.logger.warning("QR produced non-finite values, using identity")
                return np.eye(N)
            
            return K0
            
        except Exception as e:
            self.logger.error(f"Error computing K0: {str(e)}")
            return np.eye(N)

    def _compute_lambda(self, N):
        k = np.arange(1, N+1)      # 1..N
        Lambda = 1.0 / k
        return Lambda

    """  
    def _compute_lambda(self, N):
        #Compute lambda_k = exp(i (2k+1)π / N), k = 0..N-1.
        k = np.arange(N)
        Lambda = np.exp(1j * (2*k + 1) * np.pi / N)
        return Lambda

    """  
    
    """  
    def _compute_lambda(self, N):
        Lambda = np.zeros(N, dtype=complex)
        for k in range(N):
            Lambda[k] = np.exp(1j * 2 * np.pi * k / N)
        return Lambda
    """  
    """  
    def _compute_lambda(self, N):
        # Compute Lambda eigenvalues.
        Lambda = np.zeros(N, dtype=complex)
        
        for k in range(0, N-1):
            Lambda[k] = 1.0 / (N - k)
        
        return Lambda
    """

    def _apply_2D_MDFKT(self, K0, Lambda, f):
        """Apply 2D MDFKT forward transform."""
        try:
            if not np.all(np.isfinite(f)):
                f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = K0 @ f
                
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0, posinf=1e10, neginf=-1e10)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = Lambda[:, None] * temp
            
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0+0j, posinf=1e10+0j, neginf=-1e10+0j)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                Y = temp @ K0.T
            
            if not np.all(np.isfinite(Y)):
                Y = np.nan_to_num(Y, nan=0.0+0j, posinf=1e10+0j, neginf=-1e10+0j)
            
            return Y
            
        except Exception as e:
            self.logger.error(f"Error in 2D MDFKT: {str(e)}")
            return np.zeros_like(f, dtype=complex)

    def _extract_mdfkt_channel_features(self, Y, channel_name):
        """Extract features from MDFKT coefficients."""
        try:
            features = {}
            
            if not np.all(np.isfinite(Y)):
                Y = np.nan_to_num(Y, nan=0.0+0j, posinf=1e10+0j, neginf=-1e10+0j)
            
            # Extract components
            real_part = np.real(Y)
            imag_part = np.imag(Y)
            magnitude = np.abs(Y)
            phase = np.angle(Y)
            
            # Real part features
            features[f'mdfkt_{channel_name}_real_mean'] = float(np.mean(real_part))
            features[f'mdfkt_{channel_name}_real_std'] = float(np.std(real_part))
            features[f'mdfkt_{channel_name}_real_max'] = float(np.max(real_part))
            features[f'mdfkt_{channel_name}_real_min'] = float(np.min(real_part))
            features[f'mdfkt_{channel_name}_real_energy'] = float(np.sum(real_part**2))
            
            # Imaginary part features
            features[f'mdfkt_{channel_name}_imag_mean'] = float(np.mean(imag_part))
            features[f'mdfkt_{channel_name}_imag_std'] = float(np.std(imag_part))
            features[f'mdfkt_{channel_name}_imag_max'] = float(np.max(imag_part))
            features[f'mdfkt_{channel_name}_imag_min'] = float(np.min(imag_part))
            features[f'mdfkt_{channel_name}_imag_energy'] = float(np.sum(imag_part**2))
            
            # Magnitude features
            features[f'mdfkt_{channel_name}_magnitude_mean'] = float(np.mean(magnitude))
            features[f'mdfkt_{channel_name}_magnitude_std'] = float(np.std(magnitude))
            features[f'mdfkt_{channel_name}_magnitude_max'] = float(np.max(magnitude))
            features[f'mdfkt_{channel_name}_magnitude_energy'] = float(np.sum(magnitude**2))
            
            # Magnitude entropy
            mag_nonzero = magnitude[magnitude > 1e-10]
            if len(mag_nonzero) > 0:
                mag_norm = mag_nonzero / np.sum(mag_nonzero)
                features[f'mdfkt_{channel_name}_magnitude_entropy'] = float(
                    -np.sum(mag_norm * np.log2(mag_norm + 1e-10))
                )
            else:
                features[f'mdfkt_{channel_name}_magnitude_entropy'] = 0.0
            
            # Phase features
            features[f'mdfkt_{channel_name}_phase_mean'] = float(np.mean(phase))
            features[f'mdfkt_{channel_name}_phase_std'] = float(np.std(phase))
            features[f'mdfkt_{channel_name}_phase_range'] = float(np.max(phase) - np.min(phase))
            
            # Frequency-domain features
            N = Y.shape[0]
            center = N // 2
            
            low_freq_mask = np.zeros_like(magnitude, dtype=bool)
            low_freq_mask[center-N//4:center+N//4, center-N//4:center+N//4] = True
            
            high_freq_mask = ~low_freq_mask
            
            low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
            high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
            total_energy = np.sum(magnitude**2)
            
            features[f'mdfkt_{channel_name}_low_freq_ratio'] = float(
                low_freq_energy / total_energy if total_energy > 0 else 0
            )
            features[f'mdfkt_{channel_name}_high_freq_ratio'] = float(
                high_freq_energy / total_energy if total_energy > 0 else 0
            )
            
            # Spectral entropy
            if total_energy > 0:
                mag_norm = (magnitude**2) / total_energy
                mag_norm = mag_norm[mag_norm > 1e-10]
                if len(mag_norm) > 0:
                    features[f'mdfkt_{channel_name}_spectral_entropy'] = float(
                        -np.sum(mag_norm * np.log2(mag_norm))
                    )
                else:
                    features[f'mdfkt_{channel_name}_spectral_entropy'] = 0.0
            else:
                features[f'mdfkt_{channel_name}_spectral_entropy'] = 0.0
            
            # Higher-order statistics
            try:
                features[f'mdfkt_{channel_name}_mag_skewness'] = float(stats.skew(magnitude.flatten()))
                features[f'mdfkt_{channel_name}_mag_kurtosis'] = float(stats.kurtosis(magnitude.flatten()))
            except:
                features[f'mdfkt_{channel_name}_mag_skewness'] = 0.0
                features[f'mdfkt_{channel_name}_mag_kurtosis'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting MDFKT channel features: {str(e)}")
            return {}

    def _extract_mdfkt_cross_channel_features(self, features, channels):
        """Extract cross-channel MDFKT features."""
        try:
            cross_features = {}
            
            # Magnitude ratios between channels
            for i, ch1 in enumerate(channels):
                for ch2 in channels[i+1:]:
                    mag1 = features.get(f'mdfkt_{ch1}_magnitude_mean', 0)
                    mag2 = features.get(f'mdfkt_{ch2}_magnitude_mean', 0)
                    cross_features[f'mdfkt_{ch1}{ch2}_mag_ratio'] = float(
                        mag1 / mag2 if mag2 > 1e-10 else 1.0
                    )
            
            # RGB energy contributions
            r_mag = features.get('mdfkt_R_magnitude_energy', 0)
            g_mag = features.get('mdfkt_G_magnitude_energy', 0)
            b_mag = features.get('mdfkt_B_magnitude_energy', 0)
            total = r_mag + g_mag + b_mag
            
            if total > 1e-10:
                cross_features['mdfkt_rgb_r_contribution'] = float(r_mag / total)
                cross_features['mdfkt_rgb_g_contribution'] = float(g_mag / total)
                cross_features['mdfkt_rgb_b_contribution'] = float(b_mag / total)
            else:
                cross_features['mdfkt_rgb_r_contribution'] = 0.33
                cross_features['mdfkt_rgb_g_contribution'] = 0.33
                cross_features['mdfkt_rgb_b_contribution'] = 0.34
            
            # Channel variance
            mag_means = [features.get(f'mdfkt_{ch}_magnitude_mean', 0) for ch in channels]
            cross_features['mdfkt_channel_variance'] = float(np.var(mag_means))
            
            return cross_features
            
        except Exception as e:
            self.logger.error(f"Error extracting cross-channel features: {str(e)}")
            return {}

    def _get_default_mdfkt_features(self):
        """Return default MDFKT features."""
        features = {}
        channels = ['R', 'G', 'B']
        
        for ch in channels:
            for stat in ['mean', 'std', 'max', 'min', 'energy']:
                features[f'mdfkt_{ch}_real_{stat}'] = 0.0
                features[f'mdfkt_{ch}_imag_{stat}'] = 0.0
            
            for stat in ['mean', 'std', 'max', 'energy', 'entropy']:
                features[f'mdfkt_{ch}_magnitude_{stat}'] = 0.0
            
            for stat in ['mean', 'std', 'range']:
                features[f'mdfkt_{ch}_phase_{stat}'] = 0.0
            
            features[f'mdfkt_{ch}_low_freq_ratio'] = 0.0
            features[f'mdfkt_{ch}_high_freq_ratio'] = 0.0
            features[f'mdfkt_{ch}_spectral_entropy'] = 0.0
            features[f'mdfkt_{ch}_mag_skewness'] = 0.0
            features[f'mdfkt_{ch}_mag_kurtosis'] = 0.0
        
        features['mdfkt_RG_mag_ratio'] = 1.0
        features['mdfkt_RB_mag_ratio'] = 1.0
        features['mdfkt_GB_mag_ratio'] = 1.0
        features['mdfkt_rgb_r_contribution'] = 0.33
        features['mdfkt_rgb_g_contribution'] = 0.33
        features['mdfkt_rgb_b_contribution'] = 0.34
        features['mdfkt_channel_variance'] = 0.0
        
        return features


