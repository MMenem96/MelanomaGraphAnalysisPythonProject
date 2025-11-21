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
            
            # Enhanced color features
            enhanced_color_features = self.extract_enhanced_color_features(image, mask)
            features.update(enhanced_color_features)

            # Paper Fractional Krawtchouk moments
            paper_fractional_krawtchouk_moments = self.extract_paper_fractional_krawtchouk_moments(image, mask)
            features.update(paper_fractional_krawtchouk_moments)

            """

             # Paper Fractional Krawtchouk moments
            paper_fractional_krawtchouk_moments = self.extract_paper_fractional_krawtchouk_moments(image, mask)
            features.update(paper_fractional_krawtchouk_moments)

            # Extract geometric features
            geometric_features = self.extract_geometric_features(mask)
            features.update(geometric_features)

            # Extract color features from different color spaces
            color_features = self.extract_color_features(image, mask)
            features.update(color_features)

            # Extract ABCDE features
            abcde_features = self.extract_abcde_features(image, mask)
            features.update(abcde_features)

            
            # Extract texture features
            texture_features = self.extract_texture_features(image, mask)
            features.update(texture_features)
            
            # Enhanced color features
            enhanced_color_features = self.extract_enhanced_color_features(image, mask)
            features.update(enhanced_color_features)
            """

            """
            # Fractional Krawtchouk moments 
            fractional_krawtchouk_features = self.modified_extract_fractional_krawtchouk_moments(image, mask)
            features.update(fractional_krawtchouk_features)

            # Classical Krawtchouk moments
            krawtchouk_features = self.extract_krawtchouk_moments(image, mask)
            features.update(krawtchouk_features)

            # Fourier 
            fourier_features = self.extract_fourier_features(image, mask, num_coefficients=36)
            features.update(fourier_features)
            
            # DCT 
            dct_features = self.extract_dct_features(image, mask, num_coefficients=36)
            features.update(dct_features)
            
            # Hadamard 
            hadamard_features = self.extract_hadamard_features(image, mask, num_coefficients=36)
            features.update(hadamard_features)
            """
            
            
            """
            # Surface pattern features (BCC vs SK specific)
            surface_features = self.extract_surface_pattern_features(image, mask)
            features.update(surface_features)

            # Multi-scale texture features
            multiscale_features = self.extract_multiscale_texture_features(image, mask)
            features.update(multiscale_features)
            """

            # # Extract dermoscopic-specific features
            # dermoscopic_features = self.extract_dermoscopic_features(image, mask)
            # features.update(dermoscopic_features)



            
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
        
    def extract_dermoscopic_features(self, image, mask):
        """Extract dermoscopic-specific features for BCC vs SK classification."""
        try:
            features = {}
            
            # Convert to different color spaces for analysis
            if len(image.shape) == 3 and image.shape[2] >= 3:
                rgb_image = image[:,:,:3]
                hsv_image = color.rgb2hsv(rgb_image)
                lab_image = color.rgb2lab(rgb_image)
                gray = color.rgb2gray(rgb_image)
            else:
                gray = image if len(image.shape) == 2 else image[:,:,0]
                rgb_image = np.stack([gray] * 3, axis=2)
                hsv_image = color.rgb2hsv(rgb_image)
                lab_image = color.rgb2lab(rgb_image)
            
            # 1. ARBORIZING VESSELS DETECTION (BCC signature)
            # Use Frangi vesselness filter to detect vessel-like structures
            try:
                from skimage.filters import frangi
                from skimage.morphology import medial_axis
                
                # Updated frangi parameters to avoid deprecation warning
                vessels = frangi(gray, sigmas=range(1, 4, 1))
                vessels_masked = vessels[mask]
                
                if len(vessels_masked) > 0:
                    features['vessel_density'] = float(np.mean(vessels_masked))
                    features['vessel_max_response'] = float(np.max(vessels_masked))
                    features['vessel_variance'] = float(np.var(vessels_masked))
                    
                    # Detect branching patterns (arborizing characteristic)
                    vessel_binary = vessels > np.percentile(vessels, 95)
                    vessel_skeleton = medial_axis(vessel_binary)
                    features['vessel_branching_density'] = float(np.sum(vessel_skeleton) / np.sum(mask))
                else:
                    features['vessel_density'] = 0.0
                    features['vessel_max_response'] = 0.0
                    features['vessel_variance'] = 0.0
                    features['vessel_branching_density'] = 0.0
            except ImportError:
                # Fallback if frangi filter not available
                features['vessel_density'] = 0.0
                features['vessel_max_response'] = 0.0
                features['vessel_variance'] = 0.0
                features['vessel_branching_density'] = 0.0
            
            # 2. BLUE-GRAY STRUCTURES DETECTION (BCC characteristic)
            # Analyze blue channel intensity and gray-blue color patterns
            if len(rgb_image.shape) == 3:
                blue_channel = rgb_image[:,:,2]
                blue_masked = blue_channel[mask]
                
                # Detect blue-gray areas (high blue, moderate red/green)
                blue_threshold = np.percentile(blue_masked, 75) if len(blue_masked) > 0 else 0
                blue_dominant = (blue_channel > blue_threshold) & mask
                
                features['blue_gray_area_ratio'] = float(np.sum(blue_dominant) / np.sum(mask)) if np.sum(mask) > 0 else 0
                features['blue_intensity_mean'] = float(np.mean(blue_masked)) if len(blue_masked) > 0 else 0
                features['blue_intensity_std'] = float(np.std(blue_masked)) if len(blue_masked) > 0 else 0
            else:
                features['blue_gray_area_ratio'] = 0.0
                features['blue_intensity_mean'] = 0.0
                features['blue_intensity_std'] = 0.0
            
            # 3. SURFACE TEXTURE ANALYSIS (SK vs BCC differentiation)
            # Analyze surface roughness and "stuck-on" appearance
            try:
                # Calculate local standard deviation (roughness measure)
                from scipy import ndimage
                roughness = ndimage.generic_filter(gray, np.std, size=5)
                roughness_masked = roughness[mask]
                
                if len(roughness_masked) > 0:
                    features['surface_roughness_mean'] = float(np.mean(roughness_masked))
                    features['surface_roughness_std'] = float(np.std(roughness_masked))
                    features['surface_roughness_max'] = float(np.max(roughness_masked))
                    
                    # High roughness areas (warty/stuck-on appearance of SK)
                    high_roughness = roughness > np.percentile(roughness_masked, 80)
                    features['high_roughness_ratio'] = float(np.sum(high_roughness & mask) / np.sum(mask))
                else:
                    features['surface_roughness_mean'] = 0.0
                    features['surface_roughness_std'] = 0.0
                    features['surface_roughness_max'] = 0.0
                    features['high_roughness_ratio'] = 0.0
            except:
                features['surface_roughness_mean'] = 0.0
                features['surface_roughness_std'] = 0.0
                features['surface_roughness_max'] = 0.0
                features['high_roughness_ratio'] = 0.0
            
            # 4. PIGMENT PATTERN ANALYSIS
            # Analyze pigment distribution patterns (important for both BCC and SK)
            if len(hsv_image.shape) == 3:
                saturation = hsv_image[:,:,1]
                value = hsv_image[:,:,2]
                
                sat_masked = saturation[mask]
                val_masked = value[mask]
                
                if len(sat_masked) > 0:
                    # Pigment concentration analysis
                    features['pigment_saturation_mean'] = float(np.mean(sat_masked))
                    features['pigment_saturation_std'] = float(np.std(sat_masked))
                    features['pigment_value_mean'] = float(np.mean(val_masked))
                    features['pigment_value_std'] = float(np.std(val_masked))
                    
                    # Detect areas of high pigmentation
                    high_pigment = (saturation > np.percentile(sat_masked, 70)) & mask
                    features['high_pigment_ratio'] = float(np.sum(high_pigment) / np.sum(mask))
                    
                    # Pigment distribution uniformity
                    features['pigment_uniformity'] = float(1.0 / (1.0 + np.std(sat_masked)))
                else:
                    features['pigment_saturation_mean'] = 0.0
                    features['pigment_saturation_std'] = 0.0
                    features['pigment_value_mean'] = 0.0
                    features['pigment_value_std'] = 0.0
                    features['high_pigment_ratio'] = 0.0
                    features['pigment_uniformity'] = 0.0
            else:
                features['pigment_saturation_mean'] = 0.0
                features['pigment_saturation_std'] = 0.0
                features['pigment_value_mean'] = 0.0
                features['pigment_value_std'] = 0.0
                features['high_pigment_ratio'] = 0.0
                features['pigment_uniformity'] = 0.0
            
            # 5. TRANSLUCENCY ANALYSIS (BCC characteristic)
            # Analyze for translucent/pearly appearance
            if len(rgb_image.shape) == 3:
                # Calculate luminance
                luminance = 0.299 * rgb_image[:,:,0] + 0.587 * rgb_image[:,:,1] + 0.114 * rgb_image[:,:,2]
                lum_masked = luminance[mask]
                
                if len(lum_masked) > 0:
                    # High luminance with low saturation suggests translucency
                    high_luminance = luminance > np.percentile(lum_masked, 80)
                    low_saturation = hsv_image[:,:,1] < np.percentile(hsv_image[:,:,1][mask], 30)
                    translucent_areas = high_luminance & low_saturation & mask
                    
                    features['translucency_ratio'] = float(np.sum(translucent_areas) / np.sum(mask))
                    features['luminance_mean'] = float(np.mean(lum_masked))
                    features['luminance_std'] = float(np.std(lum_masked))
                else:
                    features['translucency_ratio'] = 0.0
                    features['luminance_mean'] = 0.0
                    features['luminance_std'] = 0.0
            else:
                features['translucency_ratio'] = 0.0
                features['luminance_mean'] = 0.0
                features['luminance_std'] = 0.0
            
            # 6. COMEDO-LIKE OPENINGS DETECTION (SK characteristic)
            # Detect small dark circular/oval structures
            try:
                # Use morphological operations to detect small dark spots
                dark_spots = gray < np.percentile(gray[mask], 20) if np.sum(mask) > 0 else np.zeros_like(gray, dtype=bool)
                
                # Remove small noise and keep only significant dark spots
                from skimage.morphology import opening, disk
                dark_spots_cleaned = opening(dark_spots, disk(2))
                
                # Count and characterize dark spots
                from skimage.measure import label, regionprops
                labeled_spots = label(dark_spots_cleaned)
                spot_props = regionprops(labeled_spots)
                
                features['comedo_count'] = len(spot_props)
                features['comedo_density'] = float(len(spot_props) / np.sum(mask)) if np.sum(mask) > 0 else 0
                
                if spot_props:
                    spot_areas = [prop.area for prop in spot_props]
                    features['comedo_mean_area'] = float(np.mean(spot_areas))
                    features['comedo_area_std'] = float(np.std(spot_areas))
                else:
                    features['comedo_mean_area'] = 0.0
                    features['comedo_area_std'] = 0.0
            except:
                features['comedo_count'] = 0
                features['comedo_density'] = 0.0
                features['comedo_mean_area'] = 0.0
                features['comedo_area_std'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting dermoscopic features: {str(e)}")
            return {}

 
    def extract_krawtchouk_moments(self, image, mask, max_order=4, p1=0.5, p2=0.5, normalize_size=256):
        """
        Extract classical Krawtchouk moments - CORRECTED VERSION.
        
        Based on standard definition from academic papers:
        Q_nm = Σ Σ f(x,y) * K_n(x; p1, N1) * K_m(y; p2, N2) * ρ(x; p1, N1) * ρ(y; p2, N2)
        
        where:
        K_n(x; p, N) = Σ_{k=0}^{n} a_k(n,p) * C(x,k) * C(N-x, n-k)
        a_k(n,p) = (-1)^k * C(n,k) * p^(n-k) * (1-p)^k
        ρ(x; p, N) = C(N,x) * p^x * (1-p)^(N-x)
        
        References:
        - Yap et al. (2003): "Image analysis by Krawtchouk moments"
        - Zhu et al. (2007): "Translation and scale invariants of Krawtchouk moments"
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
            
            if np.max(gray) > 1.0:
                gray = gray / 255.0
            
            # Validate mask
            if np.sum(mask) == 0:
                return self._get_default_krawtchouk_features(max_order)
            
            # Extract and crop lesion region
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return self._get_default_krawtchouk_features(max_order)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_region[~lesion_mask] = 0
            
            # Resize for efficiency
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
            
            N_y, N_x = lesion_region.shape
            
            # Pre-compute Krawtchouk polynomials WITH PROPER WEIGHTS
            K_x = np.zeros((max_order + 1, N_x), dtype=np.float64)
            K_y = np.zeros((max_order + 1, N_y), dtype=np.float64)
            
            # CRITICAL: Compute weighted polynomials K̃_n(x) = K_n(x) * sqrt(ρ(x))
            for n in range(max_order + 1):
                for x in range(N_x):
                    K_n = self._krawtchouk_polynomial(x, n, p1, N_x - 1)
                    sqrt_weight = np.sqrt(self._krawtchouk_weight(x, p1, N_x - 1))
                    K_x[n, x] = K_n * sqrt_weight
                
                for y in range(N_y):
                    K_m = self._krawtchouk_polynomial(y, n, p2, N_y - 1)
                    sqrt_weight = np.sqrt(self._krawtchouk_weight(y, p2, N_y - 1))
                    K_y[n, y] = K_m * sqrt_weight
            
            # Masked image
            mask_float = lesion_mask_resized.astype(np.float64)
            img_masked = lesion_region * mask_float
            
            # Compute moments (vectorized)
            for n in range(max_order + 1):
                for m in range(max_order + 1):
                    if n + m > max_order:
                        continue
                    
                    # Weighted basis function matrix: K̃_m(y) ⊗ K̃_n(x)
                    basis = np.outer(K_y[m, :], K_x[n, :])
                    
                    # Moment computation (already weighted)
                    moment = np.sum(img_masked * basis)
                    
                    features[f'krawtchouk_moment_{n}_{m}'] = float(moment)
                    features[f'krawtchouk_moment_abs_{n}_{m}'] = float(np.abs(moment))

            # Invariant features (Yap et al., 2003 - Translation & Scale Invariants)
            try:
                Q00 = features.get('krawtchouk_moment_0_0', 0.0)
                if Q00 > 1e-10:
                    # Scale-invariant moment: normalize by Q00
                    features['krawtchouk_invariant_1'] = float(
                        (features.get('krawtchouk_moment_abs_2_0', 0.0) + 
                        features.get('krawtchouk_moment_abs_0_2', 0.0)) / (Q00 ** 2)
                    )
                    
                    # Rotation-invariant moment
                    Q20 = features.get('krawtchouk_moment_2_0', 0.0)
                    Q02 = features.get('krawtchouk_moment_0_2', 0.0)
                    Q11 = features.get('krawtchouk_moment_1_1', 0.0)
                    
                    invariant_2 = ((Q20 - Q02) ** 2 + 4 * (Q11 ** 2)) / (Q00 ** 4)
                    features['krawtchouk_invariant_2'] = float(invariant_2)
                    
                    # Higher-order invariants
                    features['krawtchouk_invariant_3'] = float(
                        (features.get('krawtchouk_moment_abs_3_0', 0.0) + 
                        features.get('krawtchouk_moment_abs_1_2', 0.0)) / (Q00 ** 2.5)
                    )
                    features['krawtchouk_invariant_4'] = float(
                        (features.get('krawtchouk_moment_abs_0_3', 0.0) + 
                        features.get('krawtchouk_moment_abs_2_1', 0.0)) / (Q00 ** 2.5)
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

            # Energy and entropy
            try:
                moment_values = [features[f'krawtchouk_moment_abs_{n}_{m}']
                                for n in range(max_order + 1)
                                for m in range(max_order + 1)
                                if n + m <= max_order]

                mv = np.array(moment_values)
                features['krawtchouk_energy'] = float(np.sum(mv ** 2))

                s = np.sum(mv)
                if s > 1e-10:
                    probs = mv / s
                    probs = probs[probs > 1e-10]
                    features['krawtchouk_entropy'] = float(-np.sum(probs * np.log2(probs)))
                else:
                    features['krawtchouk_entropy'] = 0.0
            except Exception as e:
                self.logger.warning(f"Error computing Krawtchouk energy/entropy: {str(e)}")
                features['krawtchouk_energy'] = 0.0
                features['krawtchouk_entropy'] = 0.0            
        
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting Krawtchouk moments: {str(e)}")
            return self._get_default_krawtchouk_features(max_order)

    def _krawtchouk_polynomial(self, x, n, p, N):
        """
        Compute classical Krawtchouk polynomial - CORRECTED VERSION.
        
        K_n(x; p, N) = Σ_{k=0}^{n} a_k(n,p) * C(x,k) * C(N-x, n-k)
        
        where a_k(n,p) = (-1)^k * C(n,k) * p^(n-k) * (1-p)^k
        
        This is the STANDARD definition from academic papers.
        """
        if n > N or x > N or x < 0:
            return 0.0
        
        try:
            # Clip p to avoid numerical issues
            p = float(np.clip(p, 1e-12, 1 - 1e-12))
            
            result = 0.0
            for k in range(n + 1):
                # Check validity of binomial coefficients
                if x >= k and (N - x) >= (n - k):
                    # Compute binomial coefficients using log-space for stability
                    try:
                        log_binom_x = gammaln(x + 1) - gammaln(k + 1) - gammaln(x - k + 1)
                        log_binom_N = gammaln(N - x + 1) - gammaln(n - k + 1) - gammaln(N - x - n + k + 1)
                        log_binom_n = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                        
                        # Compute a_k
                        sign = (-1) ** k
                        log_a_k = log_binom_n + (n - k) * np.log(p) + k * np.log(1 - p)
                        
                        # Combine
                        log_term = log_binom_x + log_binom_N + log_a_k
                        
                        if np.isfinite(log_term):
                            result += sign * np.exp(log_term)
                        
                    except (ValueError, RuntimeWarning):
                        # Fallback to direct computation if log-space fails
                        binom_x = special.comb(x, k, exact=False)
                        binom_N = special.comb(N - x, n - k, exact=False)
                        binom_n = special.comb(n, k, exact=False)
                        
                        a_k = ((-1) ** k) * binom_n * (p ** (n - k)) * ((1 - p) ** k)
                        result += a_k * binom_x * binom_N
            
            return float(result)
        except Exception as e:
            self.logger.warning(f"Error in Krawtchouk polynomial: {str(e)}")
            return 0.0
        
    def _krawtchouk_weight(self, x, p, N):
        """
        Compute Krawtchouk weight function - CORRECTED VERSION.
        
        ρ(x; p, N) = C(N, x) * p^x * (1-p)^(N-x)
        
        This is the binomial distribution weight.
        """
        try:
            if x < 0 or x > N:
                return 0.0
            
            # Clip p to avoid numerical issues
            p = float(np.clip(p, 1e-12, 1 - 1e-12))
            
            # Use log-space for numerical stability
            log_weight = gammaln(N + 1) - gammaln(x + 1) - gammaln(N - x + 1) + \
                        x * np.log(p) + (N - x) * np.log(1 - p)
            
            if np.isfinite(log_weight):
                weight = np.exp(log_weight)
                return float(weight)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error in Krawtchouk weight: {str(e)}")
            return 0.0


    def extract_fourier_features(self, image, mask, num_coefficients=36):
        try:
            features = {}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy() if len(image.shape) == 2 else image[:,:,0]
            
            # Normalize to [0, 1]
            if np.max(gray) > 1.0:
                gray = gray / 255.0
            
            # Validate mask
            if np.sum(mask) == 0:
                self.logger.warning("Empty mask for Fourier features")
                return self._get_default_fourier_features(num_coefficients)
            
            # Extract lesion bounding box
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                return self._get_default_fourier_features(num_coefficients)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Crop to lesion region
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Apply mask
            lesion_region[~lesion_mask] = 0
            
            # Normalize size to 256 for fair comparison with Krawtchouk
            original_shape = lesion_region.shape
            max_dim = max(original_shape)
            normalize_size = 256
            
            scale = normalize_size / max_dim
            new_height = int(original_shape[0] * scale)
            new_width = int(original_shape[1] * scale)
            new_height = max(new_height, 64)
            new_width = max(new_width, 64)
            
            # Choose interpolation
            if scale > 1.0:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            
            lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                    interpolation=interpolation)
            lesion_mask = cv2.resize(lesion_mask.astype(np.uint8), 
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            
            # Compute 2D FFT
            fft = np.fft.fft2(lesion_region)
            fft_shifted = np.fft.fftshift(fft)  # Shift zero frequency to center
            
            # Compute magnitude and phase
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            # Extract low-frequency coefficients (most informative for shape/texture)
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            
            # Define region of interest (low frequencies near center)
            roi_size = min(20, center_y, center_x)  # 20x20 region around center
            
            y_start = center_y - roi_size
            y_end = center_y + roi_size
            x_start = center_x - roi_size
            x_end = center_x + roi_size
            
            magnitude_roi = magnitude[y_start:y_end, x_start:x_end]
            phase_roi = phase[y_start:y_end, x_start:x_end]
            
            # Flatten and extract top coefficients
            magnitude_flat = magnitude_roi.flatten()
            phase_flat = phase_roi.flatten()
            
            # Sort by magnitude (most significant coefficients)
            sorted_indices = np.argsort(magnitude_flat)[::-1]
            
            # Extract features: magnitude and phase of top coefficients
            num_features_per_type = num_coefficients // 2
            
            for i in range(min(num_features_per_type, len(sorted_indices))):
                idx = sorted_indices[i]
                features[f'fourier_magnitude_{i}'] = float(magnitude_flat[idx])
                features[f'fourier_phase_{i}'] = float(phase_flat[idx])
            
            # Fill remaining features if we have less than requested
            for i in range(len(sorted_indices), num_features_per_type):
                features[f'fourier_magnitude_{i}'] = 0.0
                features[f'fourier_phase_{i}'] = 0.0
            
            # Additional spectral features
            features['fourier_energy'] = float(np.sum(magnitude ** 2))
            features['fourier_entropy'] = float(-np.sum(
                (magnitude_flat / np.sum(magnitude_flat)) * 
                np.log2((magnitude_flat / np.sum(magnitude_flat)) + 1e-10)
            ))
            
            # Frequency distribution statistics
            features['fourier_mean_magnitude'] = float(np.mean(magnitude_roi))
            features['fourier_std_magnitude'] = float(np.std(magnitude_roi))
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting Fourier features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_fourier_features(num_coefficients)

    def _get_default_fourier_features(self, num_coefficients=36):
        """Return default Fourier features on error."""
        features = {}
        num_features_per_type = num_coefficients // 2
        for i in range(num_features_per_type):
            features[f'fourier_magnitude_{i}'] = 0.0
            features[f'fourier_phase_{i}'] = 0.0
        features['fourier_energy'] = 0.0
        features['fourier_entropy'] = 0.0
        features['fourier_mean_magnitude'] = 0.0
        features['fourier_std_magnitude'] = 0.0
        return features
    

    def extract_dct_features(self, image, mask, num_coefficients=36):

        try:
            features = {}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy() if len(image.shape) == 2 else image[:,:,0]
            
            # Normalize to [0, 255] for DCT (standard range)
            if np.max(gray) <= 1.0:
                gray = gray * 255.0
            
            # Validate mask
            if np.sum(mask) == 0:
                self.logger.warning("Empty mask for DCT features")
                return self._get_default_dct_features(num_coefficients)
            
            # Extract lesion bounding box
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                return self._get_default_dct_features(num_coefficients)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Crop to lesion region
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Apply mask
            lesion_region[~lesion_mask] = 0
            
            # Normalize size to 256 for fair comparison
            original_shape = lesion_region.shape
            max_dim = max(original_shape)
            normalize_size = 256
            
            scale = normalize_size / max_dim
            new_height = int(original_shape[0] * scale)
            new_width = int(original_shape[1] * scale)
            new_height = max(new_height, 64)
            new_width = max(new_width, 64)
            
            # Choose interpolation
            if scale > 1.0:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            
            lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                    interpolation=interpolation)
            lesion_mask = cv2.resize(lesion_mask.astype(np.uint8), 
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Apply 2D DCT using scipy
            from scipy.fftpack import dct
            
            # Compute 2D DCT (apply 1D DCT to rows, then columns)
            dct_2d = dct(dct(lesion_region.T, norm='ortho').T, norm='ortho')
            
            # Extract coefficients in zigzag order (standard for JPEG)
            # This prioritizes low-frequency components
            coefficients = []
            
            # Zigzag scan pattern
            h, w = dct_2d.shape
            
            # Generate zigzag indices
            for diag in range(h + w - 1):
                if diag % 2 == 0:  # Even diagonals go down-left
                    if diag < w:
                        x, y = 0, diag
                    else:
                        x, y = diag - w + 1, w - 1
                    while x < h and y >= 0:
                        coefficients.append(dct_2d[x, y])
                        x += 1
                        y -= 1
                        if len(coefficients) >= num_coefficients:
                            break
                else:  # Odd diagonals go up-right
                    if diag < h:
                        x, y = diag, 0
                    else:
                        x, y = h - 1, diag - h + 1
                    while x >= 0 and y < w:
                        coefficients.append(dct_2d[x, y])
                        x -= 1
                        y += 1
                        if len(coefficients) >= num_coefficients:
                            break
                
                if len(coefficients) >= num_coefficients:
                    break
            
            # Store DCT coefficients as features
            for i in range(min(num_coefficients, len(coefficients))):
                features[f'dct_coeff_{i}'] = float(coefficients[i])
            
            # Fill remaining if needed
            for i in range(len(coefficients), num_coefficients):
                features[f'dct_coeff_{i}'] = 0.0
            
            # Additional DCT statistics
            dct_abs = np.abs(dct_2d)
            features['dct_energy'] = float(np.sum(dct_2d ** 2))
            features['dct_mean'] = float(np.mean(dct_abs))
            features['dct_std'] = float(np.std(dct_abs))
            features['dct_max'] = float(np.max(dct_abs))
            
            # Energy compaction ratio (low-freq vs total energy)
            total_energy = np.sum(dct_2d ** 2)
            low_freq_size = min(h//4, w//4)
            low_freq_energy = np.sum(dct_2d[:low_freq_size, :low_freq_size] ** 2)
            features['dct_compaction_ratio'] = float(low_freq_energy / total_energy) if total_energy > 0 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting DCT features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_dct_features(num_coefficients)

    def _get_default_dct_features(self, num_coefficients=36):
        """Return default DCT features on error."""
        features = {}
        for i in range(num_coefficients):
            features[f'dct_coeff_{i}'] = 0.0
        features['dct_energy'] = 0.0
        features['dct_mean'] = 0.0
        features['dct_std'] = 0.0
        features['dct_max'] = 0.0
        features['dct_compaction_ratio'] = 0.0
        return features



    def extract_hadamard_features(self, image, mask, num_coefficients=36):

        try:
            features = {}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy() if len(image.shape) == 2 else image[:,:,0]
            
            # Normalize to [0, 1]
            if np.max(gray) > 1.0:
                gray = gray / 255.0
            
            # Validate mask
            if np.sum(mask) == 0:
                self.logger.warning("Empty mask for Hadamard features")
                return self._get_default_hadamard_features(num_coefficients)
            
            # Extract lesion bounding box
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                return self._get_default_hadamard_features(num_coefficients)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Crop to lesion region
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Apply mask
            lesion_region[~lesion_mask] = 0
            
            # Normalize size to 256 for fair comparison
            original_shape = lesion_region.shape
            max_dim = max(original_shape)
            normalize_size = 256
            
            scale = normalize_size / max_dim
            new_height = int(original_shape[0] * scale)
            new_width = int(original_shape[1] * scale)
            new_height = max(new_height, 64)
            new_width = max(new_width, 64)
            
            # Choose interpolation
            if scale > 1.0:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            
            lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                    interpolation=interpolation)
            lesion_mask = cv2.resize(lesion_mask.astype(np.uint8), 
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            
            # Pad to power of 2 for Hadamard transform
            h, w = lesion_region.shape
            next_pow2_h = 2 ** int(np.ceil(np.log2(h)))
            next_pow2_w = 2 ** int(np.ceil(np.log2(w)))
            
            padded = np.zeros((next_pow2_h, next_pow2_w))
            padded[:h, :w] = lesion_region
            
            # Compute 2D Hadamard transform
            hadamard_2d = self._hadamard_transform_2d(padded)
            
            # Extract most significant coefficients
            hadamard_abs = np.abs(hadamard_2d)
            
            # Flatten and sort by magnitude
            hadamard_flat = hadamard_abs.flatten()
            sorted_indices = np.argsort(hadamard_flat)[::-1]
            
            # Extract top coefficients
            for i in range(min(num_coefficients, len(sorted_indices))):
                idx = sorted_indices[i]
                features[f'hadamard_coeff_{i}'] = float(hadamard_2d.flatten()[idx])
            
            # Fill remaining if needed
            for i in range(len(sorted_indices), num_coefficients):
                features[f'hadamard_coeff_{i}'] = 0.0
            
            # Additional Hadamard statistics
            features['hadamard_energy'] = float(np.sum(hadamard_2d ** 2))
            features['hadamard_mean'] = float(np.mean(hadamard_abs))
            features['hadamard_std'] = float(np.std(hadamard_abs))
            features['hadamard_max'] = float(np.max(hadamard_abs))
            
            # Sequency-based features (Hadamard equivalent of frequency)
            # Low sequency = smooth patterns, high sequency = rapid changes
            low_seq_size = min(next_pow2_h//8, next_pow2_w//8)
            low_seq_energy = np.sum(hadamard_2d[:low_seq_size, :low_seq_size] ** 2)
            total_energy = np.sum(hadamard_2d ** 2)
            features['hadamard_low_seq_ratio'] = float(low_seq_energy / total_energy) if total_energy > 0 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting Hadamard features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_hadamard_features(num_coefficients)

    def _hadamard_transform_2d(self, matrix):

        n = matrix.shape[0]
        
        # Apply 1D transform to each row
        result = np.zeros_like(matrix)
        for i in range(n):
            result[i, :] = self._hadamard_transform_1d(matrix[i, :])
        
        # Apply 1D transform to each column
        for j in range(n):
            result[:, j] = self._hadamard_transform_1d(result[:, j])
        
        # Normalize
        return result / n

    def _hadamard_transform_1d(self, vector):

        n = len(vector)
        result = vector.copy()
        
        # Butterfly algorithm
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = result[j]
                    y = result[j + h]
                    result[j] = x + y
                    result[j + h] = x - y
            h *= 2
        
        return result

    def _get_default_hadamard_features(self, num_coefficients=36):
        """Return default Hadamard features on error."""
        features = {}
        for i in range(num_coefficients):
            features[f'hadamard_coeff_{i}'] = 0.0
        features['hadamard_energy'] = 0.0
        features['hadamard_mean'] = 0.0
        features['hadamard_std'] = 0.0
        features['hadamard_max'] = 0.0
        features['hadamard_low_seq_ratio'] = 0.0
        return features
    

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
        
    def extract_surface_pattern_features(self, image, mask):
        """Extract surface pattern features specific to BCC vs SK differentiation."""
        try:
            features = {}
            
            # Convert to grayscale for pattern analysis
            if len(image.shape) == 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                
            # 1. SURFACE ROUGHNESS (SK tends to be rougher - "stuck-on" appearance)
            # Local standard deviation as roughness measure
            roughness = ndimage.generic_filter(gray, np.std, size=5)
            roughness_masked = roughness[mask]
            
            if len(roughness_masked) > 0:
                features['surface_roughness_mean'] = float(np.mean(roughness_masked))
                features['surface_roughness_std'] = float(np.std(roughness_masked))
                features['surface_roughness_percentile_90'] = float(np.percentile(roughness_masked, 90))
                
                # Percentage of high-roughness areas
                high_roughness_threshold = np.percentile(roughness_masked, 75)
                high_roughness_ratio = np.sum(roughness > high_roughness_threshold) / np.sum(mask)
                features['high_roughness_area_ratio'] = float(high_roughness_ratio)
            
            # 2. SMOOTHNESS PATTERNS (BCC tends to be smoother)
            # Calculate local gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            grad_masked = gradient_magnitude[mask]
            if len(grad_masked) > 0:
                features['gradient_smoothness'] = float(1.0 / (1.0 + np.mean(grad_masked)))
                features['gradient_variation'] = float(np.std(grad_masked))
            
            # 3. PATTERN REGULARITY
            # Use autocorrelation to detect regular patterns
            try:
                # Create a small region around center for autocorrelation
                center_y, center_x = ndimage.center_of_mass(mask)
                size = min(50, mask.shape[0]//3, mask.shape[1]//3)
                
                y_start = max(0, int(center_y - size//2))
                y_end = min(mask.shape[0], int(center_y + size//2))
                x_start = max(0, int(center_x - size//2))
                x_end = min(mask.shape[1], int(center_x + size//2))
                
                region = gray[y_start:y_end, x_start:x_end]
                region_mask = mask[y_start:y_end, x_start:x_end]
                
                if np.sum(region_mask) > 100:  # Need sufficient area
                    # Normalize region
                    region_normalized = (region - np.mean(region)) / (np.std(region) + 1e-10)
                    
                    # Calculate 2D autocorrelation
                    autocorr = cv2.matchTemplate(region_normalized, region_normalized, cv2.TM_CCORR_NORMED)
                    
                    # Pattern regularity is measured by secondary peaks in autocorrelation
                    if autocorr.size > 1:
                        # Remove center peak
                        center_autocorr = autocorr.shape[0]//2, autocorr.shape[1]//2
                        autocorr_copy = autocorr.copy()
                        autocorr_copy[center_autocorr] = 0
                        
                        features['pattern_regularity'] = float(np.max(autocorr_copy))
                    else:
                        features['pattern_regularity'] = 0.0
                else:
                    features['pattern_regularity'] = 0.0
            except:
                features['pattern_regularity'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in surface pattern features: {str(e)}")
            return {}
        
    def extract_multiscale_texture_features(self, image, mask):
        """Extract multi-scale texture features optimized for BCC vs SK."""
        try:
            features = {}
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
            
            # Multi-scale analysis at different window sizes
            scales = [3, 5, 7, 9, 11]
            
            for scale in scales:
                # Local variance (texture measure)
                local_var = ndimage.generic_filter(gray, np.var, size=scale)
                var_masked = local_var[mask]
                
                if len(var_masked) > 0:
                    features[f'texture_variance_scale_{scale}'] = float(np.mean(var_masked))
                    features[f'texture_variance_std_scale_{scale}'] = float(np.std(var_masked))
                
                # Local entropy
                def local_entropy(region):
                    if len(region) == 0:
                        return 0
                    hist, _ = np.histogram(region, bins=8, range=(0, 1))
                    hist = hist[hist > 0]
                    return -np.sum((hist/np.sum(hist)) * np.log2(hist/np.sum(hist))) if len(hist) > 0 else 0
                
                local_ent = ndimage.generic_filter(gray, local_entropy, size=scale)
                ent_masked = local_ent[mask]
                
                if len(ent_masked) > 0:
                    features[f'texture_entropy_scale_{scale}'] = float(np.mean(ent_masked))
            
            # Texture directionality (important for vessel patterns in BCC)
            try:
                # Calculate gradient orientation
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                orientation = np.arctan2(grad_y, grad_x)
                
                # Calculate orientation histogram
                orientation_masked = orientation[mask]
                if len(orientation_masked) > 0:
                    hist, _ = np.histogram(orientation_masked, bins=8, range=(-np.pi, np.pi))
                    hist_normalized = hist / np.sum(hist)
                    
                    # Directionality is measured by concentration in orientation histogram
                    features['texture_directionality'] = float(1.0 - stats.entropy(hist_normalized + 1e-10))
                    features['dominant_orientation_strength'] = float(np.max(hist_normalized))
            except:
                features['texture_directionality'] = 0.0
                features['dominant_orientation_strength'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in multiscale texture features: {str(e)}")
            return {}
        
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
            



    def _log_binom(self, n, k):
        k = np.asarray(k, dtype=float)
        logv = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        invalid = (k < 0) | (k > n) | (np.floor(k) != k)
        logv = np.where(invalid, -np.inf, logv)
        return float(logv) if logv.ndim == 0 else logv

    def _sqrt_weight_vec(self, N, p):
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        xs = np.arange(N + 1, dtype=float)
        lw = self._log_binom(N, xs) + xs*np.log(p) + (N - xs)*np.log(1 - p)
        m = np.max(lw)
        sw = np.exp(0.5 * (lw - m))
        return sw, m

    def _hyp2f1_terminating_negints(self, x, alpha, N, z):
        t = 1.0
        s = 1.0
        c = 0.0
        for k in range(0, x):
            num = (-x + k) * (-alpha + k)
            den = (-N + k) * (k + 1.0)
            if den == 0.0:
                break
            t *= (num / den) * z
            y = t - c
            tmp = s + y
            c = (tmp - s) - y
            s = tmp
            if not np.isfinite(s):
                s *= 1e-300
                t *= 1e-300
                c *= 1e-300
        return float(s)

    def _modified_fractional_krawtchouk_polynomial(self, x, alpha, p, N):
        if x < 0 or x > N:
            return 0.0
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        z = 1.0 / p
        
        # Compute hypergeometric function using stable series
        F = self._hyp2f1_terminating_negints(x, alpha, N, z)
        
        # Add the missing Gamma normalization for fractional orders
        # normalization = sqrt[Γ(α+1) · Γ(N-α+1) / Γ(N+1)]
        try:
            # Use log-gamma to avoid overflow, then exp
            log_norm = 0.5 * (gammaln(alpha + 1) + gammaln(N - alpha + 1) - gammaln(N + 1))
            normalization = np.exp(log_norm)
            
            # Check for overflow/underflow
            if not np.isfinite(normalization):
                normalization = 1.0
        except:
            # Fallback if gamma computation fails
            normalization = 1.0
        
        return normalization * F

    def modified_extract_fractional_krawtchouk_moments(self, image, mask,
                                            alpha_orders=None, beta_orders=None,
                                            p1=0.5, p2=0.5, normalize_size=32):
        try:
            features = {}
            if alpha_orders is None:
                alpha_orders = [0.5, 1.0, 1.5, 2.0]
            if beta_orders is None:
                beta_orders = [0.5, 1.0, 1.5, 2.0]
            if image.ndim == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:, :, :3])
            else:
                gray = image[..., 0] if image.ndim == 3 else image.astype(float)
            gray = gray.astype(float)
            if gray.max() > 1.0:
                gray = gray / 255.0
            if mask is None or np.sum(mask) == 0:
                self.logger.warning("FKM: empty mask; returning zeros.")
                for a in alpha_orders:
                    for b in beta_orders:
                        features[f'frkm_{a}_{b}'] = 0.0
                        features[f'frkm_abs_{a}_{b}'] = 0.0
                features.update(dict(frkm_energy=0.0, frkm_entropy=0.0,
                                    frkm_sparsity=0.0,
                                    frkm_low_freq_energy=0.0,
                                    frkm_high_freq_energy=0.0,
                                    frkm_freq_ratio=0.0))
                return features
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            roi = gray[y0:y1+1, x0:x1+1].copy()
            roi_mask = mask[y0:y1+1, x0:x1+1].astype(bool).copy()
            H, W = roi.shape
            if max(H, W) > normalize_size:
                scale = normalize_size / max(H, W)
                nh, nw = max(1, int(round(H*scale))), max(1, int(round(W*scale)))
                roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
                roi_mask = cv2.resize(roi_mask.astype(np.uint8), (nw, nh),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            N1, N2 = roi.shape
            if N1 == 0 or N2 == 0 or not roi_mask.any():
                self.logger.warning("FKM: invalid ROI; returning zeros.")
                for a in alpha_orders:
                    for b in beta_orders:
                        features[f'frkm_{a}_{b}'] = 0.0
                        features[f'frkm_abs_{a}_{b}'] = 0.0
                features.update(dict(frkm_energy=0.0, frkm_entropy=0.0,
                                    frkm_sparsity=0.0,
                                    frkm_low_freq_energy=0.0,
                                    frkm_high_freq_energy=0.0,
                                    frkm_freq_ratio=0.0))
                return features
            sw_x, _ = self._sqrt_weight_vec(N2 - 1, float(np.clip(p1, 1e-12, 1 - 1e-12)))
            sw_y, _ = self._sqrt_weight_vec(N1 - 1, float(np.clip(p2, 1e-12, 1 - 1e-12)))
            K_alpha = np.zeros((len(alpha_orders), N2), dtype=float)
            for i, a in enumerate(alpha_orders):
                for x in range(N2):
                    K_alpha[i, x] = self._modified_fractional_krawtchouk_polynomial(x, a, p1, N2 - 1)
                K_alpha[i, :] *= sw_x
            K_beta = np.zeros((len(beta_orders), N1), dtype=float)
            for j, b in enumerate(beta_orders):
                for y in range(N1):
                    K_beta[j, y] = self._modified_fractional_krawtchouk_polynomial(y, b, p2, N1 - 1)
                K_beta[j, :] *= sw_y
            area = float(roi_mask.sum())
            moment_matrix = np.zeros((len(alpha_orders), len(beta_orders)), dtype=float)
            roi_z = np.where(roi_mask, roi, 0.0)
            for i in range(len(alpha_orders)):
                for j in range(len(beta_orders)):
                    tmp = roi_z.dot(K_alpha[i, :])
                    moment = float(np.dot(K_beta[j, :], tmp))
                    moment = moment / area if area > 0 else 0.0
                    if not np.isfinite(moment):
                        self.logger.warning(f"FKM: non-finite moment for alpha={alpha_orders[i]}, beta={beta_orders[j]} → fallback to 0.")
                        moment = 0.0
                    moment_matrix[i, j] = moment
                    features[f'frkm_{alpha_orders[i]}_{beta_orders[j]}'] = moment
                    features[f'frkm_abs_{alpha_orders[i]}_{beta_orders[j]}'] = abs(moment)
            mv = np.abs(moment_matrix).ravel()
            mv = mv[np.isfinite(mv)]
            if mv.size == 0:
                self.logger.warning("FKM: empty/NaN moment vector; statistical features set to 0.")
                features.update(dict(frkm_energy=0.0, frkm_entropy=0.0,
                                    frkm_sparsity=0.0,
                                    frkm_low_freq_energy=0.0,
                                    frkm_high_freq_energy=0.0,
                                    frkm_freq_ratio=0.0))
            else:
                energy = float(np.sum(mv**2))
                s = float(mv.sum())
                if s > 0:
                    probs = mv / s
                    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
                else:
                    self.logger.warning("FKM: zero sum of moments; entropy set to 0.")
                    entropy = 0.0
                sparsity = float(np.sum(mv > mv.mean()) / mv.size)
                low = mv[: mv.size//3]
                high = mv[2*mv.size//3 :]
                lf = float(np.sum(low**2)) if low.size else 0.0
                hf = float(np.sum(high**2)) if high.size else 0.0
                freq_ratio = float(lf / (hf + 1e-12)) if (lf + hf) > 0 else 0.0
                features.update(dict(frkm_energy=energy,
                                    frkm_entropy=entropy,
                                    frkm_sparsity=sparsity,
                                    frkm_low_freq_energy=lf,
                                    frkm_high_freq_energy=hf,
                                    frkm_freq_ratio=freq_ratio))
            return features
        except Exception as e:
            self.logger.error(f"FKM error: {e}", exc_info=True)
            out = {}
            for a in (alpha_orders or [0.5, 1.0]):
                for b in (beta_orders or [0.5, 1.0]):
                    self.logger.warning(f"FKM: fallback zero for alpha={a}, beta={b} due to exception.")
                    out[f'frkm_{a}_{b}'] = 0.0
                    out[f'frkm_abs_{a}_{b}'] = 0.0
            out.update(dict(frkm_energy=0.0, frkm_entropy=0.0,
                            frkm_sparsity=0.0,
                            frkm_low_freq_energy=0.0,
                            frkm_high_freq_energy=0.0,
                            frkm_freq_ratio=0.0))
            return out



    def extract_fractional_krawtchouk_moments(self, image, mask, max_order=3, p1=0.5, p2=0.5, 
                                            alpha=0.5, beta=0.5, normalize_size=128):
        """
        Extract Fractional Krawtchouk Moments using the standard academic definition.
        
        Based on the mathematical formulation:
        FKM_αβ(n,m) = Σ Σ f(x,y) * K̃_n^α(x; p1, N1) * K̃_m^β(y; p2, N2) * w(x; p1, N1) * w(y; p2, N2)
        
        where K̃_n^α is the fractional Krawtchouk polynomial:
        K̃_n^α(x; p, N) = (p^(-x)) * 2F1(-x, -α; -N; 1/p)
        
        and w(x; p, N) is the weight function:
        w(x; p, N) = C(N, x) * p^x * (1-p)^(N-x)
        
        Parameters commonly used in papers:
        - p1, p2 = 0.5 (symmetric)
        - alpha, beta = 0.5, 1.0, 1.5, 2.0, 2.5 (fractional orders)
        - max_order = 3 or 4
        
        References:
        - Xiao et al. (2010): "Image analysis by fractional-order orthogonal moments"
        - Qi et al. (2019): "A novel fractional-order Krawtchouk moment for pattern recognition"
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
            
            if np.max(gray) > 1.0:
                gray = gray / 255.0
            
            # Validate mask
            if np.sum(mask) == 0:
                return self._get_default_fkm_features(max_order, alpha, beta)
            
            # Extract and normalize lesion region
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return self._get_default_fkm_features(max_order, alpha, beta)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_region[~lesion_mask] = 0
            
            # Resize to normalize_size (common in papers: 64x64 or 128x128)
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
            
            N_y, N_x = lesion_region.shape
            
            # Pre-compute fractional Krawtchouk polynomials using scipy.special.hyp2f1
            from scipy.special import hyp2f1, comb
            
            # Compute polynomials for x-direction (columns)
            K_x = np.zeros((max_order + 1, N_x), dtype=np.float64)
            for n in range(max_order + 1):
                for x in range(N_x):
                    # Fractional Krawtchouk polynomial: K_n^α(x; p, N) = p^(-x) * 2F1(-x, -α; -N; 1/p)
                    frac_order = n + alpha
                    K_x[n, x] = self._fkm_polynomial(x, frac_order, p1, N_x - 1)
            
            # Compute polynomials for y-direction (rows)
            K_y = np.zeros((max_order + 1, N_y), dtype=np.float64)
            for m in range(max_order + 1):
                for y in range(N_y):
                    frac_order = m + beta
                    K_y[m, y] = self._fkm_polynomial(y, frac_order, p2, N_y - 1)
            
            # Compute weight functions
            w_x = np.array([self._krawtchouk_weight(x, p1, N_x - 1) for x in range(N_x)])
            w_y = np.array([self._krawtchouk_weight(y, p2, N_y - 1) for y in range(N_y)])
            
            # Normalize weights to avoid numerical issues
            w_x = w_x / np.sum(w_x) if np.sum(w_x) > 0 else w_x
            w_y = w_y / np.sum(w_y) if np.sum(w_y) > 0 else w_y
            
            # Weight matrix
            W = np.outer(w_y, w_x)
            
            # Masked image and weights
            mask_float = lesion_mask_resized.astype(np.float64)
            img_masked = lesion_region * mask_float
            
            # Compute moments
            for n in range(max_order + 1):
                for m in range(max_order + 1):
                    if n + m > max_order:
                        continue
                    
                    # Basis function matrix with weight
                    basis = np.outer(K_y[m, :], K_x[n, :])
                    
                    # Fractional Krawtchouk moment
                    moment = np.sum(img_masked * basis * W)
                    
                    # Store moments
                    features[f'fkm_{n}_{m}'] = float(moment)
                    features[f'fkm_abs_{n}_{m}'] = float(np.abs(moment))
            
            # Compute moment invariants (rotation invariant features)
            features.update(self._compute_fkm_invariants(features, max_order))
            
            # Statistical features from moment magnitudes
            moment_values = [features[f'fkm_abs_{n}_{m}']
                            for n in range(max_order + 1)
                            for m in range(max_order + 1)
                            if n + m <= max_order]
            
            mv = np.array(moment_values)
            features['fkm_energy'] = float(np.sum(mv ** 2))
            
            s = np.sum(mv)
            if s > 1e-10:
                probs = mv / s
                probs = probs[probs > 1e-10]
                features['fkm_entropy'] = float(-np.sum(probs * np.log2(probs)))
            else:
                features['fkm_entropy'] = 0.0
            
            # Frequency domain features
            n_moments = len(mv)
            low_freq_moments = mv[:n_moments//3]
            high_freq_moments = mv[2*n_moments//3:]
            
            features['fkm_low_freq_energy'] = float(np.sum(low_freq_moments ** 2))
            features['fkm_high_freq_energy'] = float(np.sum(high_freq_moments ** 2))
            features['fkm_freq_ratio'] = float(
                features['fkm_low_freq_energy'] / (features['fkm_high_freq_energy'] + 1e-10)
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting fractional Krawtchouk moments: {str(e)}")
            return self._get_default_fkm_features(max_order, alpha, beta)

    def _fkm_polynomial(self, x, alpha, p, N):
        """
        Compute fractional Krawtchouk polynomial using hypergeometric function.
        
        K_n^α(x; p, N) = (p^(-x)) * 2F1(-x, -α; -N; 1/p)
        
        where 2F1 is the Gauss hypergeometric function.
        """
        from scipy.special import hyp2f1
        
        if x < 0 or x > N:
            return 0.0
        
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        
        try:
            # Hypergeometric function: 2F1(a, b; c; z)
            # For Krawtchouk: 2F1(-x, -α; -N; 1/p)
            z = 1.0 / p
            
            # Using scipy's hyp2f1
            hypergeom = hyp2f1(-x, -alpha, -N, z)
            
            # Complete polynomial
            polynomial = (p ** (-x)) * hypergeom
            
            # Normalization factor (optional, based on paper - some use it, some don't)
            # norm_factor = np.sqrt(special.comb(N, x, exact=False) * (p ** x) * ((1 - p) ** (N - x)))
            # polynomial *= norm_factor
            
            return float(polynomial)
            
        except Exception as e:
            self.logger.warning(f"Error computing FKM polynomial: {str(e)}")
            return 0.0

    def _compute_fkm_invariants(self, features, max_order):
        """
        Compute rotation and scale invariant features from FKM moments.
        
        Based on: Qi et al. (2019) - using magnitude and phase of complex moments.
        """
        invariants = {}
        
        try:
            # Geometric invariants (similar to Hu moments but using FKM)
            M00 = features.get('fkm_0_0', 0.0)
            
            if M00 > 1e-10:
                # Scale invariant: normalize by M00
                invariants['fkm_scale_inv_1'] = float(
                    (features.get('fkm_abs_2_0', 0.0) + features.get('fkm_abs_0_2', 0.0)) / (M00 ** 2)
                )
                
                # Rotation invariant
                M20 = features.get('fkm_2_0', 0.0)
                M02 = features.get('fkm_0_2', 0.0)
                M11 = features.get('fkm_1_1', 0.0)
                
                invariants['fkm_rotation_inv_1'] = float(
                    ((M20 - M02) ** 2 + 4 * (M11 ** 2)) / (M00 ** 4)
                )
                
                # Higher order invariants
                if max_order >= 3:
                    invariants['fkm_scale_inv_2'] = float(
                        (features.get('fkm_abs_3_0', 0.0) + 
                        features.get('fkm_abs_1_2', 0.0)) / (M00 ** 2.5)
                    )
                    invariants['fkm_scale_inv_3'] = float(
                        (features.get('fkm_abs_0_3', 0.0) + 
                        features.get('fkm_abs_2_1', 0.0)) / (M00 ** 2.5)
                    )
            else:
                invariants['fkm_scale_inv_1'] = 0.0
                invariants['fkm_rotation_inv_1'] = 0.0
                if max_order >= 3:
                    invariants['fkm_scale_inv_2'] = 0.0
                    invariants['fkm_scale_inv_3'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error computing FKM invariants: {str(e)}")
            invariants['fkm_scale_inv_1'] = 0.0
            invariants['fkm_rotation_inv_1'] = 0.0
            if max_order >= 3:
                invariants['fkm_scale_inv_2'] = 0.0
                invariants['fkm_scale_inv_3'] = 0.0
        
        return invariants

    def _get_default_fkm_features(self, max_order=3, alpha=0.5, beta=0.5):
        """Return default FKM features on error."""
        features = {}
        
        for n in range(max_order + 1):
            for m in range(max_order + 1):
                if n + m <= max_order:
                    features[f'fkm_{n}_{m}'] = 0.0
                    features[f'fkm_abs_{n}_{m}'] = 0.0
        
        features['fkm_scale_inv_1'] = 0.0
        features['fkm_rotation_inv_1'] = 0.0
        if max_order >= 3:
            features['fkm_scale_inv_2'] = 0.0
            features['fkm_scale_inv_3'] = 0.0
        
        features['fkm_energy'] = 0.0
        features['fkm_entropy'] = 0.0
        features['fkm_low_freq_energy'] = 0.0
        features['fkm_high_freq_energy'] = 0.0
        features['fkm_freq_ratio'] = 0.0
        
        return features
    


    def extract_paper_fractional_krawtchouk_moments(self, image, mask,
                                                            alpha_orders=None,
                                                            beta_orders=None,
                                                            max_n=3, max_m=3,
                                                            p1=0.5, p2=0.5,
                                                            normalize_size=256):
        """
        Extract Fractional Krawtchouk Moments following the paper EXACTLY.
        
        Based on Multiparameter Discrete Fractional Krawtchouk Transform (MDFRKT).
        
        Key formulas from paper:
        1. Classical weighted polynomial (Eq. 4):
        φ_n(i) = 2F1(-n, -i; -N+1; 1/p) × sqrt[numerator/denominator]
        
        2. Classical coefficient (Eq. 7):
        C_nm = Σ Σ f(x,y) φ_n(x) φ_m(y)
        
        3. Fractional eigenvalue (Eq. 13):
        λ_n(α) = exp(j * π * α * n)  ← depends on index n!
        
        4. Fractional moment (Eq. 14):
        F_nm(α,β) = λ_n(α) * λ_m(β) * C_nm
        
        Parameters:
        -----------
        alpha_orders : list, e.g., [0.5, 1.0, 1.5, 2.0]
            Fractional orders for x-direction
        beta_orders : list, e.g., [0.5, 1.0, 1.5, 2.0]
            Fractional orders for y-direction
        max_n, max_m : int
            Maximum classical basis indices (typically 3-4)
        """
        from scipy.special import hyp2f1
        
        try:
            features = {}
            
            if alpha_orders is None:
                alpha_orders = [0.5, 1.0, 1.5, 2.0]
            if beta_orders is None:
                beta_orders = [0.5, 1.0, 1.5, 2.0]
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = color.rgb2gray(image[:,:,:3])
            else:
                gray = image.copy()
                if len(gray.shape) == 3:
                    gray = gray[:,:,0]
            
            if np.max(gray) > 1.0:
                gray = gray / 255.0
            
            if np.sum(mask) == 0:
                return self._get_default_paper_fkm_features(alpha_orders, beta_orders)
            
            # Extract lesion region
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return self._get_default_paper_fkm_features(alpha_orders, beta_orders)
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            lesion_region[~lesion_mask] = 0
            
            # Normalize size
            if max(lesion_region.shape) > normalize_size:
                scale = normalize_size / max(lesion_region.shape)
                new_height = int(lesion_region.shape[0] * scale)
                new_width = int(lesion_region.shape[1] * scale)
                lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                        interpolation=cv2.INTER_LINEAR)
                lesion_mask = cv2.resize(lesion_mask.astype(np.uint8), 
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            N_y, N_x = lesion_region.shape
            
            # ========================================
            # STEP 1: Compute CLASSICAL weighted polynomials φ_n(i)
            # Using Equation (4) from paper
            # ========================================
            
            phi_x = np.zeros((max_n + 1, N_x), dtype=np.float64)
            for n in range(max_n + 1):
                for x in range(N_x):
                    phi_x[n, x] = self._compute_weighted_polynomial_eq4(x, n, p1, N_x - 1)
            
            phi_y = np.zeros((max_m + 1, N_y), dtype=np.float64)
            for m in range(max_m + 1):
                for y in range(N_y):
                    phi_y[m, y] = self._compute_weighted_polynomial_eq4(y, m, p2, N_y - 1)
            
            # ========================================
            # STEP 2: Compute CLASSICAL coefficients C_nm
            # Using Equation (7): C_nm = Σ Σ f(x,y) φ_n(x) φ_m(y)
            # NO AREA NORMALIZATION (basis is orthonormal)
            # ========================================
            
            mask_float = lesion_mask.astype(np.float64)
            img_masked = lesion_region * mask_float
            
            C_classical = np.zeros((max_n + 1, max_m + 1), dtype=np.float64)
            
            for n in range(max_n + 1):
                for m in range(max_m + 1):
                    # Basis function: φ_m(y) ⊗ φ_n(x)
                    basis = np.outer(phi_y[m, :], phi_x[n, :])
                    
                    # Classical coefficient (NO DIVISION BY AREA!)
                    C_classical[n, m] = np.sum(img_masked * basis)
            
            # ========================================
            # STEP 3: Compute FRACTIONAL eigenvalues
            # Using Equation (13): λ_n(α) = exp(j * π * α * n)
            # ========================================
            
            # For each (α, β) pair, compute fractional moments
            for alpha in alpha_orders:
                for beta in beta_orders:
                    """
                    # Compute eigenvalues for ALL basis indices -1 roots
                    # λ_n(α) = exp(j * α * 2π * n / Nn)   for n = 0,1,...,max_n ()
                    lambda_n_alpha = np.array([
                        np.exp(1j * alpha * (2 * np.pi * n) / (max_n + 1))
                        for n in range(max_n + 1)
                    ])

                    # λ_m(β) = exp(j * β * 2π * m / Nm)   for m = 0,1,...,max_m
                    lambda_m_beta = np.array([
                        np.exp(1j * beta * (2 * np.pi * m) / (max_m + 1))
                        for m in range(max_m + 1)
                    ])

                    """
                    # Compute eigenvalues for ALL basis indices 1 roots

                    # Compute eigenvalues for ALL basis indices
                    # λ_n(α) = exp(j * α * 2π * n / Nn)   for n = 0,1,...,max_n
                    lambda_n_alpha = np.array([
                        np.exp(1j * alpha * (2 * np.pi * n) / (max_n + 1))
                        for n in range(max_n + 1)
                    ])

                    # λ_m(β) = exp(j * β * 2π * m / Nm)   for m = 0,1,...,max_m
                    lambda_m_beta = np.array([
                        np.exp(1j * beta * (2 * np.pi * m) / (max_m + 1))
                        for m in range(max_m + 1)
                    ])

                    #lamda = 1 / k , alpha = 1, p = [0.1.......0.9], ps: Don't use more than one alpha


                    """
                    # Compute eigenvalues for ALL basis indices
                    # λ_n(α) = exp(j * π * α * n) for n = 0, 1, 2, ..., max_n
                    lambda_n_alpha = np.array([np.exp(1j * np.pi * alpha * n) 
                                            for n in range(max_n + 1)])
                    
                    # λ_m(β) = exp(j * π * β * m) for m = 0, 1, 2, ..., max_m
                    lambda_m_beta = np.array([np.exp(1j * np.pi * beta * m) 
                                            for m in range(max_m + 1)])
                    
                    """
                    # ========================================
                    # STEP 4: Apply MDFRKT formula (Equation 14)
                    # Fractional_C_nm(α,β) = λ_n(α) * λ_m(β) * C_nm
                    # ========================================
                    
                    # Compute fractional moment by weighted sum
                    moment_fractional = 0.0 + 0.0j
                    
                    for n in range(max_n + 1):
                        for m in range(max_m + 1):
                            # Apply eigenvalues to classical coefficient
                            moment_fractional += lambda_n_alpha[n] * lambda_m_beta[m] * C_classical[n, m]
                    
                    # Store complex features
                    features[f'paper_fkm_{alpha}_{beta}_real'] = float(np.real(moment_fractional))
                    features[f'paper_fkm_{alpha}_{beta}_imag'] = float(np.imag(moment_fractional))
                    features[f'paper_fkm_{alpha}_{beta}_magnitude'] = float(np.abs(moment_fractional))
                    features[f'paper_fkm_{alpha}_{beta}_phase'] = float(np.angle(moment_fractional))
            
            # Statistical features
            magnitudes = [features[f'paper_fkm_{a}_{b}_magnitude'] 
                        for a in alpha_orders for b in beta_orders]
            mv = np.array(magnitudes)
            
            features['paper_fkm_energy'] = float(np.sum(mv ** 2))
            s = np.sum(mv)
            if s > 1e-10:
                probs = mv / s
                features['paper_fkm_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
            else:
                features['paper_fkm_entropy'] = 0.0


            # ========================================
            # STEP 5: Compute INVARIANTS from fractional moments
            # Based on magnitude/phase invariance properties
            # ========================================
            
            # Magnitude-based invariants (rotation invariant)
            try:
                # Get fundamental moments for invariant computation
                m_00 = features.get('paper_fkm_0.5_0.5_magnitude', 0.0)
                
                if m_00 > 1e-10:
                    # Scale invariant feature (normalized by M00)
                    m_10 = features.get('paper_fkm_1.0_0.5_magnitude', 0.0)
                    m_01 = features.get('paper_fkm_0.5_1.0_magnitude', 0.0)
                    features['paper_fkm_scale_inv_1'] = float((m_10 + m_01) / (m_00 ** 1.5))
                    
                    # Rotation invariant (using phase difference)
                    phase_10 = features.get('paper_fkm_1.0_0.5_phase', 0.0)
                    phase_01 = features.get('paper_fkm_0.5_1.0_phase', 0.0)
                    features['paper_fkm_rotation_inv_1'] = float(np.abs(phase_10 - phase_01))
                    
                    # Higher order scale invariants
                    m_15 = features.get('paper_fkm_1.5_0.5_magnitude', 0.0)
                    m_20 = features.get('paper_fkm_2.0_0.5_magnitude', 0.0)
                    features['paper_fkm_scale_inv_2'] = float((m_15 + m_20) / (m_00 ** 2.0))
                    
                    # Cross-order invariant (characteristic of BCC vessel patterns)
                    m_11 = features.get('paper_fkm_1.0_1.0_magnitude', 0.0)
                    m_15_15 = features.get('paper_fkm_1.5_1.5_magnitude', 0.0)
                    features['paper_fkm_cross_order_inv'] = float((m_11 * m_15_15) / (m_00 ** 2.5))
                    
                    # Phase coherence (uniform phase = structured pattern)
                    phases = [features.get(f'paper_fkm_{a}_{b}_phase', 0.0) 
                             for a in alpha_orders for b in beta_orders]
                    phase_std = np.std(phases)
                    features['paper_fkm_phase_coherence'] = float(1.0 / (1.0 + phase_std))
                    
                else:
                    features['paper_fkm_scale_inv_1'] = 0.0
                    features['paper_fkm_rotation_inv_1'] = 0.0
                    features['paper_fkm_scale_inv_2'] = 0.0
                    features['paper_fkm_cross_order_inv'] = 0.0
                    features['paper_fkm_phase_coherence'] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"Error computing paper FKM invariants: {str(e)}")
                features['paper_fkm_scale_inv_1'] = 0.0
                features['paper_fkm_rotation_inv_1'] = 0.0
                features['paper_fkm_scale_inv_2'] = 0.0
                features['paper_fkm_cross_order_inv'] = 0.0
                features['paper_fkm_phase_coherence'] = 0.0
            
            # ========================================
            # STEP 6: Frequency-domain analysis
            # Low fractional orders = smooth patterns
            # High fractional orders = rapid changes
            # ========================================
            
            try:
                # Separate moments by fractional order magnitude
                low_order_mags = []
                high_order_mags = []
                
                for alpha in alpha_orders:
                    for beta in beta_orders:
                        mag = features.get(f'paper_fkm_{alpha}_{beta}_magnitude', 0.0)
                        
                        # Low orders: α+β <= 1.5 (smooth features)
                        if alpha + beta <= 1.5:
                            low_order_mags.append(mag)
                        # High orders: α+β > 2.5 (texture/edge features)
                        elif alpha + beta > 2.5:
                            high_order_mags.append(mag)
                
                if low_order_mags:
                    features['paper_fkm_low_order_energy'] = float(np.sum(np.array(low_order_mags) ** 2))
                    features['paper_fkm_low_order_mean'] = float(np.mean(low_order_mags))
                else:
                    features['paper_fkm_low_order_energy'] = 0.0
                    features['paper_fkm_low_order_mean'] = 0.0
                
                if high_order_mags:
                    features['paper_fkm_high_order_energy'] = float(np.sum(np.array(high_order_mags) ** 2))
                    features['paper_fkm_high_order_mean'] = float(np.mean(high_order_mags))
                else:
                    features['paper_fkm_high_order_energy'] = 0.0
                    features['paper_fkm_high_order_mean'] = 0.0
                
                # Order energy ratio (smoothness indicator)
                low_e = features['paper_fkm_low_order_energy']
                high_e = features['paper_fkm_high_order_energy']
                features['paper_fkm_order_ratio'] = float(low_e / (high_e + 1e-10))
                
                # Sparsity in high orders (indicates localized features like vessels)
                if high_order_mags:
                    threshold = np.mean(high_order_mags)
                    features['paper_fkm_high_order_sparsity'] = float(
                        np.sum(np.array(high_order_mags) > threshold) / len(high_order_mags)
                    )
                else:
                    features['paper_fkm_high_order_sparsity'] = 0.0
                
            except Exception as e:
                self.logger.warning(f"Error in paper FKM frequency analysis: {str(e)}")
                features['paper_fkm_low_order_energy'] = 0.0
                features['paper_fkm_low_order_mean'] = 0.0
                features['paper_fkm_high_order_energy'] = 0.0
                features['paper_fkm_high_order_mean'] = 0.0
                features['paper_fkm_order_ratio'] = 0.0
                features['paper_fkm_high_order_sparsity'] = 0.0
            
            # ========================================
            # STEP 7: Complex-valued features
            # Real/imaginary balance indicates pattern symmetry
            # ========================================
            
            try:
                real_parts = [features.get(f'paper_fkm_{a}_{b}_real', 0.0) 
                             for a in alpha_orders for b in beta_orders]
                imag_parts = [features.get(f'paper_fkm_{a}_{b}_imag', 0.0) 
                             for a in alpha_orders for b in beta_orders]
                
                real_energy = np.sum(np.array(real_parts) ** 2)
                imag_energy = np.sum(np.array(imag_parts) ** 2)

                #Phase of complex number
                
                features['paper_fkm_real_energy'] = float(real_energy)
                features['paper_fkm_imag_energy'] = float(imag_energy)
                
                # Real/imaginary balance (symmetric patterns have balanced energies)
                features['paper_fkm_complex_balance'] = float(
                    min(real_energy, imag_energy) / (max(real_energy, imag_energy) + 1e-10)
                )
                
                # Complex entropy (diversity of complex patterns)
                complex_mags = np.array([np.sqrt(r**2 + i**2) 
                                        for r, i in zip(real_parts, imag_parts)])
                s_complex = np.sum(complex_mags)
                if s_complex > 1e-10:
                    probs_complex = complex_mags / s_complex
                    probs_complex = probs_complex[probs_complex > 1e-10]
                    features['paper_fkm_complex_entropy'] = float(
                        -np.sum(probs_complex * np.log2(probs_complex))
                    )
                else:
                    features['paper_fkm_complex_entropy'] = 0.0
                
            except Exception as e:
                self.logger.warning(f"Error in paper FKM complex analysis: {str(e)}")
                features['paper_fkm_real_energy'] = 0.0
                features['paper_fkm_imag_energy'] = 0.0
                features['paper_fkm_complex_balance'] = 0.0
                features['paper_fkm_complex_entropy'] = 0.0
            


            return features
            
        except Exception as e:
            self.logger.error(f"Error in CORRECTED paper FKM: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_paper_fkm_features(alpha_orders or [0.5, 1.0, 1.5, 2.0],
                                                        beta_orders or [0.5, 1.0, 1.5, 2.0])


    def _compute_weighted_polynomial_eq4(self, x, n, p, N):
        """
        Compute weighted polynomial EXACTLY as Equation (4) in paper.

        φ_n(i) = 2F1(-n, -i; -N+1; 1/p) 
                × sqrt[ C(N-1, i) * p^i * (1-p)^(N-1-i) ]
                / sqrt[ ((p-1)/p)^n * n! / (-N+1)_n ]

        All computations in LOG-SPACE for numerical stability.
        """
        from scipy.special import hyp2f1

        if x < 0 or x > N or n < 0:
            return 0.0

        p = float(np.clip(p, 1e-12, 1 - 1e-12))

        try:
            # ========================================
            # Part 1: Hypergeometric function 2F1
            # ========================================
            z = 1.0 / p
            hyp_value = hyp2f1(-n, -x, -N + 1, z)

            if not np.isfinite(hyp_value) or abs(hyp_value) < 1e-100:
                return 0.0

            # ========================================
            # Part 2: Numerator in LOG-SPACE
            # numerator = C(N-1, x) * p^x * (1-p)^(N-1-x)
            # ========================================

            # Binomial coefficient C(N-1, x) in log-space
            log_binom = gammaln(N) - gammaln(x + 1) - gammaln(N - x)

            # Weight p^x * (1-p)^(N-1-x) in log-space
            log_weight = x * np.log(p) + (N - 1 - x) * np.log(1 - p)

            log_numerator = log_binom + log_weight

            # ========================================
            # Part 3: Denominator in LOG-SPACE
            # denominator = ((p-1)/p)^n * n! / (-N+1)_n
            # ========================================

            # ((p-1)/p)^n in log-space (handle sign separately because ratio < 0 for 0<p<1)
            ratio = (p - 1.0) / p
            abs_ratio = abs(ratio)
            log_ratio = n * np.log(abs_ratio)

            # sign coming from ((p-1)/p)^n: (-1)^n when ratio < 0
            sign_ratio = -1 if ratio < 0 else 1
            sign_ratio = sign_ratio ** n

            # n! in log-space
            log_factorial = gammaln(n + 1)

            # Pochhammer (-N+1)_n in log-space and its sign
            a = -N + 1

            if n == 0:
                log_pochhammer = 0.0
                sign_pochhammer = 1
            elif a <= 0 and abs(a - round(a)) < 1e-10:
                # Negative integer case
                # (-m)_n = (-1)^n * Γ(m+n) / Γ(m)
                m = int(abs(round(a)))

                try:
                    log_pochhammer = gammaln(m + n) - gammaln(m)
                    sign_pochhammer = (-1) ** n
                except:
                    # Fallback
                    log_pochhammer = n * np.log(m + n/2)
                    sign_pochhammer = (-1) ** n
            else:
                # Normal case
                try:
                    log_pochhammer = gammaln(a + n) - gammaln(a)
                    sign_pochhammer = 1
                except:
                    log_pochhammer = 0.0
                    sign_pochhammer = 1

            # combined sign for denominator
            sign_total = sign_ratio * sign_pochhammer

            # Combined denominator (log-space)
            log_denominator = log_ratio + log_factorial - log_pochhammer

            # ========================================
            # Part 4: Final normalization
            # norm = sqrt(numerator / denominator)
            # ========================================

            log_norm = 0.5 * (log_numerator - log_denominator)

            # Clip to prevent overflow/underflow
            log_norm = np.clip(log_norm, -100, 100)

            if not np.isfinite(log_norm):
                return 0.0

            # Convert back from log-space
            normalization = np.exp(log_norm)

            # Apply total sign (from ratio and Pochhammer)
            if sign_total == -1:
                normalization *= -1

            # Final result
            result = hyp_value * normalization

            if np.isfinite(result) and abs(result) > 1e-100:
                return float(result)

            return 0.0

        except Exception as e:
            self.logger.warning(f"Error in polynomial: x={x}, n={n}, N={N}: {str(e)}")
            return 0.0






    def _get_default_paper_fkm_features(self, alpha_orders, beta_orders):
        """Return default paper FKM features with ALL computed features."""
        features = {}
        
        # Complex moment features
        for alpha in alpha_orders:
            for beta in beta_orders:
                features[f'paper_fkm_{alpha}_{beta}_real'] = 0.0
                features[f'paper_fkm_{alpha}_{beta}_imag'] = 0.0
                features[f'paper_fkm_{alpha}_{beta}_magnitude'] = 0.0
                features[f'paper_fkm_{alpha}_{beta}_phase'] = 0.0
        
        # Statistical features
        features['paper_fkm_energy'] = 0.0
        features['paper_fkm_entropy'] = 0.0
        
        # Invariant features
        features['paper_fkm_scale_inv_1'] = 0.0
        features['paper_fkm_rotation_inv_1'] = 0.0
        features['paper_fkm_scale_inv_2'] = 0.0
        features['paper_fkm_cross_order_inv'] = 0.0
        features['paper_fkm_phase_coherence'] = 0.0
        
        # Frequency domain features
        features['paper_fkm_low_order_energy'] = 0.0
        features['paper_fkm_low_order_mean'] = 0.0
        features['paper_fkm_high_order_energy'] = 0.0
        features['paper_fkm_high_order_mean'] = 0.0
        features['paper_fkm_order_ratio'] = 0.0
        features['paper_fkm_high_order_sparsity'] = 0.0
        
        # Complex-valued features
        features['paper_fkm_real_energy'] = 0.0
        features['paper_fkm_imag_energy'] = 0.0
        features['paper_fkm_complex_balance'] = 0.0
        features['paper_fkm_complex_entropy'] = 0.0
        
        return features