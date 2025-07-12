import numpy as np
import logging
from skimage import color, feature, measure, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import convex_hull_image
from scipy import ndimage, stats
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
            
            # Extract dermoscopic-specific features (NEW)
            dermoscopic_features = self.extract_dermoscopic_features(image, mask)
            features.update(dermoscopic_features)
            
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
            
            # NEW: Apply contrast enhancement to improve feature extraction
            masked_gray_enhanced = filters.rank.enhance_contrast(
                (masked_gray * 255).astype(np.uint8), 
                footprint=np.ones((5, 5)),
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