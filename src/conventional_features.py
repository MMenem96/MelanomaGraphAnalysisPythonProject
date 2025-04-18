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
        """Extract texture features using various methods."""
        try:
            features = {}
            
            # Convert to grayscale for texture analysis
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray_image = color.rgb2gray(image[:,:,:3])
            elif len(image.shape) == 3:
                gray_image = image[:,:,0].astype(float)
            else:
                gray_image = image.astype(float)
            
            # Normalize to 8-bit for GLCM
            gray_image_8bit = (gray_image * 255).astype(np.uint8)
            
            # Only consider pixels within the mask
            masked_gray = np.zeros_like(gray_image_8bit)
            masked_gray[mask] = gray_image_8bit[mask]
            
            # Check if we have enough pixels for texture analysis
            if np.sum(mask) <= 5:
                self.logger.warning("Not enough pixels in mask for texture analysis")
                return features
                
            # GLCM features (Haralick)
            distances = [1, 3]  # Distance between pixel pairs
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles to consider
            
            # Make sure we have some intensity variation
            if np.max(masked_gray) > np.min(masked_gray):
                try:
                    # Reduce levels to avoid sparse matrices with few pixels
                    levels = min(256, np.sum(mask))
                    levels = max(2, min(levels, 16))  # Between 2 and 16 levels
                    
                    # Rescale to reduced levels
                    masked_scaled = np.interp(masked_gray, 
                                            (np.min(masked_gray), np.max(masked_gray)), 
                                            (0, levels-1)).astype(np.uint8)
                    
                    glcm = graycomatrix(masked_scaled, distances, angles, 
                                       levels=levels, symmetric=True, normed=True)
                    
                    # Compute Haralick texture features
                    properties = ['contrast', 'dissimilarity', 'homogeneity', 
                                  'energy', 'correlation', 'ASM']
                    
                    for prop in properties:
                        propmatrix = graycoprops(glcm, prop)
                        for d_idx, distance in enumerate(distances):
                            for a_idx, angle in enumerate(angles):
                                features[f'glcm_{prop}_d{distance}_a{int(angle*180/np.pi)}'] = float(propmatrix[d_idx, a_idx])
                except Exception as e:
                    self.logger.warning(f"Failed to compute GLCM: {str(e)}")
            
            # Wavelet features
            # Apply wavelet transform
            try:
                coeffs = pywt.dwt2(gray_image, 'db1')
                # Extract energy from each coefficient
                cA, (cH, cV, cD) = coeffs
                features['wavelet_cA_energy'] = float(np.mean(cA**2))
                features['wavelet_cH_energy'] = float(np.mean(cH**2))
                features['wavelet_cV_energy'] = float(np.mean(cV**2))
                features['wavelet_cD_energy'] = float(np.mean(cD**2))
            except Exception as e:
                self.logger.warning(f"Failed to compute wavelet features: {str(e)}")
                features['wavelet_cA_energy'] = 0.0
                features['wavelet_cH_energy'] = 0.0
                features['wavelet_cV_energy'] = 0.0
                features['wavelet_cD_energy'] = 0.0
            
            # Apply LBP (Local Binary Pattern)
            try:
                radius = 3
                n_points = 8 * radius
                lbp = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')
                
                # Only consider pixels within the mask
                lbp_masked = lbp[mask]
                if len(lbp_masked) > 0:
                    lbp_hist, _ = np.histogram(lbp_masked, bins=n_points+2, range=(0, n_points+2), density=True)
                    
                    # Add LBP histogram features
                    for i, value in enumerate(lbp_hist):
                        features[f'lbp_hist_{i}'] = float(value)
            except Exception as e:
                self.logger.warning(f"Failed to compute LBP features: {str(e)}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {str(e)}")
            return {}