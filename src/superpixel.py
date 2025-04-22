import numpy as np
from skimage.segmentation import slic
from skimage import color
import logging

class SuperpixelGenerator:
    def __init__(self, n_segments=20, compactness=10, sigma=1):
        """Initialize superpixel generator with segmentation parameters."""
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.logger = logging.getLogger(__name__)
        
    def generate_superpixels(self, image):
        """Generate superpixels for an image using SLIC."""
        try:
            # Apply SLIC algorithm to generate superpixels
            segments = slic(
                image,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=0
            )
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Error generating superpixels: {str(e)}")
            raise
            
    def compute_superpixel_features(self, image, segments):
        """Compute features for each superpixel."""
        try:
            # Get the number of segments
            n_segments = np.max(segments) + 1
            
            # Convert to LAB color space (better for color similarity)
            image_lab = color.rgb2lab(image)
            
            # Initialize feature matrix
            # Features: L, a, b (color), x, y (position)
            features = np.zeros((n_segments, 5))
            
            # For each segment, compute mean color and position
            for i in range(n_segments):
                mask = segments == i
                
                # Skip empty segments
                if np.sum(mask) == 0:
                    continue
                
                # Mean color in LAB space
                features[i, :3] = np.mean(image_lab[mask], axis=0)
                
                # Mean x, y position
                y, x = np.where(mask)
                features[i, 3:] = [np.mean(y), np.mean(x)]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error computing superpixel features: {str(e)}")
            raise