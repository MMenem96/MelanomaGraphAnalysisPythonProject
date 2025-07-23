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


    def generate_superpixels_with_mask(self, image, lesion_mask, save_visualization=False, image_name=None):
        """Generate superpixels only within the lesion area with optional visualization"""
        try:
            from skimage.segmentation import slic
            
            # Reduce number of segments for pre-segmented lesions
            effective_segments = max(10, self.n_segments // 2)
            
            print(f"Generating {effective_segments} superpixels for masked image")
            
            # Apply SLIC only to lesion area
            segments = slic(image, 
                        n_segments=effective_segments, 
                        compactness=self.compactness, 
                        start_label=1,
                        mask=lesion_mask,  # Only segment lesion area
                        channel_axis=-1 if len(image.shape) == 3 else None)
            
            # Ensure background pixels are labeled as 0
            segments[~lesion_mask] = 0
            
            # Count actual segments created
            unique_segments = np.unique(segments[segments > 0])
            print(f"Successfully created {len(unique_segments)} superpixel segments")
            
            return segments
            
        except Exception as e:
            print(f"Error in masked superpixel generation: {str(e)}")
            # Fallback to regular segmentation
            return self.generate_superpixels(image)

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