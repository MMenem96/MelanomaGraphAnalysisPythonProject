import numpy as np
import logging
from skimage.segmentation import slic
from scipy.ndimage import binary_erosion

class SuperpixelGenerator:
    def __init__(self, n_segments=20, compactness=10):
        """Initialize SuperpixelGenerator with parameters from the paper."""
        self.n_segments = n_segments
        self.compactness = compactness
        self.logger = logging.getLogger(__name__)

    def generate_superpixels(self, image):
        """Generate superpixels using SLIC algorithm as specified in the paper."""
        try:
            # Generate initial superpixels with exact count
            segments = slic(
                image, 
                n_segments=self.n_segments,
                compactness=self.compactness,
                convert2lab=False,  # Image already in LAB space
                start_label=0
            )

            # Ensure we have exactly n_segments
            unique_segments = len(np.unique(segments))
            if unique_segments != self.n_segments:
                segments = self._merge_superpixels(image, segments)

            return segments

        except Exception as e:
            self.logger.error(f"Error generating superpixels: {str(e)}")
            raise RuntimeError(f"Error generating superpixels: {str(e)}")

    def _merge_superpixels(self, image, segments):
        """Merge superpixels to achieve desired number of segments."""
        try:
            current_segments = len(np.unique(segments))

            while current_segments > self.n_segments:
                # Calculate mean color values for each segment
                segment_means = self._calculate_segment_means(image, segments)

                # Find most similar adjacent segments based on color and spatial proximity
                min_dist = float('inf')
                merge_pair = None

                # Get unique segment labels
                unique_labels = np.unique(segments)

                # Find adjacent segments with most similar properties
                for i in range(len(unique_labels)):
                    mask_i = segments == unique_labels[i]
                    dilated = binary_erosion(mask_i, structure=np.ones((3,3)))
                    neighbors = np.unique(segments[dilated & ~mask_i])

                    pos_i = np.mean(np.where(mask_i), axis=1)

                    for neighbor in neighbors:
                        if neighbor != unique_labels[i]:
                            # Calculate color difference
                            color_dist = np.linalg.norm(
                                segment_means[unique_labels[i]] - segment_means[neighbor]
                            )

                            # Calculate spatial distance
                            mask_n = segments == neighbor
                            pos_n = np.mean(np.where(mask_n), axis=1)
                            spatial_dist = np.linalg.norm(pos_i - pos_n)

                            # Combined distance metric (0.7 color + 0.3 spatial)
                            dist = 0.7 * color_dist + 0.3 * spatial_dist

                            if dist < min_dist:
                                min_dist = dist
                                merge_pair = (unique_labels[i], neighbor)

                if merge_pair:
                    # Merge segments
                    segments[segments == merge_pair[1]] = merge_pair[0]
                    # Update count
                    current_segments = len(np.unique(segments))
                else:
                    break

            # Relabel segments to ensure consecutive numbering
            unique_labels = np.unique(segments)
            relabel_map = {old: new for new, old in enumerate(unique_labels)}
            segments = np.vectorize(relabel_map.get)(segments)

            return segments

        except Exception as e:
            self.logger.error(f"Error merging superpixels: {str(e)}")
            raise RuntimeError(f"Error merging superpixels: {str(e)}")

    def _calculate_segment_means(self, image, segments):
        """Calculate mean LAB values for each segment."""
        try:
            unique_labels = np.unique(segments)
            means = {}

            for label in unique_labels:
                mask = segments == label
                if np.any(mask):
                    means[label] = np.mean(image[mask], axis=0)
                else:
                    means[label] = np.zeros(image.shape[-1])

            return means

        except Exception as e:
            self.logger.error(f"Error calculating segment means: {str(e)}")
            raise RuntimeError(f"Error calculating segment means: {str(e)}")

    def compute_superpixel_features(self, image, segments):
        """Compute features for each superpixel according to paper specifications."""
        try:
            unique_labels = np.unique(segments)
            n_features = 11  # 3(LAB) + 3(std) + 2(pos) + 3(shape)
            features = np.zeros((len(unique_labels), n_features))

            for i, label in enumerate(unique_labels):
                mask = segments == label
                if np.any(mask):
                    # Color features (LAB values)
                    color_values = image[mask]
                    color_mean = np.mean(color_values, axis=0)
                    color_std = np.std(color_values, axis=0)

                    # Position features (normalized centroid)
                    positions = np.array(np.where(mask))
                    centroid = np.mean(positions, axis=1)
                    centroid_norm = centroid / np.array(image.shape[:2])

                    # Shape features
                    area = np.sum(mask) / float(mask.size)
                    perimeter = self._calculate_perimeter(mask)
                    perimeter_norm = perimeter / float(np.sum(mask))
                    compactness = 4 * np.pi * area / (perimeter_norm**2) if perimeter_norm > 0 else 0

                    # Combine features
                    features[i] = np.concatenate([
                        color_mean,               # 3 features
                        color_std,                # 3 features
                        centroid_norm,            # 2 features
                        [area, perimeter_norm, compactness]  # 3 features
                    ])

            return features

        except Exception as e:
            self.logger.error(f"Error computing superpixel features: {str(e)}")
            raise RuntimeError(f"Error computing superpixel features: {str(e)}")

    def _calculate_perimeter(self, mask):
        """Calculate perimeter of a binary mask."""
        try:
            eroded = binary_erosion(mask)
            perimeter = np.sum(mask & ~eroded)
            return float(perimeter if perimeter > 0 else 1)
        except Exception as e:
            self.logger.error(f"Error calculating perimeter: {str(e)}")
            raise RuntimeError(f"Error calculating perimeter: {str(e)}")