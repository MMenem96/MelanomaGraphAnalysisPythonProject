import os
import numpy as np
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
import logging
from typing import List, Tuple
import glob
import pandas as pd

class DatasetHandler:
    def __init__(self, 
                 n_segments: int = 20, 
                 compactness: float = 10,
                 connectivity_threshold: float = 0.5):
        """Initialize dataset handler with processing parameters."""
        self.logger = logging.getLogger(__name__)

        # Initialize processing components
        self.preprocessor = ImagePreprocessor()
        self.superpixel_gen = SuperpixelGenerator(n_segments=n_segments, compactness=compactness)
        self.graph_constructor = GraphConstructor(connectivity_threshold)
        self.feature_extractor = FeatureExtractor()

        # Create necessary directories
        os.makedirs('data/melanoma', exist_ok=True)
        os.makedirs('data/benign', exist_ok=True)
        os.makedirs('test', exist_ok=True)

    def process_dataset(self, 
                       melanoma_dir: str, 
                       benign_dir: str) -> Tuple[List, List]:
        """Process all images in the dataset and return graphs and labels."""
        try:
            # Validate directories
            if not os.path.exists(melanoma_dir):
                raise ValueError(f"Melanoma directory not found: {melanoma_dir}")
            if not os.path.exists(benign_dir):
                raise ValueError(f"Benign directory not found: {benign_dir}")

            # Process melanoma images
            self.logger.info(f"Processing melanoma images from {melanoma_dir}")
            melanoma_graphs = self._process_directory(melanoma_dir)
            melanoma_labels = np.ones(len(melanoma_graphs))

            # Process benign images
            self.logger.info(f"Processing benign images from {benign_dir}")
            benign_graphs = self._process_directory(benign_dir)
            benign_labels = np.zeros(len(benign_graphs))

            # Validate we have data
            if not melanoma_graphs and not benign_graphs:
                raise ValueError("No valid images found in either directory")

            # Combine data
            graphs = melanoma_graphs + benign_graphs
            labels = np.concatenate([melanoma_labels, benign_labels])

            self.logger.info(f"Processed {len(melanoma_graphs)} melanoma and {len(benign_graphs)} benign images")
            return graphs, labels

        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise

    def _process_directory(self, directory: str) -> List:
        """Process all images in a directory and return their graph representations."""
        try:
            graphs = []
            image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                         glob.glob(os.path.join(directory, "*.jpeg")) + \
                         glob.glob(os.path.join(directory, "*.png")) + \
                         glob.glob(os.path.join(directory, "*.bmp"))

            if not image_files:
                self.logger.warning(f"No image files found in directory: {directory}")
                return graphs

            # Process all images
            for image_path in image_files:
                try:
                    # Load and preprocess image
                    image = self.preprocessor.load_image(image_path)
                    processed_image = self.preprocessor.preprocess(image)

                    # Generate superpixels
                    segments = self.superpixel_gen.generate_superpixels(processed_image)
                    features = self.superpixel_gen.compute_superpixel_features(
                        processed_image, segments)

                    # Construct graph
                    G = self.graph_constructor.build_graph(features, segments)

                    # Extract and store features in graph
                    G.graph['features'] = {
                        **self.feature_extractor.extract_local_features(G),
                        **self.feature_extractor.extract_global_features(G),
                        **self.feature_extractor.extract_spectral_features(G)
                    }

                    graphs.append(G)

                except Exception as e:
                    self.logger.warning(f"Error processing image {image_path}: {str(e)}")
                    continue

            self.logger.info(f"Successfully processed {len(graphs)} images from {directory}")
            return graphs

        except Exception as e:
            self.logger.error(f"Error processing directory {directory}: {str(e)}")
            raise

    def split_dataset(self, 
                     graphs: List, 
                     labels: np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """Split dataset into training and testing sets."""
        try:
            if not graphs:
                raise ValueError("No graphs provided for splitting")

            if len(graphs) < 2:
                raise ValueError("Need at least 2 samples to split the dataset")

            # Generate random indices
            np.random.seed(random_state)
            indices = np.random.permutation(len(labels))

            # Calculate split point
            split_point = max(1, int(len(labels) * (1 - test_size)))

            # Split data
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]

            train_graphs = [graphs[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            self.logger.info(f"Split dataset: {len(train_graphs)} training, {len(test_graphs)} testing samples")
            return train_graphs, test_graphs, train_labels, test_labels

        except Exception as e:
            self.logger.error(f"Error splitting dataset: {str(e)}")
            raise

    def save_feature_matrix(self, graphs, labels, output_path='data/features.csv'):
        """Save extracted features to a CSV file with labels."""
        try:
            # Create feature matrix
            feature_matrix = self.feature_extractor.create_feature_matrix(graphs, labels)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to CSV
            feature_matrix.to_csv(output_path, index=False)
            self.logger.info(f"Feature matrix saved to {output_path}")

            return feature_matrix

        except Exception as e:
            self.logger.error(f"Error saving feature matrix: {str(e)}")
            raise RuntimeError(f"Error saving feature matrix: {str(e)}")