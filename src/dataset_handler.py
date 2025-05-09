import os
import numpy as np
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
from src.conventional_features import ConventionalFeatureExtractor
import logging
from typing import List, Tuple
import glob
import pandas as pd

class DatasetHandler:
    def __init__(self, 
                 n_segments: int = 20, 
                 compactness: float = 10,
                 connectivity_threshold: float = 0.5,
                 max_images_per_class: int = 2000):
        """Initialize dataset handler with processing parameters.
        
        Args:
            n_segments: Number of superpixels to generate
            compactness: Compactness parameter for SLIC algorithm
            connectivity_threshold: Threshold for connecting nodes in the graph
            max_images_per_class: Maximum number of images to process per class (to balance dataset)
        """
        self.logger = logging.getLogger(__name__)
        # Initialize processing components
        self.preprocessor = ImagePreprocessor()
        self.superpixel_gen = SuperpixelGenerator(n_segments=n_segments, compactness=compactness)
        self.graph_constructor = GraphConstructor(connectivity_threshold)
        self.feature_extractor = FeatureExtractor()
        self.conv_feature_extractor = ConventionalFeatureExtractor()
        self.max_images_per_class = max_images_per_class
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
            
            # Limit number of images per class for balanced dataset
            if len(image_files) > self.max_images_per_class:
                self.logger.info(f"Limiting to {self.max_images_per_class} images from {len(image_files)} in {directory}")
                # Randomly select max_images_per_class images
                np.random.seed(42)  # For reproducibility
                image_files = np.random.choice(image_files, self.max_images_per_class, replace=False).tolist()
            
            # Process all images
            for image_path in image_files:
                try:
                    # Load and preprocess image
                    original_image = self.preprocessor.load_image(image_path)
                    processed_image = self.preprocessor.preprocess(original_image)
                    
                    # Generate superpixels
                    segments = self.superpixel_gen.generate_superpixels(processed_image)
                    features = self.superpixel_gen.compute_superpixel_features(
                        processed_image, segments)
                    
                    # Construct graph
                    G = self.graph_constructor.build_graph(features, segments)
                    
                    # Extract and store graph-based features in graph
                    G.graph['features'] = {
                        **self.feature_extractor.extract_local_features(G),
                        **self.feature_extractor.extract_global_features(G),
                        **self.feature_extractor.extract_spectral_features(G)
                    }
                    
                    # Calculate mask of the lesion (combining all superpixels)
                    lesion_mask = segments > -1  # All superpixels are part of the lesion
                    
                    # Extract conventional image features
                    conventional_features = self.conv_feature_extractor.extract_all_features(
                        original_image, lesion_mask)
                    
                    # Store conventional features in the graph
                    G.graph['conventional_features'] = conventional_features
                    
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
            
            self.logger.info(f"Split dataset: {len(train_graphs)} training, {len(test_graphs)} testing")
            
            return train_graphs, test_graphs, train_labels, test_labels
            
        except Exception as e:
            self.logger.error(f"Error splitting dataset: {str(e)}")
            raise

    def save_feature_matrix(self, graphs: List, labels: np.ndarray, output_path: str = 'output/features.csv') -> np.ndarray:
        """Extract and save feature matrix for analysis."""
        try:
            from src.classifier import MelanomaClassifier
            
            # Initialize classifier to use its feature preparation method
            classifier = MelanomaClassifier()
            
            # Prepare feature matrix
            X = classifier.prepare_features(graphs)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create feature names
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            df['label'] = labels
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Saved feature matrix with {X.shape[1]} features to {output_path}")
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error saving feature matrix: {str(e)}")
            raise