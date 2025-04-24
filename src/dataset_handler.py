import os
import numpy as np
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
from src.conventional_features import ConventionalFeatureExtractor
from src.dermoscopic_features import DermoscopicFeatureDetector
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
        self.dermo_feature_detector = DermoscopicFeatureDetector()
        self.max_images_per_class = max_images_per_class
        # Create necessary directories
        os.makedirs('data/bcc', exist_ok=True)
        os.makedirs('data/sk', exist_ok=True)
        os.makedirs('test', exist_ok=True)

    def process_dataset(self, 
                       bcc_dir: str, 
                       sk_dir: str) -> Tuple[List, List]:
        """Process all images in the dataset and return graphs and labels."""
        try:
            # Validate directories
            if not os.path.exists(bcc_dir):
                raise ValueError(f"BCC directory not found: {bcc_dir}")
            if not os.path.exists(sk_dir):
                raise ValueError(f"SK directory not found: {sk_dir}")
                
            # Process BCC images
            self.logger.info(f"Processing BCC images from {bcc_dir}")
            bcc_graphs = self._process_directory(bcc_dir)
            bcc_labels = np.ones(len(bcc_graphs))
            
            # Process SK images
            self.logger.info(f"Processing SK images from {sk_dir}")
            sk_graphs = self._process_directory(sk_dir)
            sk_labels = np.zeros(len(sk_graphs))
            
            # Validate we have data
            if not bcc_graphs and not sk_graphs:
                raise ValueError("No valid images found in either directory")
            
            # Combine data
            graphs = bcc_graphs + sk_graphs
            labels = np.concatenate([bcc_labels, sk_labels])
            
            self.logger.info(f"Processed {len(bcc_graphs)} BCC and {len(sk_graphs)} SK images")
            
            return graphs, labels
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise

    def _process_directory(self, directory: str) -> List:
        """Process all images in a directory and return their graph representations."""
        try:
            graphs = []
            # Search for image files with case-insensitive extensions
            image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                         glob.glob(os.path.join(directory, "*.jpeg")) + \
                         glob.glob(os.path.join(directory, "*.png")) + \
                         glob.glob(os.path.join(directory, "*.bmp")) + \
                         glob.glob(os.path.join(directory, "*.JPG")) + \
                         glob.glob(os.path.join(directory, "*.JPEG")) + \
                         glob.glob(os.path.join(directory, "*.PNG")) + \
                         glob.glob(os.path.join(directory, "*.BMP"))
                         
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
                    
                    # Extract specialized dermoscopic features (highly important for BCC vs SK)
                    dermoscopic_features = self.dermo_feature_detector.detect_all_features(
                        original_image, lesion_mask)
                    
                    # Store all features in the graph
                    G.graph['conventional_features'] = conventional_features
                    G.graph['dermoscopic_features'] = dermoscopic_features
                    
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
            from src.classifier import BCCSKClassifier
            
            # Initialize classifier to use its feature preparation method
            classifier = BCCSKClassifier()
            
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