import os
import argparse
import logging
from src.dataset_handler import DatasetHandler
from src.classifier import MelanomaClassifier

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("melanoma_detection.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MelanomaDetection")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Detection System')
    
    # Dataset paths
    parser.add_argument('--melanoma-dir', type=str, default='data/melanoma',
                        help='Directory containing melanoma images')
    parser.add_argument('--benign-dir', type=str, default='data/benign',
                        help='Directory containing benign lesion images')
    
    # Dataset balance parameters
    parser.add_argument('--max-images-per-class', type=int, default=2000,
                        help='Maximum number of images to use per class for balanced dataset')
    
    # Model parameters
    parser.add_argument('--classifier', type=str, choices=['svm', 'rf'], default='svm',
                        help='Type of classifier to use')
    
    # Superpixel parameters
    parser.add_argument('--n-segments', type=int, default=20,
                        help='Number of superpixel segments')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='Compactness parameter for SLIC')
    
    # Graph construction parameters
    parser.add_argument('--connectivity-threshold', type=float, default=0.5,
                        help='Threshold for connecting superpixels in graph')
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Operation mode: train or evaluate')
    
    return parser.parse_args()

def train(args, logger):
    """Train the melanoma detection model."""
    try:
        # Initialize dataset handler
        dataset_handler = DatasetHandler(
            n_segments=args.n_segments,
            compactness=args.compactness,
            connectivity_threshold=args.connectivity_threshold,
            max_images_per_class=args.max_images_per_class
        )

        # Process dataset
        logger.info("Processing dataset...")
        graphs, labels = dataset_handler.process_dataset(
            args.melanoma_dir,
            args.benign_dir
        )

        # Save feature matrix
        logger.info("Creating and saving feature matrix...")
        feature_matrix = dataset_handler.save_feature_matrix(graphs, labels)
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")

        # Split dataset
        train_graphs, test_graphs, train_labels, test_labels = \
            dataset_handler.split_dataset(graphs, labels)

        # Initialize classifier
        classifier = MelanomaClassifier(classifier_type=args.classifier)

        # Prepare features
        logger.info("Preparing features...")
        X_train = classifier.prepare_features(train_graphs)
        X_test = classifier.prepare_features(test_graphs)
        
        logger.info(f"Feature matrix shape: {X_train.shape} with {X_train.shape[1]} features")

        # Train and evaluate
        logger.info("Training and evaluating model...")
        results = classifier.train_evaluate(X_train, train_labels)

        # Log results
        logger.info("Training Results:")
        logger.info(f"Accuracy: {results['accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        logger.info(f"Precision: {results['precision']:.3f} ± {results['std_precision']:.3f}")
        logger.info(f"Recall: {results['recall']:.3f} ± {results['std_recall']:.3f}")
        
        # Perform additional evaluation on test set
        logger.info("Performing evaluation on test set...")
        test_evaluation = classifier.evaluate_model(X_test, test_labels)
        
        logger.info("Test Set Evaluation Complete")
        logger.info("Check output directory for detailed results and visualizations")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def evaluate(args, logger):
    """Evaluate a trained model."""
    try:
        # Initialize dataset handler
        dataset_handler = DatasetHandler(
            n_segments=args.n_segments,
            compactness=args.compactness,
            connectivity_threshold=args.connectivity_threshold,
            max_images_per_class=args.max_images_per_class
        )

        # Process test dataset
        logger.info("Processing test dataset...")
        graphs, labels = dataset_handler.process_dataset(
            args.melanoma_dir,
            args.benign_dir
        )

        # Initialize classifier
        classifier = MelanomaClassifier(classifier_type=args.classifier)

        # Prepare features
        logger.info("Preparing features...")
        X = classifier.prepare_features(graphs)

        # Load model
        logger.info("Loading model...")
        model_path = 'model/melanoma_classifier.joblib'
        scaler_path = 'model/scaler.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.error("Model not found. Train the model first.")
            return
            
        classifier.load_model(model_path, scaler_path)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = classifier.evaluate_model(X, labels)
        
        logger.info("Evaluation Complete")
        logger.info("Check output directory for detailed results and visualizations")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    """Main entry point."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('data/melanoma', exist_ok=True)
    os.makedirs('data/benign', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    logger.info(f"Running in {args.mode} mode")
    
    # Execute based on mode
    if args.mode == 'train':
        train(args, logger)
    elif args.mode == 'evaluate':
        evaluate(args, logger)

if __name__ == "__main__":
    main()