import argparse
import logging
import os
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
from src.visualization import Visualizer
from src.classifier import MelanomaClassifier
from src.dataset_handler import DatasetHandler
from src.utils import setup_logging, validate_image_path, validate_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='Melanoma Detection using Superpixel Graphs')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode of operation: train or predict')
    parser.add_argument('--melanoma-dir', help='Directory containing melanoma images (for training)')
    parser.add_argument('--benign-dir', help='Directory containing benign images (for training)')
    parser.add_argument('--image', help='Path to the input image (for prediction)')
    parser.add_argument('--n-segments', type=int, default=20, help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=10, help='Compactness parameter for SLIC')
    parser.add_argument('--connectivity-threshold', type=float, default=0.5, 
                      help='Threshold for graph connectivity')
    parser.add_argument('--classifier', choices=['svm', 'rf'], default='svm',
                      help='Classifier type: Support Vector Machine (svm) or Random Forest (rf)')
    return parser.parse_args()

def train(args, logger):
    """Train the melanoma detection model."""
    try:
        # Initialize dataset handler
        dataset_handler = DatasetHandler(
            n_segments=args.n_segments,
            compactness=args.compactness,
            connectivity_threshold=args.connectivity_threshold
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

        # Train and evaluate
        logger.info("Training and evaluating model...")
        results = classifier.train_evaluate(X_train, train_labels)

        # Log results
        logger.info("Training Results:")
        logger.info(f"Accuracy: {results['accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        logger.info(f"Precision: {results['precision']:.3f} ± {results['std_precision']:.3f}")
        logger.info(f"Recall: {results['recall']:.3f} ± {results['std_recall']:.3f}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def predict(args, logger):
    """Predict melanoma probability for a single image."""
    try:
        # Initialize components
        preprocessor = ImagePreprocessor()
        superpixel_gen = SuperpixelGenerator(
            n_segments=args.n_segments,
            compactness=args.compactness
        )
        graph_constructor = GraphConstructor(
            connectivity_threshold=args.connectivity_threshold
        )
        feature_extractor = FeatureExtractor()
        visualizer = Visualizer()
        classifier = MelanomaClassifier(classifier_type=args.classifier)

        # Process image
        logger.info("Loading and preprocessing image...")
        image = preprocessor.load_image(args.image)
        processed_image = preprocessor.preprocess(image)

        # Generate superpixels
        logger.info("Generating superpixels...")
        segments = superpixel_gen.generate_superpixels(processed_image)
        features = superpixel_gen.compute_superpixel_features(processed_image, segments)

        # Construct graph
        logger.info("Constructing graph...")
        G = graph_constructor.build_graph(features, segments)

        # Extract features
        logger.info("Extracting features...")
        G.graph['features'] = {
            **feature_extractor.extract_local_features(G),
            **feature_extractor.extract_global_features(G),
            **feature_extractor.extract_spectral_features(G)
        }

        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer.plot_superpixels(image, segments)
        visualizer.plot_graph(G)
        visualizer.plot_features(
            G.graph['features'],
            {k: v for k, v in G.graph['features'].items() 
             if not isinstance(v, dict)}
        )

        # Make prediction if model exists
        model_path = 'model/melanoma_classifier.joblib'
        scaler_path = 'model/scaler.joblib'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info("Making prediction...")
            # Prepare features for prediction
            X = classifier.prepare_features([G])

            # Load model and scaler
            from joblib import load
            model = load(model_path)
            scaler = load(scaler_path)

            # Scale features and predict
            X_scaled = scaler.transform(X)
            probability = model.predict_proba(X_scaled)[0][1]

            logger.info(f"Probability of melanoma: {probability:.2%}")
            risk_level = 'HIGH' if probability > 0.5 else 'LOW'
            logger.info(f"{risk_level} risk of melanoma")

            # Save prediction result
            with open(os.path.join('output', 'prediction_result.txt'), 'w') as f:
                f.write(f"Melanoma Risk Assessment\n")
                f.write(f"------------------------\n")
                f.write(f"Image: {args.image}\n")
                f.write(f"Probability: {probability:.2%}\n")
                f.write(f"Risk Level: {risk_level}\n")
        else:
            logger.warning("No trained model found. Please train the model first using --mode train")

        logger.info("Processing completed successfully.")
        logger.info("Visualization files have been saved in the 'output' directory:")
        logger.info("- output/superpixels.png")
        logger.info("- output/graph.png")
        logger.info("- output/features.png")
        if os.path.exists(model_path):
            logger.info("- output/prediction_result.txt")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    # Setup logging
    logger = setup_logging()

    # Parse arguments
    args = parse_args()

    try:
        # Validate inputs
        if args.mode == 'train':
            if not args.melanoma_dir or not args.benign_dir:
                raise ValueError("Training mode requires --melanoma-dir and --benign-dir")
        else:  # predict mode
            if not args.image:
                raise ValueError("Prediction mode requires --image")
            validate_image_path(args.image)

        validate_parameters(vars(args))

        # Execute requested mode
        if args.mode == 'train':
            train(args, logger)
        else:
            predict(args, logger)

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()