import os
import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import glob
from PIL import Image

from src.dataset_handler import DatasetHandler
from src.classifier import MelanomaClassifier
from src.cnn_classifier import CNNMelanomaClassifier

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
    parser.add_argument('--classifier', type=str, choices=['svm', 'rf', 'cnn'], default='svm',
                        help='Single classifier to use (legacy option)')
    parser.add_argument('--classifiers', type=str, default='all',
                        help='Comma-separated list of classifiers to train (e.g., svm_rbf,knn,rf,cnn) or "all"')
    
    # CNN-specific parameters
    parser.add_argument('--cnn-model', type=str, 
                      choices=['custom', 'resnet50', 'efficient_net', 'inception_v3'], 
                      default='efficient_net',
                      help='CNN architecture to use')
    parser.add_argument('--input-size', type=int, default=224,
                      help='Input image size for CNN (square)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs for CNN')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for CNN training')
    
    # Superpixel parameters
    parser.add_argument('--n-segments', type=int, default=20,
                        help='Number of superpixel segments')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='Compactness parameter for SLIC')
    
    # Graph construction parameters
    parser.add_argument('--connectivity-threshold', type=float, default=0.5,
                        help='Threshold for connecting superpixels in graph')
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'compare'], default='train',
                        help='Operation mode: train (single model), evaluate, or compare (multiple models)')
    
    return parser.parse_args()

def train(args, logger):
    """Train the melanoma detection model."""
    try:
        # Check if using CNN mode
        if args.classifier == 'cnn':
            train_cnn(args, logger)
            return
            
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
        
        # Save the model
        logger.info("Saving model to model directory...")
        model_info = classifier.save_model(output_dir='model')
        logger.info(f"Model saved to {model_info['model_path']}")
        logger.info(f"Scaler saved to {model_info['scaler_path']}")
        
        logger.info("Test Set Evaluation Complete")
        logger.info("Check output directory for detailed results and visualizations")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def train_cnn(args, logger):
    """Train a CNN-based melanoma detection model."""
    try:
        start_time = time.time()
        logger.info(f"Starting CNN training with {args.cnn_model} architecture")
        
        # Configure input shape
        input_shape = (args.input_size, args.input_size, 3)
        
        # Get image paths from melanoma and benign directories
        melanoma_paths = glob.glob(os.path.join(args.melanoma_dir, "*.jpg")) + \
                       glob.glob(os.path.join(args.melanoma_dir, "*.png")) + \
                       glob.glob(os.path.join(args.melanoma_dir, "*.jpeg"))
                       
        benign_paths = glob.glob(os.path.join(args.benign_dir, "*.jpg")) + \
                      glob.glob(os.path.join(args.benign_dir, "*.png")) + \
                      glob.glob(os.path.join(args.benign_dir, "*.jpeg"))
        
        # Limit number of images if needed
        if args.max_images_per_class > 0:
            np.random.seed(42)  # For reproducibility
            if len(melanoma_paths) > args.max_images_per_class:
                melanoma_paths = np.random.choice(melanoma_paths, args.max_images_per_class, replace=False).tolist()
            if len(benign_paths) > args.max_images_per_class:
                benign_paths = np.random.choice(benign_paths, args.max_images_per_class, replace=False).tolist()
        
        # Combine paths and create labels
        all_paths = melanoma_paths + benign_paths
        labels = np.array([1] * len(melanoma_paths) + [0] * len(benign_paths))
        
        # Shuffle data
        indices = np.arange(len(all_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        all_paths = np.array(all_paths)[indices].tolist()
        labels = labels[indices]
        
        logger.info(f"Dataset contains {len(melanoma_paths)} melanoma and {len(benign_paths)} benign images")
        
        # Initialize the CNN classifier
        cnn_classifier = CNNMelanomaClassifier(
            model_type=args.cnn_model,
            input_shape=input_shape
        )
        
        # Split data into training and test sets (80/20)
        from sklearn.model_selection import train_test_split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        logger.info(f"Training CNN with {len(train_paths)} images, validating with {len(test_paths)} images")
        history = cnn_classifier.fit(
            train_paths, train_labels,
            X_val=test_paths, y_val=test_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unfreeze_layers=10,  # Unfreeze some layers for fine-tuning
            fine_tune_epochs=20,
            model_path='model/cnn_model.h5'
        )
        
        # Evaluate the model
        logger.info("Evaluating CNN model...")
        eval_results = cnn_classifier.evaluate(test_paths, test_labels)
        
        # Print evaluation results
        logger.info("CNN Evaluation Results:")
        logger.info(f"Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"AUC: {eval_results['auc']:.4f}")
        logger.info(f"Sensitivity: {eval_results['sensitivity']:.4f}")
        logger.info(f"Specificity: {eval_results['specificity']:.4f}")
        
        # Plot and save training history
        cnn_classifier.plot_training_history(save_path='output/cnn_training_history.png')
        
        # Calculate training time
        total_time = time.time() - start_time
        logger.info(f"CNN Training completed in {total_time/60:.2f} minutes")
        
        logger.info("CNN Training Complete")
        logger.info("Check output directory for detailed results and visualizations")
    
    except Exception as e:
        logger.error(f"Error during CNN training: {str(e)}")
        raise

def evaluate(args, logger):
    """Evaluate a trained model."""
    try:
        # Check if evaluating CNN model
        if args.classifier == 'cnn':
            evaluate_cnn(args, logger)
            return
            
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
        
def evaluate_cnn(args, logger):
    """Evaluate a trained CNN model."""
    try:
        logger.info("Evaluating CNN model...")
        
        # Configure input shape
        input_shape = (args.input_size, args.input_size, 3)
        
        # Get image paths from test directories
        melanoma_paths = glob.glob(os.path.join(args.melanoma_dir, "*.jpg")) + \
                       glob.glob(os.path.join(args.melanoma_dir, "*.png")) + \
                       glob.glob(os.path.join(args.melanoma_dir, "*.jpeg"))
                       
        benign_paths = glob.glob(os.path.join(args.benign_dir, "*.jpg")) + \
                      glob.glob(os.path.join(args.benign_dir, "*.png")) + \
                      glob.glob(os.path.join(args.benign_dir, "*.jpeg"))
        
        # Limit number of images if needed
        if args.max_images_per_class > 0:
            np.random.seed(42)  # For reproducibility
            if len(melanoma_paths) > args.max_images_per_class:
                melanoma_paths = np.random.choice(melanoma_paths, args.max_images_per_class, replace=False).tolist()
            if len(benign_paths) > args.max_images_per_class:
                benign_paths = np.random.choice(benign_paths, args.max_images_per_class, replace=False).tolist()
        
        # Combine paths and create labels
        all_paths = melanoma_paths + benign_paths
        labels = np.array([1] * len(melanoma_paths) + [0] * len(benign_paths))
        
        # Shuffle data
        indices = np.arange(len(all_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        all_paths = np.array(all_paths)[indices].tolist()
        labels = labels[indices]
        
        logger.info(f"Evaluating CNN with {len(all_paths)} images ({len(melanoma_paths)} melanoma, {len(benign_paths)} benign)")
        
        # Initialize the CNN classifier
        cnn_classifier = CNNMelanomaClassifier(
            model_type=args.cnn_model,
            input_shape=input_shape
        )
        
        # Load model
        model_path = 'model/cnn_model.h5'
        if not os.path.exists(model_path):
            logger.error("CNN model not found. Train the model first.")
            return
            
        cnn_classifier.load_model(model_path)
        
        # Evaluate model
        eval_results = cnn_classifier.evaluate(all_paths, labels)
        
        # Print evaluation results
        logger.info("CNN Evaluation Results:")
        logger.info(f"Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"AUC: {eval_results['auc']:.4f}")
        logger.info(f"Sensitivity: {eval_results['sensitivity']:.4f}")
        logger.info(f"Specificity: {eval_results['specificity']:.4f}")
        
        logger.info("CNN Evaluation Complete")
        
    except Exception as e:
        logger.error(f"Error during CNN evaluation: {str(e)}")
        raise

# Dictionary of available classifiers
CLASSIFIERS = {
    'SVM (RBF)': {
        'class': SVC,
        'params': {
            'kernel': 'rbf', 
            'probability': True, 
            'C': 10.0, 
            'gamma': 'scale',
            'random_state': 42,
            'class_weight': 'balanced'
        }
    },
    'SVM (Sigmoid)': {
        'class': SVC,
        'params': {
            'kernel': 'sigmoid', 
            'probability': True, 
            'C': 10.0, 
            'gamma': 'scale',
            'random_state': 42,
            'class_weight': 'balanced'
        }
    },
    'SVM (Poly)': {
        'class': SVC,
        'params': {
            'kernel': 'poly', 
            'probability': True, 
            'C': 10.0, 
            'gamma': 'scale',
            'degree': 3,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    },
    'KNN': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 30,
            'n_jobs': -1
        }
    },
    'MLP': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42
        }
    },
    'RF': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    }
}

def specificity_score(y_true, y_pred):
    """Calculate the specificity score (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def generate_summary_table(results, logger, table_num=1, title="Melanoma Detection Model Comparison Summary"):
    """
    Generate a clean summary table showing metrics for each classifier.
    
    Args:
        results: Dictionary containing results for all classifiers
        logger: Logger instance
        table_num: Table number (1-5) for saving to appropriate file
        title: Title for the table visualization
    """
    logger.info(f"Generating summary table {table_num} of model performance")
    
    try:
        # The metrics we want to include
        metrics = ['AC', 'AUC', 'SN', 'SP']
        metric_names = {
            'AC': 'Accuracy (%)',
            'AUC': 'AUC (%)',
            'SN': 'Sensitivity (%)',
            'SP': 'Specificity (%)'
        }
        
        # Create a DataFrame with classifiers as rows and metrics as columns
        classifiers = list(results.keys())
        df = pd.DataFrame(index=classifiers, columns=metrics)
        
        # Fill in metrics for each classifier
        for cls in classifiers:
            for m in metrics:
                value = results[cls].get(m, 0.0)
                df.loc[cls, m] = value
        
        # Round values
        df = df.round(2)
        
        # Rename columns for better readability
        df = df.rename(columns=metric_names)
        
        # Save table to CSV
        os.makedirs('output', exist_ok=True)
        csv_path = os.path.join('output', f'model_summary_{table_num}.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved summary table to {csv_path}")
        
        # Create a nice visualization
        # Figure size based on table dimensions
        rows, cols = df.shape
        fig_height = max(rows * 0.5 + 2, 6)  # Minimum height of 6 inches
        fig_width = max(cols * 1.5 + 2, 8)  # Minimum width of 8 inches
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Replace NaN values with '0.00' for visualization
        df_display = df.copy()
        df_display = df_display.fillna(0.0)
        
        # Create the table
        table = ax.table(
            cellText=df_display.values,
            rowLabels=df_display.index,
            colLabels=df_display.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(title, fontsize=14, y=1.02)
        
        # Save the figure to output directory
        plt.tight_layout()
        output_path = os.path.join('output', 'model_comparison_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary table visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating summary table: {str(e)}")

def compare(args, logger):
    """Compare performance of multiple classifier models."""
    try:
        # Start timer
        start_time = time.time()
        
        # Check if CNN is specified in classifiers
        include_cnn = 'cnn' in args.classifiers.lower() or args.classifiers.lower() == 'all'
        
        # Initialize dataset handler
        logger.info("Initializing dataset handler")
        dataset_handler = DatasetHandler(
            n_segments=args.n_segments,
            compactness=args.compactness,
            connectivity_threshold=args.connectivity_threshold,
            max_images_per_class=args.max_images_per_class
        )
        
        # Process dataset
        logger.info(f"Processing dataset from {args.melanoma_dir} and {args.benign_dir}")
        graphs, labels = dataset_handler.process_dataset(
            args.melanoma_dir,
            args.benign_dir
        )
        
        # Split dataset
        train_graphs, test_graphs, train_labels, test_labels = dataset_handler.split_dataset(
            graphs, labels, test_size=0.2, random_state=42
        )
        
        logger.info(f"Dataset split: {len(train_graphs)} training samples, {len(test_graphs)} test samples")
        
        # Gather all image paths for potential CNN training
        all_image_paths = []
        all_image_labels = []
        if include_cnn:
            logger.info("Gathering image paths for CNN training")
            melanoma_paths = glob.glob(os.path.join(args.melanoma_dir, "*.jpg")) + \
                           glob.glob(os.path.join(args.melanoma_dir, "*.png")) + \
                           glob.glob(os.path.join(args.melanoma_dir, "*.jpeg"))
                           
            benign_paths = glob.glob(os.path.join(args.benign_dir, "*.jpg")) + \
                          glob.glob(os.path.join(args.benign_dir, "*.png")) + \
                          glob.glob(os.path.join(args.benign_dir, "*.jpeg"))
            
            # Limit number of images if needed
            if args.max_images_per_class > 0:
                np.random.seed(42)  # For reproducibility
                if len(melanoma_paths) > args.max_images_per_class:
                    melanoma_paths = np.random.choice(melanoma_paths, args.max_images_per_class, replace=False).tolist()
                if len(benign_paths) > args.max_images_per_class:
                    benign_paths = np.random.choice(benign_paths, args.max_images_per_class, replace=False).tolist()
            
            # Combine paths and create labels
            all_image_paths = melanoma_paths + benign_paths
            all_image_labels = np.array([1] * len(melanoma_paths) + [0] * len(benign_paths))
            
            # Shuffle data
            indices = np.arange(len(all_image_paths))
            np.random.seed(42)
            np.random.shuffle(indices)
            all_image_paths = np.array(all_image_paths)[indices].tolist()
            all_image_labels = all_image_labels[indices]
            
            # Split for CNN
            from sklearn.model_selection import train_test_split
            cnn_train_paths, cnn_test_paths, cnn_train_labels, cnn_test_labels = train_test_split(
                all_image_paths, all_image_labels, test_size=0.2, random_state=42, stratify=all_image_labels
            )
            
            logger.info(f"CNN dataset: {len(cnn_train_paths)} training, {len(cnn_test_paths)} testing images")
        
        # Determine which classifiers to train
        selected_classifiers = {}
        if args.classifiers.lower() == 'all':
            selected_classifiers = CLASSIFIERS
            logger.info("Training all traditional classifiers")
        else:
            # Map from command-line names to actual classifier names
            classifier_map = {
                'svm_rbf': 'SVM (RBF)',
                'svm_sigmoid': 'SVM (Sigmoid)',
                'svm_poly': 'SVM (Poly)',
                'knn': 'KNN',
                'mlp': 'MLP',
                'rf': 'RF'
            }
            
            for clf_name in args.classifiers.split(','):
                clf_name = clf_name.strip().lower()
                if clf_name == 'cnn':
                    # CNN is handled separately
                    continue
                if clf_name in classifier_map and classifier_map[clf_name] in CLASSIFIERS:
                    selected_classifiers[classifier_map[clf_name]] = CLASSIFIERS[classifier_map[clf_name]]
                elif clf_name in CLASSIFIERS:
                    selected_classifiers[clf_name] = CLASSIFIERS[clf_name]
            
            if not selected_classifiers and not include_cnn:
                logger.warning(f"No valid classifiers found in '{args.classifiers}'. Using all classifiers.")
                selected_classifiers = CLASSIFIERS
            else:
                logger.info(f"Training selected classifiers: {', '.join(selected_classifiers.keys())}")
        
        # Results to store all metrics
        results = {}
        
        # Use a MelanomaClassifier to prepare the features for traditional models
        if selected_classifiers:
            temp_classifier = MelanomaClassifier()
            logger.info("Preparing features for traditional models...")
            X_train = temp_classifier.prepare_features(train_graphs)
            X_test = temp_classifier.prepare_features(test_graphs)
            logger.info(f"Feature matrix shape: {X_train.shape} with {X_train.shape[1]} features")
            
            # Initialize scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate each traditional classifier
            for classifier_name, classifier_info in selected_classifiers.items():
                logger.info(f"Processing classifier: {classifier_name}")
                
                # Create model directories
                model_subdir = os.path.join('model', classifier_name.replace(' ', '_').replace('(', '').replace(')', ''))
                os.makedirs(model_subdir, exist_ok=True)
                
                # Define model and scaler paths
                model_path = os.path.join(model_subdir, "model.joblib")
                scaler_path = os.path.join(model_subdir, "scaler.joblib")
                
                try:
                    # Initialize classifier
                    clf = classifier_info['class'](**classifier_info['params'])
                    
                    # Train classifier
                    start_training = time.time()
                    logger.info(f"Training {classifier_name}...")
                    clf.fit(X_train_scaled, train_labels)
                    training_time = time.time() - start_training
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    
                    # Save model and scaler
                    dump(clf, model_path)
                    dump(scaler, scaler_path)
                    logger.info(f"Model saved to {model_path}")
                    
                    # Evaluate model
                    logger.info(f"Evaluating {classifier_name}...")
                    y_pred = clf.predict(X_test_scaled)
                    
                    # Get prediction probabilities if available
                    y_proba = None
                    if hasattr(clf, 'predict_proba'):
                        try:
                            probs = clf.predict_proba(X_test_scaled)
                            if probs.shape[1] >= 2:  # Binary classification
                                y_proba = probs[:, 1]  # Probability of positive class
                        except:
                            logger.warning(f"Could not get probabilities for {classifier_name}")
                    
                    # Calculate metrics
                    metrics = {}
                    metrics['AC'] = accuracy_score(test_labels, y_pred) * 100
                    metrics['SN'] = recall_score(test_labels, y_pred, zero_division=0) * 100
                    metrics['SP'] = specificity_score(test_labels, y_pred) * 100
                    
                    if y_proba is not None:
                        try:
                            metrics['AUC'] = roc_auc_score(test_labels, y_proba) * 100
                        except:
                            metrics['AUC'] = 50.0  # Default for random classifier
                    else:
                        metrics['AUC'] = 50.0
                    
                    # Log results
                    logger.info(f"Results for {classifier_name}:")
                    logger.info(f"  Accuracy: {metrics['AC']:.2f}%")
                    logger.info(f"  AUC: {metrics['AUC']:.2f}%")
                    logger.info(f"  Sensitivity: {metrics['SN']:.2f}%")
                    logger.info(f"  Specificity: {metrics['SP']:.2f}%")
                    
                    # Store results
                    results[classifier_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error processing {classifier_name}: {str(e)}")
        
        # Train and evaluate CNN if requested
        if include_cnn:
            logger.info("Processing CNN classifier")
            
            try:
                # Configure input shape
                input_shape = (args.input_size, args.input_size, 3)
                
                # Create model directory
                model_subdir = os.path.join('model', 'CNN')
                os.makedirs(model_subdir, exist_ok=True)
                
                # Initialize the CNN classifier
                cnn_classifier = CNNMelanomaClassifier(
                    model_type=args.cnn_model,
                    input_shape=input_shape
                )
                
                # Train the model
                start_training = time.time()
                logger.info(f"Training CNN with {args.cnn_model} architecture...")
                history = cnn_classifier.fit(
                    cnn_train_paths, cnn_train_labels,
                    X_val=cnn_test_paths, y_val=cnn_test_labels,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    unfreeze_layers=10,  # Unfreeze some layers for fine-tuning
                    fine_tune_epochs=20,
                    model_path=os.path.join(model_subdir, "cnn_model.h5")
                )
                training_time = time.time() - start_training
                logger.info(f"CNN training completed in {training_time:.2f} seconds")
                
                # Plot training history
                cnn_classifier.plot_training_history(save_path='output/cnn_training_history.png')
                
                # Evaluate the model
                logger.info("Evaluating CNN model...")
                eval_results = cnn_classifier.evaluate(cnn_test_paths, cnn_test_labels)
                
                # Calculate metrics
                metrics = {}
                metrics['AC'] = eval_results['accuracy'] * 100
                metrics['SN'] = eval_results['sensitivity'] * 100
                metrics['SP'] = eval_results['specificity'] * 100
                metrics['AUC'] = eval_results['auc'] * 100
                
                # Log results
                logger.info(f"Results for CNN ({args.cnn_model}):")
                logger.info(f"  Accuracy: {metrics['AC']:.2f}%")
                logger.info(f"  AUC: {metrics['AUC']:.2f}%")
                logger.info(f"  Sensitivity: {metrics['SN']:.2f}%")
                logger.info(f"  Specificity: {metrics['SP']:.2f}%")
                
                # Store results
                results[f"CNN ({args.cnn_model})"] = metrics
                
            except Exception as e:
                logger.error(f"Error processing CNN: {str(e)}")
        
        # Generate summary table
        if results:
            # Generate a single comprehensive model comparison table
            generate_summary_table(
                results, 
                logger, 
                table_num=1, 
                title="Melanoma Detection Model Comparison Summary"
            )
        else:
            logger.error("No results to display. All classifiers failed.")
        
        # Report time taken
        elapsed_time = time.time() - start_time
        logger.info(f"Model comparison completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
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
    os.makedirs('output/images', exist_ok=True)
    
    logger.info(f"Running in {args.mode} mode with classifier '{args.classifier}'")
    
    # Display CNN configuration if applicable
    if args.classifier == 'cnn' or 'cnn' in args.classifiers.lower():
        logger.info(f"CNN configuration: {args.cnn_model} architecture, " +
                   f"{args.input_size}x{args.input_size} input size, " +
                   f"{args.epochs} epochs, batch size {args.batch_size}")
    
    # Execute based on mode
    if args.mode == 'train':
        train(args, logger)
    elif args.mode == 'evaluate':
        evaluate(args, logger)
    elif args.mode == 'compare':
        compare(args, logger)

if __name__ == "__main__":
    main()