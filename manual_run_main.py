import os
import sys
import argparse
import logging
import time
import traceback
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, roc_auc_score,
    precision_score, f1_score, roc_curve, precision_recall_curve, auc,
    brier_score_loss, calibration_curve
)
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.dataset_handler import DatasetHandler
from src.classifier import MelanomaClassifier
# Import CNN conditionally to avoid errors when just showing help
# We'll import it only when needed in the functions

# Dictionary of available classifiers
CLASSIFIERS = {
    'SVM (RBF)': {
        'class': SVC,
        'params': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    },
    'SVM (Sigmoid)': {
        'class': SVC,
        'params': {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    },
    'SVM (Poly)': {
        'class': SVC,
        'params': {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    },
    'RF': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    },
    'MLP': {
        'class': MLPClassifier,
        'params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                 'learning_rate': 'adaptive', 'max_iter': 200, 'random_state': 42}
    },
    'KNN': {
        'class': KNeighborsClassifier,
        'params': {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto', 'p': 2}
    }
}

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

    # Single image classification parameters
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to a single image to classify (used in classify mode)')
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'classify'], default='train',
                        help='Operation mode: train (train models) or classify (single image)')

    return parser.parse_args()

def train(args, logger):
    """Train one or more melanoma detection models based on specified classifiers with enhanced feature selection and evaluation."""
    try:
        # Start timer
        start_time = time.time()

        # Check if CNN is specified in classifiers (TEMPORARILY DISABLED)
        include_cnn = False  # Set to False to disable CNN training
        # include_cnn = 'cnn' in args.classifiers.lower() or args.classifiers.lower() == 'all'

        # Initialize dataset handler for graph-based classifiers
        logger.info("Initializing dataset handler")
        dataset_handler = DatasetHandler(
            n_segments=args.n_segments,
            compactness=args.compactness,
            connectivity_threshold=args.connectivity_threshold,
            max_images_per_class=args.max_images_per_class
        )

        # Variables for CNN training
        cnn_train_paths = None
        cnn_train_labels = None
        cnn_test_paths = None
        cnn_test_labels = None

        # Gather all image paths for potential CNN training
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

        # Determine which traditional classifiers to train
        selected_classifiers = {}
        include_traditional = True

        if args.classifiers.lower() == 'all':
            selected_classifiers = CLASSIFIERS
            logger.info("Training all traditional classifiers")
        elif args.classifiers.lower() == 'cnn':
            include_traditional = False
            logger.info("Training only CNN classifier")
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
                include_traditional = True
            elif len(selected_classifiers) > 0:
                logger.info(f"Training selected classifiers: {', '.join(selected_classifiers.keys())}")

        # Results to store all metrics
        results = {}

        # Process data and train traditional classifiers if needed
        if include_traditional:
            # Process dataset for graph-based features
            logger.info(f"Processing dataset from {args.melanoma_dir} and {args.benign_dir}")
            graphs, labels = dataset_handler.process_dataset(
                args.melanoma_dir,
                args.benign_dir
            )

            # Save feature matrix for analysis
            logger.info("Creating and saving feature matrix...")
            feature_matrix = dataset_handler.save_feature_matrix(graphs, labels)
            logger.info(f"Feature matrix shape: {feature_matrix.shape}")

            # Split dataset
            train_graphs, test_graphs, train_labels, test_labels = dataset_handler.split_dataset(
                graphs, labels, test_size=0.2, random_state=42
            )

            logger.info(f"Dataset split: {len(train_graphs)} training samples, {len(test_graphs)} test samples")

            # Log total dataset size and class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.info(f"Dataset loaded: {len(graphs)} total samples")
            for label, count in zip(unique_labels, counts):
                label_name = "Melanoma" if label == 1 else "Benign"
                logger.info(f"  {label_name}: {count} samples")
            
            if len(graphs) < 20:
                logger.warning("Very small dataset detected! Results may not be reliable.")
                logger.warning("Recommended minimum: 100 samples per class")
            elif len(graphs) < 100:
                logger.warning("Small dataset detected. Consider adding more training data for better results.")

            # Use our optimized MelanomaClassifier to prepare the features
            logger.info("Preparing features with optimized extraction methods...")
            melanoma_classifier = MelanomaClassifier()
            X_train = melanoma_classifier.prepare_features(train_graphs)
            X_test = melanoma_classifier.prepare_features(test_graphs)
            logger.info(f"Feature matrix shape: {X_train.shape} with {X_train.shape[1]} features")

            # Train and evaluate each traditional classifier
            for classifier_name, classifier_info in selected_classifiers.items():
                logger.info(f"Processing classifier: {classifier_name}")

                # Create model directories using standardized naming
                if classifier_name == 'SVM (RBF)':
                    model_dir_name = 'SVM_RBF'
                    classifier_type = 'svm_rbf'
                elif classifier_name == 'SVM (Sigmoid)':
                    model_dir_name = 'SVM_Sigmoid'
                    classifier_type = 'svm_sigmoid'
                elif classifier_name == 'SVM (Poly)':
                    model_dir_name = 'SVM_Poly'
                    classifier_type = 'svm_poly'
                elif classifier_name == 'RF':
                    model_dir_name = 'RF'
                    classifier_type = 'rf'
                elif classifier_name == 'KNN':
                    model_dir_name = 'KNN'
                    classifier_type = 'knn'
                elif classifier_name == 'MLP':
                    model_dir_name = 'MLP'
                    classifier_type = 'mlp'
                else:
                    model_dir_name = classifier_name.replace(' ', '_').replace('(', '').replace(')', '')
                    classifier_type = model_dir_name.lower()

                model_subdir = os.path.join('model', model_dir_name)
                os.makedirs(model_subdir, exist_ok=True)
                logger.info(f"Saving model to directory: {model_subdir}")

                try:
                    # Use our enhanced MelanomaClassifier with all optimizations
                    logger.info(f"Initializing {classifier_name} with optimized parameters")
                    classifier = MelanomaClassifier(classifier_type=classifier_type)
                    
                    # Scale features
                    logger.info("Scaling features...")
                    X_train_scaled = classifier.scaler.fit_transform(X_train)
                    X_test_scaled = classifier.scaler.transform(X_test)
                    
                    # Select features using the enhanced method
                    logger.info("Performing feature selection...")
                    feature_selection_method = 'combined'  # Use both mutual info and f-test
                    X_train_selected = classifier.select_features(X_train_scaled, train_labels, 
                                                               method=feature_selection_method,
                                                               n_features=min(100, X_train.shape[1]))
                    
                    # Transform test data with the same feature selector
                    X_test_selected = classifier.feature_selector.transform(X_test_scaled)
                    
                    logger.info(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")

                    # Train classifier with cross-validation
                    start_training = time.time()
                    logger.info(f"Training {classifier_name} with cross-validation...")
                    
                    # Optimize hyperparameters if dataset is large enough
                    if len(train_graphs) >= 50:
                        logger.info("Optimizing hyperparameters...")
                        # Convert labels to integers for bincount
                        train_labels_int = train_labels.astype(int)
                        cv_value = min(5, min(np.bincount(train_labels_int)))
                        logger.info(f"Using {cv_value}-fold cross-validation for hyperparameter optimization")
                        classifier.optimize_hyperparameters(X_train_selected, train_labels, cv=cv_value)
                    
                    # Train and evaluate with cross-validation
                    # Convert labels to integers for bincount
                    train_labels_int = train_labels.astype(int)
                    cv_value = min(5, min(np.bincount(train_labels_int)))
                    logger.info(f"Using {cv_value}-fold cross-validation for evaluation")
                    cv_results = classifier.train_evaluate(X_train_selected, train_labels, cv=cv_value)
                    
                    training_time = time.time() - start_training
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    
                    # Log cross-validation results
                    logger.info(f"Cross-validation results:")
                    logger.info(f"  CV Accuracy: {cv_results['accuracy']:.2f}% (±{cv_results['std_accuracy']:.2f})")
                    logger.info(f"  CV Precision: {cv_results['precision']:.2f}% (±{cv_results['std_precision']:.2f})")
                    logger.info(f"  CV Recall/Sensitivity: {cv_results['recall']:.2f}% (±{cv_results['std_recall']:.2f})")
                    logger.info(f"  CV F1 Score: {cv_results['f1']:.2f}% (±{cv_results['std_f1']:.2f})")
                    
                    # Final evaluation on test set
                    logger.info("Evaluating on test set...")
                    y_pred = classifier.classifier.predict(X_test_selected)
                    y_proba = classifier.classifier.predict_proba(X_test_selected)[:, 1] if hasattr(classifier.classifier, 'predict_proba') else None
                    
                    # Calculate metrics
                    metrics = {}
                    metrics['AC'] = accuracy_score(test_labels, y_pred) * 100
                    metrics['SN'] = recall_score(test_labels, y_pred, zero_division=0) * 100
                    metrics['SP'] = specificity_score(test_labels, y_pred) * 100
                    metrics['PR'] = precision_score(test_labels, y_pred, zero_division=0) * 100
                    metrics['F1'] = f1_score(test_labels, y_pred, zero_division=0) * 100
                    metrics['NUM_FEATURES'] = X_train.shape[1]  # Total features
                    metrics['NUM_SELECTED'] = X_train_selected.shape[1]  # Selected features
                    metrics['FEATURE_REDUCTION'] = ((X_train.shape[1] - X_train_selected.shape[1]) / X_train.shape[1]) * 100
                    
                    if y_proba is not None:
                        try:
                            metrics['AUC'] = roc_auc_score(test_labels, y_proba) * 100
                            
                            # Generate ROC curve
                            fpr, tpr, _ = roc_curve(test_labels, y_proba)
                            
                            # Create ROC curve plot
                            plt.figure(figsize=(8, 6))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["AUC"]:.2f}%)')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title(f'ROC Curve for {classifier_name}')
                            plt.legend(loc="lower right")
                            
                            # Save ROC curve
                            roc_curve_path = os.path.join(model_subdir, "roc_curve.png")
                            plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"ROC curve saved to {roc_curve_path}")
                            
                            # Generate precision-recall curve
                            precision, recall, _ = precision_recall_curve(test_labels, y_proba)
                            pr_auc = auc(recall, precision)
                            metrics['PR_AUC'] = pr_auc * 100
                            
                            # Create precision-recall curve plot
                            plt.figure(figsize=(8, 6))
                            plt.plot(recall, precision, color='green', lw=2, 
                                   label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.ylim([0.0, 1.05])
                            plt.xlim([0.0, 1.0])
                            plt.title(f'Precision-Recall Curve for {classifier_name}')
                            plt.legend(loc="lower left")
                            
                            # Save precision-recall curve
                            pr_curve_path = os.path.join(model_subdir, "precision_recall_curve.png")
                            plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"Precision-Recall curve saved to {pr_curve_path}")
                            
                            # Create output directory for additional visualizations
                            output_dir = os.path.join('output', 'images', model_dir_name)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Generate and save learning curve
                            logger.info(f"Generating learning curve for {classifier_name}...")
                            learning_curve_path = os.path.join(output_dir, "learning_curve.png")
                            plot_learning_curve(
                                classifier.classifier, 
                                X_train_selected, 
                                train_labels,
                                cv=cv_value,
                                title=f"Learning Curve for {classifier_name}",
                                save_path=learning_curve_path
                            )
                            logger.info(f"Learning curve saved to {learning_curve_path}")
                            
                            # Generate and save calibration curve
                            logger.info(f"Generating calibration curve for {classifier_name}...")
                            calibration_curve_path = os.path.join(output_dir, "calibration_curve.png")
                            plot_calibration_curve(
                                classifier.classifier,
                                X_test_selected,
                                test_labels,
                                name=classifier_name,
                                save_path=calibration_curve_path
                            )
                            logger.info(f"Calibration curve saved to {calibration_curve_path}")
                            
                            # Generate and save prediction histogram
                            logger.info(f"Generating prediction histogram for {classifier_name}...")
                            pred_hist_path = os.path.join(output_dir, "prediction_histogram.png")
                            plot_prediction_histogram(
                                classifier.classifier,
                                X_test_selected,
                                test_labels,
                                save_path=pred_hist_path
                            )
                            logger.info(f"Prediction histogram saved to {pred_hist_path}")
                            
                            # Generate and save F1 score vs. threshold curve
                            logger.info(f"Generating F1 score curve for {classifier_name}...")
                            f1_curve_path = os.path.join(output_dir, "f1_threshold_curve.png")
                            plot_f1_threshold_curve(
                                classifier.classifier,
                                X_test_selected,
                                test_labels,
                                save_path=f1_curve_path
                            )
                            logger.info(f"F1 score curve saved to {f1_curve_path}")
                            
                            # Generate and save feature importance plot if the classifier supports it
                            logger.info(f"Generating feature importance plot for {classifier_name}...")
                            feature_importance_path = os.path.join(output_dir, "feature_importance.png")
                            try:
                                plot_feature_importance(
                                    classifier.classifier,
                                    X_test_selected,
                                    test_labels,
                                    feature_names=[f"Feature {i}" for i in range(X_test_selected.shape[1])],
                                    top_n=min(20, X_test_selected.shape[1]),
                                    save_path=feature_importance_path
                                )
                                logger.info(f"Feature importance plot saved to {feature_importance_path}")
                            except Exception as imp_err:
                                logger.warning(f"Could not generate feature importance plot: {str(imp_err)}")
                            
                        except Exception as curve_err:
                            logger.warning(f"Error generating curves: {str(curve_err)}")
                            metrics['AUC'] = 50.0
                            metrics['PR_AUC'] = 50.0
                    else:
                        metrics['AUC'] = 50.0
                        metrics['PR_AUC'] = 50.0
                    
                    # Log test set results
                    logger.info(f"Test set results for {classifier_name}:")
                    logger.info(f"  Accuracy: {metrics['AC']:.2f}%")
                    logger.info(f"  AUC: {metrics['AUC']:.2f}%")
                    logger.info(f"  Sensitivity/Recall: {metrics['SN']:.2f}%")
                    logger.info(f"  Specificity: {metrics['SP']:.2f}%")
                    logger.info(f"  Precision: {metrics['PR']:.2f}%")
                    logger.info(f"  F1 Score: {metrics['F1']:.2f}%")
                    if 'PR_AUC' in metrics:
                        logger.info(f"  PR-AUC: {metrics['PR_AUC']:.2f}%")
                    
                    # Log feature information
                    logger.info(f"Feature information:")
                    logger.info(f"  Total features: {metrics['NUM_FEATURES']}")
                    logger.info(f"  Selected features: {metrics['NUM_SELECTED']}")
                    logger.info(f"  Feature reduction: {metrics['FEATURE_REDUCTION']:.2f}%")
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(test_labels, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    logger.info(f"  Confusion Matrix: [[{tn} {fp}], [{fn} {tp}]]")
                    
                    # Generate confusion matrix plot
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title(f'Confusion Matrix for {classifier_name}')
                    plt.colorbar()
                    plt.xticks([0, 1], ['Benign', 'Melanoma'])
                    plt.yticks([0, 1], ['Benign', 'Melanoma'])
                    
                    # Add text annotations to the confusion matrix
                    thresh = cm.max() / 2
                    for i in range(2):
                        for j in range(2):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")
                    
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    # Save confusion matrix
                    cm_path = os.path.join(model_subdir, "confusion_matrix.png")
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Confusion matrix saved to {cm_path}")
                    
                    # Store results
                    results[classifier_name] = metrics
                    
                    # Save training summary to a text file
                    os.makedirs(os.path.join('output', 'summaries'), exist_ok=True)
                    summary_path = os.path.join('output', 'summaries', f"{model_dir_name}_training_summary.txt")
                    with open(summary_path, 'w') as f:
                        f.write(f"Training Summary for {classifier_name}\n")
                        f.write(f"==================================\n\n")
                        f.write(f"Performance Metrics:\n")
                        f.write(f"  Accuracy: {metrics['AC']:.2f}%\n")
                        f.write(f"  AUC: {metrics['AUC']:.2f}%\n")
                        f.write(f"  Sensitivity/Recall: {metrics['SN']:.2f}%\n")
                        f.write(f"  Specificity: {metrics['SP']:.2f}%\n")
                        f.write(f"  Precision: {metrics['PR']:.2f}%\n")
                        f.write(f"  F1 Score: {metrics['F1']:.2f}%\n")
                        if 'PR_AUC' in metrics:
                            f.write(f"  PR-AUC: {metrics['PR_AUC']:.2f}%\n\n")
                        f.write(f"Feature Information:\n")
                        f.write(f"  Total features: {metrics['NUM_FEATURES']}\n")
                        f.write(f"  Selected features: {metrics['NUM_SELECTED']}\n")
                        f.write(f"  Feature reduction: {metrics['FEATURE_REDUCTION']:.2f}%\n\n")
                        f.write(f"Confusion Matrix:\n")
                        f.write(f"  [[{tn} {fp}]\n   [{fn} {tp}]]\n\n")
                        f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    logger.info(f"Training summary saved to {summary_path}")
                    
                    # Save model, scaler, and feature selector
                    model_path = os.path.join(model_subdir, "model.joblib")
                    scaler_path = os.path.join(model_subdir, "scaler.joblib")
                    feature_selector_path = os.path.join(model_subdir, "feature_selector.joblib")
                    
                    dump(classifier.classifier, model_path)
                    dump(classifier.scaler, scaler_path)
                    dump(classifier.feature_selector, feature_selector_path)
                    
                    logger.info(f"Model saved to {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error training {classifier_name}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Train CNN if requested
        if include_cnn and cnn_train_paths is not None:
            logger.info("Training CNN classifier")

            try:
                # Import CNN module only when needed
                from src.cnn_classifier import CNNMelanomaClassifier
                
                # Create model directory
                model_subdir = os.path.join('model', 'CNN')
                os.makedirs(model_subdir, exist_ok=True)

                # Configure input shape and training parameters
                input_shape = (args.input_size, args.input_size, 3)
                epochs = args.epochs
                batch_size = args.batch_size

                # Initialize the CNN classifier
                cnn_classifier = CNNMelanomaClassifier(
                    model_type=args.cnn_model,
                    input_shape=input_shape
                )

                # Train the CNN
                start_training = time.time()
                logger.info(f"Training CNN ({args.cnn_model})...")
                history = cnn_classifier.fit(
                    cnn_train_paths, cnn_train_labels,
                    X_val=cnn_test_paths, y_val=cnn_test_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    model_path=os.path.join(model_subdir, "cnn_model.h5")
                )
                training_time = time.time() - start_training
                logger.info(f"CNN training completed in {training_time:.2f} seconds")

                # Evaluate the CNN
                logger.info("Evaluating CNN model...")
                eval_results = cnn_classifier.evaluate(cnn_test_paths, cnn_test_labels)

                # Calculate metrics
                metrics = {}
                metrics['AC'] = eval_results['accuracy'] * 100
                metrics['SN'] = eval_results['sensitivity'] * 100
                metrics['SP'] = eval_results['specificity'] * 100
                metrics['PR'] = eval_results['precision'] * 100 if 'precision' in eval_results else 0
                metrics['F1'] = eval_results['f1'] * 100 if 'f1' in eval_results else 0
                metrics['AUC'] = eval_results['auc'] * 100
                
                # Add feature information if available from CNN architecture
                if hasattr(cnn_classifier, 'feature_counts'):
                    metrics['NUM_FEATURES'] = cnn_classifier.feature_counts.get('total', 0)
                    metrics['NUM_SELECTED'] = cnn_classifier.feature_counts.get('used', 0)
                    if metrics['NUM_FEATURES'] > 0:
                        metrics['FEATURE_REDUCTION'] = ((metrics['NUM_FEATURES'] - metrics['NUM_SELECTED']) / metrics['NUM_FEATURES']) * 100
                    else:
                        metrics['FEATURE_REDUCTION'] = 0
                
                # Log comprehensive results
                logger.info(f"Results for CNN ({args.cnn_model}):")
                logger.info(f"  Accuracy: {metrics['AC']:.2f}%")
                logger.info(f"  AUC: {metrics['AUC']:.2f}%")
                logger.info(f"  Sensitivity/Recall: {metrics['SN']:.2f}%")
                logger.info(f"  Specificity: {metrics['SP']:.2f}%")
                logger.info(f"  Precision: {metrics['PR']:.2f}%")
                logger.info(f"  F1 Score: {metrics['F1']:.2f}%")
                
                # Store results
                results[f"CNN ({args.cnn_model})"] = metrics
                
                # Save detailed evaluation results to a text file
                os.makedirs(os.path.join('output', 'summaries'), exist_ok=True)
                summary_path = os.path.join('output', 'summaries', f"CNN_{args.cnn_model}_training_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"CNN ({args.cnn_model}) Training Summary\n")
                    f.write(f"{'=' * 40}\n\n")
                    f.write(f"Performance Metrics:\n")
                    f.write(f"  Accuracy: {metrics['AC']:.2f}%\n")
                    f.write(f"  AUC: {metrics['AUC']:.2f}%\n")
                    f.write(f"  Sensitivity/Recall: {metrics['SN']:.2f}%\n")
                    f.write(f"  Specificity: {metrics['SP']:.2f}%\n")
                    f.write(f"  Precision: {metrics['PR']:.2f}%\n")
                    f.write(f"  F1 Score: {metrics['F1']:.2f}%\n\n")
                    
                    if 'NUM_FEATURES' in metrics:
                        f.write(f"Architecture Information:\n")
                        f.write(f"  Model Type: {args.cnn_model}\n")
                        f.write(f"  Input Size: {args.input_size}x{args.input_size}\n")
                        f.write(f"  Total Parameters: {metrics['NUM_FEATURES']:,d}\n")
                        f.write(f"  Trainable Parameters: {metrics['NUM_SELECTED']:,d}\n")
                        f.write(f"  Parameter Reduction: {metrics['FEATURE_REDUCTION']:.2f}%\n\n")
                    
                    f.write(f"Training Configuration:\n")
                    f.write(f"  Batch Size: {batch_size}\n")
                    f.write(f"  Epochs: {epochs}\n")
                    f.write(f"  Training Images: {len(cnn_train_paths)}\n")
                    f.write(f"  Testing Images: {len(cnn_test_paths)}\n\n")
                    
                    f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                logger.info(f"CNN training summary saved to {summary_path}")
                
                # Plot training history
                output_images_dir = os.path.join('output', 'images')
                os.makedirs(output_images_dir, exist_ok=True)
                cnn_classifier.plot_training_history(save_path=os.path.join(output_images_dir, f"CNN_{args.cnn_model}_training_history.png"))
                
                # Generate confusion matrix from evaluation results if available
                if 'confusion_matrix' in eval_results:
                    cm = eval_results['confusion_matrix']
                    
                    # Plot confusion matrix
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title(f'Confusion Matrix for CNN ({args.cnn_model})')
                    plt.colorbar()
                    plt.xticks([0, 1], ['Benign', 'Melanoma'])
                    plt.yticks([0, 1], ['Benign', 'Melanoma'])
                    
                    # Add text annotations
                    thresh = cm.max() / 2
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
                    
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    # Save confusion matrix
                    output_images_dir = os.path.join('output', 'images')
                    os.makedirs(output_images_dir, exist_ok=True)
                    cm_path = os.path.join(output_images_dir, f"CNN_{args.cnn_model}_confusion_matrix.png")
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"CNN confusion matrix saved to {cm_path}")
                
                # Generate ROC curve if probabilities are available
                if 'y_true' in eval_results and 'y_prob' in eval_results:
                    y_true = eval_results['y_true']
                    y_prob = eval_results['y_prob']
                    
                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {metrics["AUC"]:.2f}%)')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve for CNN ({args.cnn_model})')
                    plt.legend(loc="lower right")
                    
                    # Save ROC curve
                    output_images_dir = os.path.join('output', 'images')
                    os.makedirs(output_images_dir, exist_ok=True)
                    roc_curve_path = os.path.join(output_images_dir, f"CNN_{args.cnn_model}_roc_curve.png")
                    plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"ROC curve saved to {roc_curve_path}")
                    
                    # Precision-recall curve
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    pr_auc = auc(recall, precision)
                    metrics['PR_AUC'] = pr_auc * 100
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, color='green', lw=2, 
                           label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
                           
                    # Create CNN-specific output directory for additional visualizations
                    output_dir = os.path.join('output', 'images', f'CNN_{args.cnn_model}')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate prediction histogram for CNN
                    logger.info(f"Generating prediction histogram for CNN ({args.cnn_model})...")
                    pred_hist_path = os.path.join(output_dir, "prediction_histogram.png")
                    plt.figure(figsize=(10, 6))
                    
                    # Separate probabilities by true class
                    y_prob_pos = y_prob[y_true == 1]
                    y_prob_neg = y_prob[y_true == 0]
                    
                    plt.hist(y_prob_pos, bins=20, alpha=0.6, color='red', 
                             label=f'True Melanoma (n={len(y_prob_pos)})')
                    plt.hist(y_prob_neg, bins=20, alpha=0.6, color='green', 
                             label=f'True Benign (n={len(y_prob_neg)})')
                    
                    plt.xlabel('Predicted Probability of Melanoma')
                    plt.ylabel('Count')
                    plt.title(f'Histogram of Prediction Probabilities for CNN ({args.cnn_model})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(pred_hist_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Prediction histogram saved to {pred_hist_path}")
                    
                    # Generate F1 score vs. threshold curve for CNN
                    logger.info(f"Generating F1 score curve for CNN ({args.cnn_model})...")
                    f1_curve_path = os.path.join(output_dir, "f1_threshold_curve.png")
                    plt.figure(figsize=(10, 6))
                    
                    # Calculate F1 score for different thresholds
                    thresholds = np.linspace(0, 1, 100)
                    f1_scores = []
                    precision_scores = []
                    recall_scores = []
                    
                    for threshold in thresholds:
                        y_pred_t = (y_prob >= threshold).astype(int)
                        f1 = f1_score(y_true, y_pred_t, zero_division=0)
                        precision = precision_score(y_true, y_pred_t, zero_division=0)
                        recall = recall_score(y_true, y_pred_t, zero_division=0)
                        
                        f1_scores.append(f1)
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                    
                    # Find threshold with best F1 score
                    best_threshold_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_threshold_idx]
                    best_f1 = f1_scores[best_threshold_idx]
                    
                    # Plot curves
                    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
                    plt.plot(thresholds, precision_scores, 'g-', label='Precision')
                    plt.plot(thresholds, recall_scores, 'r-', label='Recall')
                    
                    # Mark the best threshold
                    plt.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7)
                    plt.plot(best_threshold, best_f1, 'bo', markersize=8, 
                            label=f'Best Threshold = {best_threshold:.2f}, F1 = {best_f1:.2f}')
                    
                    plt.xlabel('Decision Threshold')
                    plt.ylabel('Score')
                    plt.title(f'F1 Score, Precision, and Recall vs. Decision Threshold for CNN ({args.cnn_model})')
                    plt.legend(loc='best')
                    plt.grid(True)
                    plt.savefig(f1_curve_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"F1 score curve saved to {f1_curve_path}")
                    
                    # Generate calibration curve for CNN
                    logger.info(f"Generating calibration curve for CNN ({args.cnn_model})...")
                    calibration_curve_path = os.path.join(output_dir, "calibration_curve.png")
                    plt.figure(figsize=(10, 6))
                    
                    # Calculate Brier score loss
                    brier_score = brier_score_loss(y_true, y_prob)
                    
                    # Calculate calibration curve
                    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
                    
                    # Plot perfectly calibrated
                    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
                    
                    # Plot model calibration curve
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                             label=f"CNN ({args.cnn_model}) (Brier score: {brier_score:.3f})")
                    
                    plt.ylabel("Fraction of positives (Empirical)")
                    plt.xlabel("Mean predicted probability (Model)")
                    plt.title(f'Calibration Curve for CNN ({args.cnn_model})')
                    plt.legend(loc="best")
                    plt.grid(True)
                    plt.savefig(calibration_curve_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Calibration curve saved to {calibration_curve_path}")
                    
                    # Continue with precision-recall curve (fixing the plot that was messed up in our edit)
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, color='green', lw=2, 
                           label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.title(f'Precision-Recall Curve for CNN ({args.cnn_model})')
                    plt.legend(loc="lower left")
                    
                    # Save precision-recall curve
                    output_images_dir = os.path.join('output', 'images')
                    os.makedirs(output_images_dir, exist_ok=True)
                    pr_curve_path = os.path.join(output_images_dir, f"CNN_{args.cnn_model}_precision_recall_curve.png")
                    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Precision-Recall curve saved to {pr_curve_path}")
                    
                    # Update results with PR-AUC
                    results[f"CNN ({args.cnn_model})"] = metrics

            except Exception as e:
                logger.error(f"Error training CNN: {str(e)}")
                traceback.print_exc()

        # Generate summary table if multiple classifiers were trained
        if len(results) > 1:
            generate_summary_table(results, logger, table_num=1, title="Melanoma Detection Model Training Summary")
        elif len(results) == 1:
            logger.info(f"Trained 1 classifier: {list(results.keys())[0]}")
        else:
            logger.warning("No classifiers were successfully trained.")

        # End timer
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def classify(args, logger):
    """Classify a single image using trained models with enhanced feature extraction."""
    try:
        # Start timer for benchmarking
        start_time = time.time()
        
        # Check if image path is provided
        if args.image_path is None or not os.path.exists(args.image_path):
            logger.error("Image path is required for classification and must exist")
            return
        
        logger.info(f"Classifying image: {args.image_path}")
        
        # Determine which classifiers to use
        selected_classifiers = {}
        include_cnn = False
        include_traditional = True
        
        if args.classifiers.lower() == 'all':
            # Get all classifier directories in the model folder
            model_dirs = [d for d in os.listdir('model') if os.path.isdir(os.path.join('model', d))]
            
            for d in model_dirs:
                if d.lower() == 'cnn':
                    include_cnn = True
                    continue
                    
                # Map directory names to classifier names
                if d == 'SVM_RBF':
                    classifier_name = 'SVM (RBF)'
                elif d == 'SVM_Sigmoid':
                    classifier_name = 'SVM (Sigmoid)'
                elif d == 'SVM_Poly':
                    classifier_name = 'SVM (Poly)'
                else:
                    classifier_name = d
                    
                # Check if model files exist
                model_path = os.path.join('model', d, 'model.joblib')
                scaler_path = os.path.join('model', d, 'scaler.joblib')
                feature_selector_path = os.path.join('model', d, 'feature_selector.joblib')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    selected_classifiers[classifier_name] = {
                        'model_path': model_path,
                        'scaler_path': scaler_path,
                        'feature_selector_path': feature_selector_path if os.path.exists(feature_selector_path) else None
                    }
            
            logger.info("Using all available classifiers")
            
        elif args.classifiers.lower() == 'cnn':
            include_traditional = False
            include_cnn = True
            logger.info("Using only CNN classifier")
            
        else:
            # Map from command-line names to classifier directories
            classifier_map = {
                'svm_rbf': 'SVM_RBF',
                'svm_sigmoid': 'SVM_Sigmoid',
                'svm_poly': 'SVM_Poly',
                'knn': 'KNN',
                'mlp': 'MLP',
                'rf': 'RF',
                'cnn': 'CNN'
            }
            
            for clf_name in args.classifiers.split(','):
                clf_name = clf_name.strip().lower()
                
                if clf_name == 'cnn':
                    include_cnn = True
                    continue
                    
                dir_name = classifier_map.get(clf_name, clf_name.upper())
                model_path = os.path.join('model', dir_name, 'model.joblib')
                scaler_path = os.path.join('model', dir_name, 'scaler.joblib')
                feature_selector_path = os.path.join('model', dir_name, 'feature_selector.joblib')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    # Map directory names to display names
                    if dir_name == 'SVM_RBF':
                        display_name = 'SVM (RBF)'
                    elif dir_name == 'SVM_Sigmoid':
                        display_name = 'SVM (Sigmoid)'
                    elif dir_name == 'SVM_Poly':
                        display_name = 'SVM (Poly)'
                    else:
                        display_name = dir_name
                        
                    selected_classifiers[display_name] = {
                        'model_path': model_path,
                        'scaler_path': scaler_path,
                        'feature_selector_path': feature_selector_path if os.path.exists(feature_selector_path) else None
                    }
                else:
                    logger.warning(f"Could not find model files for {clf_name}")
            
            if not selected_classifiers and not include_cnn:
                logger.error("No valid classifiers found. Train models first.")
                return
            else:
                logger.info(f"Using selected classifiers: {', '.join(selected_classifiers.keys())}")
        
        # Process the single image
        logger.info("Processing the image for classification")
        
        # Import necessary modules
        from src.preprocessing import ImagePreprocessor
        from src.image_validator import ImageValidator
        from src.superpixel import SuperpixelGenerator
        from src.graph_construction import GraphConstructor
        from src.feature_extraction import FeatureExtractor
        from src.conventional_features import ConventionalFeatureExtractor
        
        # Initialize components
        preprocessor = ImagePreprocessor()
        image_validator = ImageValidator()
        superpixel_gen = SuperpixelGenerator(
            n_segments=args.n_segments,
            compactness=args.compactness
        )
        graph_constructor = GraphConstructor(
            connectivity_threshold=args.connectivity_threshold
        )
        feature_extractor = FeatureExtractor()
        conv_feature_extractor = ConventionalFeatureExtractor()
        
        # Load and validate image
        logger.info("Loading and validating image")
        original_image = preprocessor.load_image(args.image_path)
        if original_image is None:
            logger.error("Failed to load image")
            return
            
        is_valid, validation_message = image_validator.validate_skin_image(original_image)
        if not is_valid:
            logger.error(f"Invalid skin image: {validation_message}")
            return
            
        # Preprocess image
        logger.info("Preprocessing image")
        processed_image = preprocessor.preprocess(original_image)
        
        # Generate superpixels
        logger.info("Generating superpixels")
        segments = superpixel_gen.generate_superpixels(processed_image)
        features = superpixel_gen.compute_superpixel_features(processed_image, segments)
        
        # Construct graph
        logger.info("Constructing graph representation")
        G = graph_constructor.build_graph(features, segments)
        
        # Extract graph features
        logger.info("Extracting features")
        local_features = feature_extractor.extract_local_features(G)
        global_features = feature_extractor.extract_global_features(G)
        spectral_features = feature_extractor.extract_spectral_features(G)
        
        # Calculate mask of the lesion
        lesion_mask = segments > -1
        
        # Extract conventional features
        logger.info("Extracting conventional image features")
        conventional_features = conv_feature_extractor.extract_all_features(original_image, lesion_mask)
        
        # Create graph object with features
        G.graph = {}
        G.graph['features'] = {
            **local_features,
            **global_features,
            **spectral_features
        }
        G.graph['conventional_features'] = conventional_features
        
        # Results dictionary to store predictions
        results = {}
        
        # Classify using traditional classifiers
        if include_traditional and selected_classifiers:
            # Create feature matrix
            temp_classifier = MelanomaClassifier()
            X = temp_classifier.prepare_features([G])
            
            # Classify with each traditional classifier
            for classifier_name, paths in selected_classifiers.items():
                logger.info(f"Classifying with {classifier_name}")
                
                try:
                    # Load model and scaler
                    model_path = paths['model_path']
                    scaler_path = paths['scaler_path']
                    feature_selector_path = paths.get('feature_selector_path')
                    
                    # Check if model exists
                    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                        logger.warning(f"Model files for {classifier_name} not found. Skipping classification.")
                        continue
                        
                    # Load classifier and scaler
                    clf = load(model_path)
                    scaler = load(scaler_path)
                    
                    # Scale features
                    X_scaled = scaler.transform(X)
                    
                    # Apply feature selection if available
                    if feature_selector_path and os.path.exists(feature_selector_path):
                        feature_selector = load(feature_selector_path)
                        X_selected = feature_selector.transform(X_scaled)
                        logger.info(f"Applied feature selection: using {X_selected.shape[1]} features")
                        
                        # Make prediction with selected features
                        pred = clf.predict(X_selected)[0]
                    else:
                        # Make prediction without feature selection
                        pred = clf.predict(X_scaled)[0]
                    
                    # Get probability if available
                    prob = None
                    if hasattr(clf, 'predict_proba'):
                        try:
                            # Use selected features for probability if available
                            if feature_selector_path and os.path.exists(feature_selector_path) and 'X_selected' in locals():
                                probs = clf.predict_proba(X_selected)
                                prob = probs[0, 1]  # Probability of positive class (melanoma)
                                logger.info(f"Using feature-selected data for probability calculation")
                            else:
                                probs = clf.predict_proba(X_scaled)
                                prob = probs[0, 1]  # Probability of positive class (melanoma)
                        except Exception as prob_err:
                            logger.warning(f"Could not get probability for {classifier_name}: {str(prob_err)}")
                            
                    # Determine risk level based on probability
                    risk_level = ""
                    explanation = ""
                    
                    if prob is not None:
                        prob_pct = prob * 100
                        if prob_pct > 75:
                            risk_level = "HIGH"
                            explanation = "High probability of melanoma. Immediate medical consultation recommended."
                        elif prob_pct > 50:
                            risk_level = "MODERATE TO HIGH"
                            explanation = "Elevated probability of melanoma. Prompt medical consultation recommended."
                        elif prob_pct > 25:
                            risk_level = "MODERATE"
                            explanation = "Some features of concern present. Medical evaluation advised."
                        else:
                            risk_level = "LOW"
                            explanation = "Low probability of melanoma. Regular self-examination advised."
                    
                    # Store result with enhanced information
                    results[classifier_name] = {
                        'prediction': 'Melanoma' if pred == 1 else 'Benign',
                        'probability': prob * 100 if prob is not None else None,
                        'risk_level': risk_level if prob is not None else "UNKNOWN",
                        'explanation': explanation if prob is not None else "Unable to assess risk level due to missing probability"
                    }
                    
                    # Log result
                    logger.info(f"{classifier_name} prediction: {results[classifier_name]['prediction']}")
                    if prob is not None:
                        logger.info(f"{classifier_name} probability: {prob * 100:.2f}%")
                        
                except Exception as e:
                    logger.error(f"Error classifying with {classifier_name}: {str(e)}")
        
        # Classify using CNN if requested
        if include_cnn:
            logger.info("Classifying with CNN")
            
            model_path = os.path.join('model', 'CNN', 'cnn_model.h5')
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"CNN model file not found at {model_path}. Skipping classification.")
            else:
                try:
                    # Import CNN module only when needed
                    from src.cnn_classifier import CNNMelanomaClassifier
                    
                    # Initialize the CNN classifier
                    cnn_classifier = CNNMelanomaClassifier(
                        model_type=args.cnn_model,
                        input_shape=(args.input_size, args.input_size, 3)
                    )
                    
                    # Load the model
                    cnn_classifier.load_model(model_path)
                    
                    # Make prediction
                    pred_prob = cnn_classifier.predict([args.image_path])[0]
                    pred_class = 1 if pred_prob > 0.5 else 0
                    
                    # Determine risk level based on probability
                    risk_level = ""
                    explanation = ""
                    
                    prob_pct = pred_prob * 100
                    if prob_pct > 75:
                        risk_level = "HIGH"
                        explanation = "High probability of melanoma. Immediate medical consultation recommended."
                    elif prob_pct > 50:
                        risk_level = "MODERATE TO HIGH"
                        explanation = "Elevated probability of melanoma. Prompt medical consultation recommended."
                    elif prob_pct > 25:
                        risk_level = "MODERATE"
                        explanation = "Some features of concern present. Medical evaluation advised."
                    else:
                        risk_level = "LOW"
                        explanation = "Low probability of melanoma. Regular self-examination advised."
                    
                    # Store result with enhanced information
                    results['CNN'] = {
                        'prediction': 'Melanoma' if pred_class == 1 else 'Benign',
                        'probability': pred_prob * 100,
                        'risk_level': risk_level,
                        'explanation': explanation
                    }
                    
                    # Log result
                    logger.info(f"CNN prediction: {results['CNN']['prediction']}")
                    logger.info(f"CNN probability: {pred_prob * 100:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error classifying with CNN: {str(e)}")
        
        # Display summary of results
        if results:
            logger.info("\nClassification Results Summary:")
            logger.info("-" * 60)
            logger.info(f"Image: {args.image_path}")
            logger.info("-" * 60)
            
            # Count predictions by class
            melanoma_count = sum(1 for r in results.values() if r['prediction'] == 'Melanoma')
            benign_count = sum(1 for r in results.values() if r['prediction'] == 'Benign')
            total_count = len(results)
            
            # Format results in a table
            logger.info(f"{'Classifier':<15} | {'Prediction':<10} | {'Probability':<12} | {'Risk Level':<15}")
            logger.info("-" * 80)
            
            for classifier_name, result in results.items():
                prob_str = f"{result['probability']:.2f}%" if result['probability'] is not None else "N/A"
                risk_level = result.get('risk_level', 'N/A')
                logger.info(f"{classifier_name:<15} | {result['prediction']:<10} | {prob_str:<12} | {risk_level:<15}")
                
                # Display explanation for risk level if available
                if 'explanation' in result:
                    logger.info(f"    └─ {result['explanation']}")
            
            logger.info("-" * 60)
            logger.info(f"Summary: {melanoma_count}/{total_count} classifiers predict Melanoma")
            logger.info(f"         {benign_count}/{total_count} classifiers predict Benign")
            
            # Overall majority prediction
            if melanoma_count > benign_count:
                logger.info("\nOverall prediction: MELANOMA (potentially malignant)")
            elif benign_count > melanoma_count:
                logger.info("\nOverall prediction: BENIGN (likely non-malignant)")
            else:
                logger.info("\nOverall prediction: INCONCLUSIVE (equal votes)")
            
            # Add calculated metrics (for comparison)
            logger.info("\nPerformance Metrics:")
            logger.info("-" * 60)
            logger.info("NOTE: These are not validation metrics, just individual model confidence indicators.")
            
            for classifier_name, result in results.items():
                if result['probability'] is not None:
                    # Calculate metrics based on probability thresholds
                    confidence = result['probability'] / 100 if result['prediction'] == 'Melanoma' else 1 - (result['probability'] / 100)
                    
                    logger.info(f"{classifier_name}:")
                    logger.info(f"  Confidence: {result['probability']:.2f}%")
                    logger.info(f"  Decision threshold: {'>.5' if result['prediction'] == 'Melanoma' else '≤.5'}")
            
            logger.info("\nNote: This is a research prototype. Always consult a medical professional.")
            
            # End timer
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Classification completed in {total_time:.2f} seconds")
        else:
            logger.error("No classification results were produced. Check the logs for errors.")
            
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        traceback.print_exc()

def specificity_score(y_true, y_pred):
    """Calculate the specificity score (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5), 
                       title="Learning Curve", save_path=None):
    """
    Generate and plot a learning curve for a classifier.
    
    Args:
        estimator: The classifier object
        X: Features
        y: Labels
        cv: Cross-validation folds
        n_jobs: Number of parallel jobs
        train_sizes: Array of training sizes to plot
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, 
        scoring='accuracy', shuffle=True, random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1) * 100
    train_scores_std = np.std(train_scores, axis=1) * 100
    test_scores_mean = np.mean(test_scores, axis=1) * 100
    test_scores_std = np.std(test_scores, axis=1) * 100
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def plot_calibration_curve(clf, X, y, name, save_path=None):
    """
    Generate and plot a calibration curve for a classifier.
    
    Args:
        clf: Trained classifier
        X: Features
        y: True labels
        name: Classifier name
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X)[:, 1]
        
        # Calculate Brier score loss
        brier_score = brier_score_loss(y, prob_pos)
        
        # Plot calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
        
        # Plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        
        # Plot model calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name} (Brier score: {brier_score:.3f})")
        
        plt.ylabel("Fraction of positives (Empirical)")
        plt.xlabel("Mean predicted probability (Model)")
        plt.title(f'Calibration Curve for {name}')
        plt.legend(loc="best")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    else:
        plt.text(0.5, 0.5, f"Classifier {name} does not support predict_proba",
                horizontalalignment='center', verticalalignment='center')
        plt.title("Calibration Curve Not Available")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def plot_feature_importance(clf, X, y, feature_names=None, top_n=20, save_path=None):
    """
    Generate and plot feature importance for a classifier.
    
    Args:
        clf: Trained classifier
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    importance_type = None
    importances = None
    
    # Random Forest and other tree ensembles have feature_importances_
    if hasattr(clf, 'feature_importances_'):
        importance_type = "Native Feature Importance"
        importances = clf.feature_importances_
    else:
        # Use permutation importance for models that don't have native feature importance
        importance_type = "Permutation Importance"
        result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
    
    if importances is not None:
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Get indices of top features
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot top features
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f'Top {top_n} Feature Importance ({importance_type})')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model",
                horizontalalignment='center', verticalalignment='center')
        plt.title("Feature Importance Not Available")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def plot_prediction_histogram(clf, X, y, save_path=None):
    """
    Plot histogram of prediction probabilities.
    
    Args:
        clf: Trained classifier
        X: Feature matrix
        y: True labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X)
        pos_probs = probas[:, 1]  # Probability of positive class
        
        # Separate probabilities by true class
        pos_probs_pos = pos_probs[y == 1]
        pos_probs_neg = pos_probs[y == 0]
        
        plt.hist(pos_probs_pos, bins=20, alpha=0.6, color='red', 
                 label=f'True Melanoma (n={len(pos_probs_pos)})')
        plt.hist(pos_probs_neg, bins=20, alpha=0.6, color='green', 
                 label=f'True Benign (n={len(pos_probs_neg)})')
        
        plt.xlabel('Predicted Probability of Melanoma')
        plt.ylabel('Count')
        plt.title('Histogram of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    else:
        plt.text(0.5, 0.5, "This model does not support predict_proba",
                horizontalalignment='center', verticalalignment='center')
        plt.title("Prediction Histogram Not Available")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def plot_f1_threshold_curve(clf, X, y, save_path=None):
    """
    Plot F1 score vs decision threshold.
    
    Args:
        clf: Trained classifier
        X: Feature matrix
        y: True labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(clf, "predict_proba"):
        y_scores = clf.predict_proba(X)[:, 1]
        
        # Calculate F1 score for different thresholds
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # Find threshold with best F1 score
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]
        
        # Plot curves
        plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
        plt.plot(thresholds, precision_scores, 'g-', label='Precision')
        plt.plot(thresholds, recall_scores, 'r-', label='Recall')
        
        # Mark the best threshold
        plt.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7)
        plt.plot(best_threshold, best_f1, 'bo', markersize=8, 
                label=f'Best Threshold = {best_threshold:.2f}, F1 = {best_f1:.2f}')
        
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('F1 Score, Precision, and Recall vs. Decision Threshold')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    else:
        plt.text(0.5, 0.5, "This model does not support predict_proba",
                horizontalalignment='center', verticalalignment='center')
        plt.title("F1 Score Curve Not Available")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def generate_summary_table(results, logger, table_num=1, title="Melanoma Detection Model Comparison Summary"):
    """
    Generate a clean summary table showing metrics for each classifier.

    Args:
        results: Dictionary containing results for all classifiers
        logger: Logger instance
        table_num: Table number (1-5) for saving to appropriate file
        title: Title for the table visualization
    """
    if not results:
        logger.warning("No results to generate summary table.")
        return

    # Create DataFrame for results
    df = pd.DataFrame(results).T
    
    # Reorder columns if they exist
    cols_order = ['AC', 'AUC', 'PR_AUC', 'SN', 'SP', 'PR', 'F1', 'NUM_FEATURES', 'NUM_SELECTED', 'FEATURE_REDUCTION']
    existing_cols = [col for col in cols_order if col in df.columns]
    df = df[existing_cols]
    
    # Rename columns for display
    column_names = {
        'AC': 'Accuracy (%)', 
        'AUC': 'AUC ROC (%)', 
        'PR_AUC': 'AUC PR (%)',
        'SN': 'Sensitivity (%)', 
        'SP': 'Specificity (%)',
        'PR': 'Precision (%)',
        'F1': 'F1 Score (%)',
        'NUM_FEATURES': 'Total Features',
        'NUM_SELECTED': 'Selected Features',
        'FEATURE_REDUCTION': 'Feature Reduction (%)'
    }
    
    df.columns = [column_names.get(col, col) for col in df.columns]
    
    # Display table in logger
    logger.info("\n" + title)
    logger.info("-" * 120)
    logger.info(df.to_string(float_format=lambda x: f"{x:.2f}"))
    logger.info("-" * 120)
    
    # Save results to CSV
    metrics_dir = os.path.join('output', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, f'model_comparison_results_{table_num}.csv')
    df.to_csv(csv_path, float_format='%.2f')
    logger.info(f"Results saved to CSV: {csv_path}")
    
    # Split into performance metrics and feature metrics for better visualization
    performance_metrics = ['Accuracy (%)', 'AUC ROC (%)', 'AUC PR (%)', 'Sensitivity (%)', 
                          'Specificity (%)', 'Precision (%)', 'F1 Score (%)']
    feature_metrics = ['Total Features', 'Selected Features', 'Feature Reduction (%)']
    
    # Filter columns that exist in our dataframe
    perf_cols = [col for col in performance_metrics if col in df.columns]
    feat_cols = [col for col in feature_metrics if col in df.columns]
    
    # Create performance metrics table
    if perf_cols:
        perf_df = df[perf_cols]
        fig, ax = plt.figure(figsize=(12, len(results) * 0.5 + 2)), plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=perf_df.values.round(2), 
                        rowLabels=perf_df.index, 
                        colLabels=perf_df.columns, 
                        cellLoc='center',
                        loc='center')
                        
        # Apply alternating row colors for better readability
        for i, key in enumerate(table._cells):
            if i == 0:  # Header row
                continue
            if i % 2 == 0:
                table._cells[key].set_facecolor('lightgray')
            else:
                table._cells[key].set_facecolor('white')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(f"Performance Metrics - {title}", fontsize=14, pad=20)
        
        # Save figure
        output_dir = 'output/images'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'performance_metrics_table_{table_num}.png'), 
                    bbox_inches='tight', dpi=300)
        logger.info(f"Performance metrics table saved to output/images/performance_metrics_table_{table_num}.png")
        plt.close()
    
    # Create feature metrics table
    if feat_cols:
        feat_df = df[feat_cols]
        fig, ax = plt.figure(figsize=(10, len(results) * 0.5 + 2)), plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=feat_df.values.round(2), 
                        rowLabels=feat_df.index, 
                        colLabels=feat_df.columns, 
                        cellLoc='center',
                        loc='center')
                        
        # Apply alternating row colors for better readability
        for i, key in enumerate(table._cells):
            if i == 0:  # Header row
                continue
            if i % 2 == 0:
                table._cells[key].set_facecolor('lightgray')
            else:
                table._cells[key].set_facecolor('white')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(f"Feature Metrics - {title}", fontsize=14, pad=20)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'feature_metrics_table_{table_num}.png'), 
                    bbox_inches='tight', dpi=300)
        logger.info(f"Feature metrics table saved to output/images/feature_metrics_table_{table_num}.png")
        plt.close()
    
    # Create complete table with all metrics (original behavior)
    fig, ax = plt.figure(figsize=(14, len(results) * 0.5 + 2)), plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values.round(2), 
                    rowLabels=df.index, 
                    colLabels=df.columns, 
                    cellLoc='center',
                    loc='center')
                    
    # Apply alternating row colors for better readability
    for i, key in enumerate(table._cells):
        if i == 0:  # Header row
            continue
        if i % 2 == 0:
            table._cells[key].set_facecolor('lightgray')
        else:
            table._cells[key].set_facecolor('white')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly smaller font to fit more columns
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title(title, fontsize=14, pad=20)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'model_comparison_table_{table_num}.png'), 
                bbox_inches='tight', dpi=300)
    logger.info(f"Complete summary table saved to output/images/model_comparison_table_{table_num}.png")
    plt.close()
    
    # Create a training summary text file with all results
    summaries_dir = os.path.join('output', 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    summary_path = os.path.join(summaries_dir, f'training_summary_{table_num}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"{'=' * len(title)}\n\n")
        f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Results by classifier:\n")
        f.write(f"{'-' * 80}\n\n")
        
        for classifier_name, metrics in results.items():
            f.write(f"Classifier: {classifier_name}\n")
            f.write(f"{'-' * (len(classifier_name) + 11)}\n")
            
            # Performance metrics
            f.write("Performance Metrics:\n")
            for metric in performance_metrics:
                clean_metric = metric.replace(' (%)', '')
                if clean_metric.upper() in metrics or clean_metric in metrics:
                    key = clean_metric.upper() if clean_metric.upper() in metrics else clean_metric
                    f.write(f"  {metric}: {metrics[key]:.2f}%\n")
            
            # Feature metrics
            if 'NUM_FEATURES' in metrics:
                f.write("\nFeature Information:\n")
                f.write(f"  Total Features: {metrics['NUM_FEATURES']}\n")
                f.write(f"  Selected Features: {metrics.get('NUM_SELECTED', 'N/A')}\n")
                if 'FEATURE_REDUCTION' in metrics:
                    f.write(f"  Feature Reduction: {metrics['FEATURE_REDUCTION']:.2f}%\n")
            
            f.write("\n\n")
        
        f.write(f"Summary Table:\n")
        f.write(f"{'-' * 80}\n")
        f.write(df.to_string(float_format=lambda x: f"{x:.2f}"))
        f.write("\n\n")
        
    logger.info(f"Comprehensive training summary saved to {summary_path}")

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
    os.makedirs('output/metrics', exist_ok=True)
    os.makedirs('output/features', exist_ok=True)
    os.makedirs('output/summaries', exist_ok=True)

    logger.info(f"Running in {args.mode} mode with classifiers: '{args.classifiers}'")

    # Display CNN configuration if applicable
    if 'cnn' in args.classifiers.lower() or args.classifiers.lower() == 'all':
        logger.info(f"CNN configuration: {args.cnn_model} architecture, " +
                   f"{args.input_size}x{args.input_size} input size, " +
                   f"{args.epochs} epochs, batch size {args.batch_size}")

    # Execute based on mode
    try:
        if args.mode == 'train':
            train(args, logger)
        elif args.mode == 'classify':
            classify(args, logger)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()