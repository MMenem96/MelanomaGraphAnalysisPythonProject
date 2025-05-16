import os
import sys
import argparse
import logging
import time
import traceback
import glob
import json
import pickle
import random
import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, roc_auc_score,
    precision_score, f1_score, roc_curve, precision_recall_curve, auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
# In newer sklearn versions, calibration_curve moved to calibration module
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (
    learning_curve, StratifiedKFold, cross_validate, train_test_split,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif, RFE
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.dataset_handler import DatasetHandler
from src.classifier import BCCSKClassifier
from src.conventional_features import ConventionalFeatureExtractor
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
            logging.FileHandler("bcc_sk_detection.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("BCCSKDetection")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BCC vs SK Detection System')

    # Dataset paths
    parser.add_argument('--bcc-dir', type=str, default='data/bcc',
                        help='Directory containing Basal-cell Carcinoma (BCC) images')
    parser.add_argument('--sk-dir', type=str, default='data/sk',
                        help='Directory containing Seborrheic Keratosis (SK) images')

    # Dataset balance parameters
    parser.add_argument('--max-images-per-class', type=int, default=2000,
                        help='Maximum number of images to use per class for balanced dataset')

    # Model parameters
    parser.add_argument('--classifiers', type=str, default='all',
                        help='Comma-separated list of classifiers to train (e.g., svm_rbf,knn,rf,cnn) or "all"')

    # CNN-specific parameters
    parser.add_argument('--cnn-model', type=str, 
                      choices=['custom', 'resnet50', 'efficient_net', 'inception_v3', 'enhanced_efficientnet'], 
                      default='enhanced_efficientnet',
                      help='CNN architecture to use (enhanced_efficientnet offers best performance)')
    parser.add_argument('--enhanced', action='store_true',
                      help='Use enhanced training techniques (MixUp data augmentation, cyclic learning rates)')
    parser.add_argument('--input-size', type=int, default=224,
                      help='Input image size for CNN (square)')
    parser.add_argument('--epochs', type=int, default=75,
                      help='Number of training epochs for CNN')
    parser.add_argument('--fine-tune-epochs', type=int, default=30,
                      help='Number of fine-tuning epochs for CNN transfer learning')
    parser.add_argument('--unfreeze-layers', type=int, default=30,
                      help='Number of layers to unfreeze during fine-tuning')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                      help='Alpha parameter for MixUp augmentation (0.2-0.4 recommended for skin lesions)')
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
    parser.add_argument('--mode', type=str, choices=['train', 'train_features', 'classify'], default='train',
                        help='Operation mode: train (train graph-based models), train_features (train using conventional feature engineering), or classify (single image)')
    
    # Feature engineering parameters (for train_features mode)
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['basic', 'color', 'texture', 'shape', 'dermoscopy', 'full'],
                        help='Set of features to use for conventional feature engineering')
    parser.add_argument('--feature_selection', type=str, default='mutual_info',
                        choices=['none', 'mutual_info', 'chi2', 'f_test', 'rfe'],
                        help='Feature selection method for conventional feature engineering')
    parser.add_argument('--n_features', type=int, default=100,
                        help='Number of features to select when using feature selection')
    parser.add_argument('--feature_classifiers', type=str, default='rf,svm_rbf,xgboost',
                        help='Comma-separated list of classifiers to train with feature engineering')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization for feature-based classifiers')

    return parser.parse_args()

def train(args, logger):
    """Train one or more skin lesion classification models (BCC vs SK) based on specified classifiers with enhanced feature selection and evaluation."""
    try:
        # Start timer
        start_time = time.time()

        # Check if CNN is specified in classifiers
        include_cnn = 'cnn' in args.classifiers.lower() or args.classifiers.lower() == 'all'

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
            bcc_paths = glob.glob(os.path.join(args.bcc_dir, "*.jpg")) + \
                       glob.glob(os.path.join(args.bcc_dir, "*.png")) + \
                       glob.glob(os.path.join(args.bcc_dir, "*.jpeg"))

            sk_paths = glob.glob(os.path.join(args.sk_dir, "*.jpg")) + \
                      glob.glob(os.path.join(args.sk_dir, "*.png")) + \
                      glob.glob(os.path.join(args.sk_dir, "*.jpeg"))

            # Limit number of images if needed
            if args.max_images_per_class > 0:
                np.random.seed(42)  # For reproducibility
                if len(bcc_paths) > args.max_images_per_class:
                    bcc_paths = np.random.choice(bcc_paths, args.max_images_per_class, replace=False).tolist()
                if len(sk_paths) > args.max_images_per_class:
                    sk_paths = np.random.choice(sk_paths, args.max_images_per_class, replace=False).tolist()

            # Combine paths and create labels
            all_image_paths = bcc_paths + sk_paths
            all_image_labels = np.array([1] * len(bcc_paths) + [0] * len(sk_paths))

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
            logger.info(f"Processing dataset from {args.bcc_dir} and {args.sk_dir}")
            graphs, labels = dataset_handler.process_dataset(
                args.bcc_dir,
                args.sk_dir
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
                label_name = "Basal-cell Carcinoma (BCC)" if label == 1 else "Seborrheic Keratosis (SK)"
                logger.info(f"  {label_name}: {count} samples")
            
            if len(graphs) < 20:
                logger.warning("Very small dataset detected! Results may not be reliable.")
                logger.warning("Recommended minimum: 100 samples per class")
            elif len(graphs) < 100:
                logger.warning("Small dataset detected. Consider adding more training data for better results.")

            # Use our optimized BCCSKClassifier to prepare the features
            logger.info("Preparing features with optimized extraction methods...")
            bcc_sk_classifier = BCCSKClassifier()
            X_train = bcc_sk_classifier.prepare_features(train_graphs)
            X_test = bcc_sk_classifier.prepare_features(test_graphs)
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
                    # Use our enhanced BCCSKClassifier with all optimizations
                    logger.info(f"Initializing {classifier_name} with optimized parameters")
                    classifier = BCCSKClassifier(classifier_type=classifier_type)
                    
                    # Scale features
                    logger.info("Scaling features...")
                    X_train_scaled = classifier.scaler.fit_transform(X_train)
                    X_test_scaled = classifier.scaler.transform(X_test)
                    
                    # Feature selection fully disabled - using all features without any constant removal
                    logger.info("Feature selection completely disabled - using all features...")
                    # X_train_selected = classifier.select_features(X_train_scaled, train_labels)
                    
                    # Directly use the scaled features without any feature selection/constant removal
                    X_train_selected = X_train_scaled
                    X_test_selected = X_test_scaled
                    
                    # Set feature_selector to None to bypass any selection
                    classifier.feature_selector = None
                    
                    # # Since we're using all features, just need to make sure we handle constant features consistently
                    # # This will return X_test_scaled with only constant features removed if any
                    # if classifier.feature_selector is not None:
                    #     try:
                    #         X_test_selected = classifier.feature_selector.transform(X_test_scaled)
                    #     except ValueError as e:
                    #         logger.warning(f"Feature selection transform error: {str(e)}")
                    #         logger.info("Falling back to using all test features")
                    #         X_test_selected = X_test_scaled
                    # else:
                    #     X_test_selected = X_test_scaled
                    
                    logger.info(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")

                    # Train classifier with cross-validation
                    start_training = time.time()
                    logger.info(f"Training {classifier_name} with cross-validation...")
                    
                    # Optimize hyperparameters if dataset is large enough
                    if len(train_graphs) >= 50:
                        logger.info("Optimizing hyperparameters...")
                        # Convert labels to integers for bincount
                        train_labels_int = train_labels.astype(int)
                        
                        # Get minimum class count and ensure it's at least 2x the number of folds
                        min_class_count = min(np.bincount(train_labels_int))
                        # Calculate safe CV value - at minimum each class needs 2 samples per fold
                        cv_value = min(5, min_class_count // 2)
                        # Ensure at least 2 folds 
                        cv_value = max(2, cv_value)
                        
                        logger.info(f"Using {cv_value}-fold cross-validation for hyperparameter optimization (min class count: {min_class_count})")
                        classifier.optimize_hyperparameters(X_train_selected, train_labels, cv=cv_value)
                    
                    # Train and evaluate with cross-validation
                    # Convert labels to integers for bincount
                    train_labels_int = train_labels.astype(int)
                    
                    # Get minimum class count and ensure it's at least 2x the number of folds
                    min_class_count = min(np.bincount(train_labels_int))
                    # Calculate safe CV value - at minimum each class needs 2 samples per fold
                    cv_value = min(5, min_class_count // 2)
                    # Ensure at least 2 folds
                    cv_value = max(2, cv_value)
                    
                    logger.info(f"Using {cv_value}-fold cross-validation for evaluation (min class count: {min_class_count})")
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
                    # Use enhanced predict methods that handle feature dimension mismatches
                    y_pred = classifier.predict(X_test_selected)
                    y_proba = classifier.predict_proba(X_test_selected)[:, 1] if hasattr(classifier, 'predict_proba') else None
                    
                    # Calculate metrics
                    metrics = {}
                    metrics['AC'] = accuracy_score(test_labels, y_pred) * 100
                    metrics['SN'] = recall_score(test_labels, y_pred, zero_division='warn') * 100
                    metrics['SP'] = specificity_score(test_labels, y_pred) * 100
                    metrics['PR'] = precision_score(test_labels, y_pred, zero_division='warn') * 100
                    metrics['F1'] = f1_score(test_labels, y_pred, zero_division='warn') * 100
                    metrics['NUM_FEATURES'] = X_train.shape[1]  # Total features
                    metrics['NUM_SELECTED'] = X_train_selected.shape[1]  # Selected features
                    metrics['FEATURE_REDUCTION'] = ((X_train.shape[1] - X_train_selected.shape[1]) / X_train.shape[1]) * 100
                    
                    if y_proba is not None:
                        try:
                            metrics['AUC'] = roc_auc_score(test_labels, y_proba) * 100
                            
                            # # Generate ROC curve - COMMENTED OUT FOR FASTER TRAINING
                            # fpr, tpr, _ = roc_curve(test_labels, y_proba)
                            # 
                            # # Create ROC curve plot
                            # plt.figure(figsize=(8, 6))
                            # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["AUC"]:.2f}%)')
                            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            # plt.xlim([0.0, 1.0])
                            # plt.ylim([0.0, 1.05])
                            # plt.xlabel('False Positive Rate')
                            # plt.ylabel('True Positive Rate')
                            # plt.title(f'ROC Curve for {classifier_name}')
                            # plt.legend(loc="lower right")
                            # 
                            # # Save ROC curve
                            # roc_curve_path = os.path.join(model_subdir, "roc_curve.png")
                            # plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
                            # plt.close()
                            # logger.info(f"ROC curve saved to {roc_curve_path}")
                            
                            # # Generate precision-recall curve - COMMENTED OUT FOR FASTER TRAINING
                            precision, recall, _ = precision_recall_curve(test_labels, y_proba)
                            pr_auc = auc(recall, precision)
                            metrics['PR_AUC'] = pr_auc * 100
                            
                            # # Create precision-recall curve plot
                            # plt.figure(figsize=(8, 6))
                            # plt.plot(recall, precision, color='green', lw=2, 
                            #        label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
                            # plt.xlabel('Recall')
                            # plt.ylabel('Precision')
                            # plt.ylim([0.0, 1.05])
                            # plt.xlim([0.0, 1.0])
                            # plt.title(f'Precision-Recall Curve for {classifier_name}')
                            # plt.legend(loc="lower left")
                            # 
                            # # Save precision-recall curve
                            # pr_curve_path = os.path.join(model_subdir, "precision_recall_curve.png")
                            # plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
                            # plt.close()
                            # logger.info(f"Precision-Recall curve saved to {pr_curve_path}")
                            
                            # Create output directory for additional visualizations
                            output_dir = os.path.join('output', 'images', model_dir_name)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # # Generate and save learning curve - COMMENTED OUT FOR FASTER TRAINING
                            # logger.info(f"Generating learning curve for {classifier_name}...")
                            # learning_curve_path = os.path.join(output_dir, "learning_curve.png")
                            # plot_learning_curve(
                            #     classifier.classifier, 
                            #     X_train_selected, 
                            #     train_labels,
                            #     cv=cv_value,
                            #     title=f"Learning Curve for {classifier_name}",
                            #     save_path=learning_curve_path
                            # )
                            # logger.info(f"Learning curve saved to {learning_curve_path}")
                            
                            # # Generate and save calibration curve - COMMENTED OUT FOR FASTER TRAINING
                            # logger.info(f"Generating calibration curve for {classifier_name}...")
                            # calibration_curve_path = os.path.join(output_dir, "calibration_curve.png")
                            # plot_calibration_curve(
                            #     classifier.classifier,
                            #     X_test_selected,
                            #     test_labels,
                            #     name=classifier_name,
                            #     save_path=calibration_curve_path
                            # )
                            # logger.info(f"Calibration curve saved to {calibration_curve_path}")
                            
                            # # Generate and save prediction histogram - COMMENTED OUT FOR FASTER TRAINING
                            # logger.info(f"Generating prediction histogram for {classifier_name}...")
                            # pred_hist_path = os.path.join(output_dir, "prediction_histogram.png")
                            # plot_prediction_histogram(
                            #     classifier.classifier,
                            #     X_test_selected,
                            #     test_labels,
                            #     save_path=pred_hist_path
                            # )
                            # logger.info(f"Prediction histogram saved to {pred_hist_path}")
                            # 
                            # # Generate and save F1 score vs. threshold curve - COMMENTED OUT FOR FASTER TRAINING
                            # logger.info(f"Generating F1 score curve for {classifier_name}...")
                            # f1_curve_path = os.path.join(output_dir, "f1_threshold_curve.png")
                            # plot_f1_threshold_curve(
                            #     classifier.classifier,
                            #     X_test_selected,
                            #     test_labels,
                            #     save_path=f1_curve_path
                            # )
                            # logger.info(f"F1 score curve saved to {f1_curve_path}")
                            
                            # # Generate and save feature importance plot if the classifier supports it - COMMENTED OUT FOR FASTER TRAINING
                            # logger.info(f"Generating feature importance plot for {classifier_name}...")
                            # feature_importance_path = os.path.join(output_dir, "feature_importance.png")
                            # try:
                            #     # Ensure feature dimensions match before plotting importance
                            #     X_to_plot = X_test_selected
                            #     if hasattr(classifier.classifier, 'n_features_in_'):
                            #         expected_features = classifier.classifier.n_features_in_
                            #         current_features = X_test_selected.shape[1]
                            #         
                            #         if current_features != expected_features:
                            #             logger.warning(f"Adjusting feature dimensions for importance plot: {current_features} to {expected_features}")
                            #             
                            #             if current_features > expected_features:
                            #                 # Truncate features
                            #                 X_to_plot = X_test_selected[:, :expected_features]
                            #             else:
                            #                 # Pad with zeros
                            #                 padding = np.zeros((X_test_selected.shape[0], expected_features - current_features))
                            #                 X_to_plot = np.hstack((X_test_selected, padding))
                            #     
                            #     plot_feature_importance(
                            #         classifier.classifier,
                            #         X_to_plot,
                            #         test_labels,
                            #         feature_names=[f"Feature {i}" for i in range(X_to_plot.shape[1])],
                            #         top_n=min(20, X_to_plot.shape[1]),
                            #         save_path=feature_importance_path
                            #     )
                            #     logger.info(f"Feature importance plot saved to {feature_importance_path}")
                            # except Exception as imp_err:
                            #     logger.warning(f"Could not generate feature importance plot: {str(imp_err)}")
                            
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
                    
                    # # Generate confusion matrix plot - COMMENTED OUT FOR FASTER TRAINING
                    # plt.figure(figsize=(8, 6))
                    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    # plt.title(f'Confusion Matrix for {classifier_name}')
                    # plt.colorbar()
                    # plt.xticks([0, 1], ['Seborrheic Keratosis (SK)', 'Basal-cell Carcinoma (BCC)'])
                    # plt.yticks([0, 1], ['Seborrheic Keratosis (SK)', 'Basal-cell Carcinoma (BCC)'])
                    # 
                    # # Add text annotations to the confusion matrix
                    # thresh = cm.max() / 2
                    # for i in range(2):
                    #     for j in range(2):
                    #         plt.text(j, i, format(cm[i, j], 'd'),
                    #                 ha="center", va="center",
                    #                 color="white" if cm[i, j] > thresh else "black")
                    # 
                    # plt.ylabel('True Label')
                    # plt.xlabel('Predicted Label')
                    # 
                    # # Save confusion matrix
                    # cm_path = os.path.join(model_subdir, "confusion_matrix.png")
                    # plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    # plt.close()
                    # logger.info(f"Confusion matrix saved to {cm_path}")
                    
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
                from src.cnn_classifier import CNNBCCSKClassifier
                
                # Create model directory
                model_subdir = os.path.join('model', 'CNN')
                os.makedirs(model_subdir, exist_ok=True)

                # Configure input shape and training parameters
                input_shape = (args.input_size, args.input_size, 3)
                epochs = args.epochs
                batch_size = args.batch_size

                # Initialize the CNN classifier
                cnn_classifier = CNNBCCSKClassifier(
                    model_type=args.cnn_model,
                    input_shape=input_shape,
                    enhanced=args.enhanced
                )

                # Train the CNN
                start_training = time.time()
                
                # Determine model file name based on model type and enhanced flag
                model_file = f"bcc_sk_{args.cnn_model}_"
                model_file += "enhanced_" if args.enhanced else ""
                model_file += "model.h5"
                model_path = os.path.join(model_subdir, model_file)
                
                # Log detailed training approach with all parameters
                if args.enhanced or args.cnn_model == 'enhanced_efficientnet':
                    logger.info(f"Training enhanced CNN ({args.cnn_model}) with advanced techniques...")
                    logger.info(f"Training configuration:")
                    logger.info(f"  - Input shape: {input_shape}")
                    logger.info(f"  - Initial epochs: {epochs}")
                    logger.info(f"  - Batch size: {batch_size}")
                    logger.info(f"  - Fine-tuning epochs: {args.fine_tune_epochs}")
                    logger.info(f"  - Unfrozen layers: {args.unfreeze_layers}")
                    logger.info(f"  - MixUp alpha: {args.mixup_alpha} (higher = stronger interpolation)")
                    logger.info(f"  - Using cyclic learning rate scheduling")
                else:
                    logger.info(f"Training CNN ({args.cnn_model})...")
                    logger.info(f"Training configuration:")
                    logger.info(f"  - Input shape: {input_shape}")
                    logger.info(f"  - Epochs: {epochs}")
                    logger.info(f"  - Batch size: {batch_size}")
                    logger.info(f"  - Fine-tuning epochs: {args.fine_tune_epochs}")
                    logger.info(f"  - Unfrozen layers: {args.unfreeze_layers}")
                
                # Configure fine-tuning layers
                unfreeze_layers = 0
                fine_tune_epochs = 0
                
                # Set fine-tuning parameters based on command line arguments and model type
                # Default to command line arguments
                unfreeze_layers = args.unfreeze_layers
                fine_tune_epochs = args.fine_tune_epochs
                
                # Log fine-tuning strategy based on model type
                if args.cnn_model == 'enhanced_efficientnet':
                    # Enhanced EfficientNet has built-in fine-tuning layers
                    logger.info("Enhanced EfficientNet uses built-in fine-tuning with residual connections")
                elif args.enhanced:
                    # Enhanced training with standard models gets intensive fine-tuning
                    logger.info(f"Enhanced fine-tuning will be applied: last {unfreeze_layers} layers for {fine_tune_epochs} epochs")
                else:
                    # Standard training
                    logger.info(f"Standard fine-tuning will be applied: last {unfreeze_layers} layers for {fine_tune_epochs} epochs")
                
                # Train with fine-tuning parameters and MixUp augmentation
                history = cnn_classifier.fit(
                    cnn_train_paths, cnn_train_labels,
                    X_val=cnn_test_paths, y_val=cnn_test_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    unfreeze_layers=unfreeze_layers,
                    fine_tune_epochs=fine_tune_epochs,
                    model_path=model_path,
                    mixup_alpha=args.mixup_alpha
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
                
                # Get model summary information if available
                if 'model_info' in eval_results:
                    model_info = eval_results['model_info']
                    hyperparams = model_info.get('hyperparameters', {})
                    feature_counts = model_info.get('feature_counts', {})
                    
                    # Log detailed model architecture information
                    logger.info("CNN Model Architecture Summary:")
                    logger.info(f"  Model type: {model_info.get('model_type', 'unknown')}")
                    logger.info(f"  Enhanced training: {model_info.get('enhanced', False)}")
                    logger.info(f"  Input shape: {model_info.get('input_shape', 'unknown')}")
                    
                    # Log hyperparameters
                    logger.info("CNN Training Hyperparameters:")
                    logger.info(f"  Total epochs: {hyperparams.get('total_epochs', 0)}")
                    logger.info(f"  Batch size: {hyperparams.get('batch_size', 0)}")
                    logger.info(f"  Fine-tune epochs: {hyperparams.get('fine_tune_epochs', 0)}")
                    logger.info(f"  Unfrozen layers: {hyperparams.get('unfreeze_layers', 0)}")
                    logger.info(f"  MixUp alpha: {hyperparams.get('mixup_alpha', 0)}")
                    
                    # Log feature counts
                    logger.info("CNN Model Parameters:")
                    logger.info(f"  Total parameters: {feature_counts.get('total', 0):,}")
                    logger.info(f"  Trainable parameters: {feature_counts.get('used', 0):,}")
                    logger.info(f"  Frozen parameters: {feature_counts.get('frozen', 0):,}")
                    
                    # Add key metrics to results dictionary
                    metrics['NUM_FEATURES'] = feature_counts.get('total', 0)
                    metrics['NUM_SELECTED'] = feature_counts.get('used', 0)
                    if metrics['NUM_FEATURES'] > 0:
                        metrics['FEATURE_REDUCTION'] = ((metrics['NUM_FEATURES'] - metrics['NUM_SELECTED']) / metrics['NUM_FEATURES']) * 100
                    else:
                        metrics['FEATURE_REDUCTION'] = 0
                
                # Fallback to direct feature counts if model_info not available
                elif hasattr(cnn_classifier, 'feature_counts'):
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
                    f.write(f"  Initial Epochs: {epochs}\n")
                    f.write(f"  Fine-tuning Epochs: {args.fine_tune_epochs}\n")
                    f.write(f"  Unfrozen Layers: {args.unfreeze_layers}\n")
                    f.write(f"  Enhanced Training: {args.enhanced}\n")
                    f.write(f"  MixUp Alpha: {args.mixup_alpha}\n")
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
                    plt.xticks([0, 1], ['Seborrheic Keratosis (SK)', 'Basal-cell Carcinoma (BCC)'])
                    plt.yticks([0, 1], ['Seborrheic Keratosis (SK)', 'Basal-cell Carcinoma (BCC)'])
                    
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
                             label=f'True Basal-cell Carcinoma (BCC) (n={len(y_prob_pos)})')
                    plt.hist(y_prob_neg, bins=20, alpha=0.6, color='green', 
                             label=f'True Seborrheic Keratosis (SK) (n={len(y_prob_neg)})')
                    
                    plt.xlabel('Predicted Probability of Basal-cell Carcinoma (BCC)')
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
                        f1 = f1_score(y_true, y_pred_t, zero_division="warn")
                        precision = precision_score(y_true, y_pred_t, zero_division="warn")
                        recall = recall_score(y_true, y_pred_t, zero_division="warn")
                        
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
            generate_summary_table(results, logger, table_num=1, title="BCC vs SK Detection Model Training Summary")
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

def train_features(args, logger):
    """Train skin lesion classification models using conventional feature engineering approach with dermoscopic features.
    
    This function implements a complete machine learning pipeline using extracted features from dermoscopic images:
    1. Load and preprocess images from BCC and SK datasets
    2. Extract comprehensive feature sets (color, texture, morphology, dermoscopic)
    3. Apply feature selection and dimensionality reduction
    4. Train and optimize multiple classifier models
    5. Evaluate performance with cross-validation and detailed metrics
    6. Generate visualizations of feature importance and model comparisons
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    # Explicitly import the train_test_split function to make sure it's in scope
    from sklearn.model_selection import train_test_split
    try:
        # Start timer
        start_time = time.time()
        
        logger.info("Starting conventional feature engineering-based training")
        logger.info(f"Feature set: {args.feature_set}, Selection method: {args.feature_selection}")
        
        # Load image paths
        logger.info("Loading image paths")
        bcc_paths = glob.glob(os.path.join(args.bcc_dir, "*.jpg")) + \
                    glob.glob(os.path.join(args.bcc_dir, "*.png")) + \
                    glob.glob(os.path.join(args.bcc_dir, "*.jpeg"))
        
        sk_paths = glob.glob(os.path.join(args.sk_dir, "*.jpg")) + \
                   glob.glob(os.path.join(args.sk_dir, "*.png")) + \
                   glob.glob(os.path.join(args.sk_dir, "*.jpeg"))
        
        # Balance dataset
        if args.max_images_per_class > 0:
            random.shuffle(bcc_paths)
            random.shuffle(sk_paths)
            bcc_paths = bcc_paths[:min(len(bcc_paths), args.max_images_per_class)]
            sk_paths = sk_paths[:min(len(sk_paths), args.max_images_per_class)]
        
        # Create labels, explicitly as integers (not floats)
        bcc_labels = np.ones(len(bcc_paths), dtype=np.int32)
        sk_labels = np.zeros(len(sk_paths), dtype=np.int32)
        
        # Combine datasets
        all_image_paths = bcc_paths + sk_paths
        all_labels = np.concatenate([bcc_labels, sk_labels]).astype(np.int32)
        
        logger.info(f"Loaded {len(bcc_paths)} BCC images and {len(sk_paths)} SK images")
        
        # Initialize the feature extractor
        feature_extractor = ConventionalFeatureExtractor()
        
        # Extract features from all images
        logger.info("Extracting features from images...")
        features_list = []
        success_count = 0
        
        for idx, image_path in enumerate(all_image_paths):
            if idx % 100 == 0:
                logger.info(f"Processing image {idx+1}/{len(all_image_paths)}")
            
            try:
                # Load and preprocess image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing (resize to manageable dimensions if needed)
                max_dim = 512
                if max(image.shape[0], image.shape[1]) > max_dim:
                    scale = max_dim / max(image.shape[0], image.shape[1])
                    new_width = int(image.shape[1] * scale)
                    new_height = int(image.shape[0] * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Extract features based on selected feature set
                if args.feature_set == 'full':
                    # Extract all available features
                    features = feature_extractor.extract_all_features(image)
                elif args.feature_set == 'color':
                    # Only color features
                    features = feature_extractor.extract_color_features(image, np.ones(image.shape[:2], dtype=bool))
                elif args.feature_set == 'texture':
                    # Only texture features
                    features = feature_extractor.extract_texture_features(image, np.ones(image.shape[:2], dtype=bool))
                elif args.feature_set == 'shape':
                    # Generate mask using Otsu's method for shape analysis
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    features = feature_extractor.extract_geometric_features(mask.astype(bool))
                else:
                    # Basic set - combine a subset of key features
                    color_features = feature_extractor.extract_color_features(image, np.ones(image.shape[:2], dtype=bool))
                    texture_features = feature_extractor.extract_texture_features(image, np.ones(image.shape[:2], dtype=bool))
                    
                    # Select only key features from each category
                    features = {}
                    for key in color_features:
                        if any(x in key for x in ['mean', 'std', 'entropy', 'color_variance']):
                            features[key] = color_features[key]
                    
                    for key in texture_features:
                        if any(x in key for x in ['glcm_contrast', 'glcm_homogeneity', 'wavelet_approx', 'gradient_mag']):
                            features[key] = texture_features[key]
                
                features_list.append(features)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                # Add empty feature dict to maintain alignment with labels
                features_list.append({})
        
        logger.info(f"Successfully processed {success_count} images out of {len(all_image_paths)}")
        
        # Convert features to a usable format for machine learning
        # First, identify all features and handle lists/arrays
        expanded_feature_keys = []
        
        # First pass: find all feature keys and identify which ones have list/array values
        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (list, np.ndarray)):
                    # For list features, create individual keys for each element
                    for i in range(len(value)):
                        expanded_key = f"{key}_{i}"
                        if expanded_key not in expanded_feature_keys:
                            expanded_feature_keys.append(expanded_key)
                else:
                    # For scalar features, use as is
                    if key not in expanded_feature_keys:
                        expanded_feature_keys.append(key)
        
        expanded_feature_keys = sorted(expanded_feature_keys)
        logger.info(f"Total number of extracted features (after expansion): {len(expanded_feature_keys)}")
        
        # Create feature matrix with expanded features
        X = np.zeros((len(features_list), len(expanded_feature_keys)))
        
        for i, features in enumerate(features_list):
            for j, key in enumerate(expanded_feature_keys):
                # Check if this is an expanded list feature
                if '_' in key and key.rsplit('_', 1)[0] in features:
                    base_key, idx_str = key.rsplit('_', 1)
                    # Only process if it's a numeric index
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        value = features.get(base_key)
                        if isinstance(value, (list, np.ndarray)) and idx < len(value):
                            X[i, j] = value[idx]
                        else:
                            X[i, j] = 0
                else:
                    # Regular scalar feature
                    value = features.get(key, 0)
                    # Ensure it's a scalar
                    if not isinstance(value, (list, np.ndarray)):
                        X[i, j] = value
                    else:
                        X[i, j] = 0  # Default for unexpected list/array
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Ensure label data types are explicitly integer (prevent float conversions)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        # Determine if we're working with a very small dataset (early initialization)
        is_very_small_dataset = len(all_labels) < 30
        if is_very_small_dataset:
            logger.warning(f"Working with a very small dataset (only {len(all_labels)} samples). Adapting training process.")
        
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # Apply feature selection if specified
        if args.feature_selection != 'none':
            logger.info(f"Applying feature selection: {args.feature_selection}")
            
            if args.feature_selection == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'chi2':
                # Chi2 requires non-negative features
                X_train_min = X_train.min(axis=0)
                X_train = X_train - X_train_min
                X_test = X_test - X_train_min
                selector = SelectKBest(chi2, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'f_test':
                selector = SelectKBest(f_classif, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'rfe':
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator=base_model, n_features_to_select=min(args.n_features, X_train.shape[1]))
            
            # Apply selection
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
            # Get selected feature names for interpretation
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = [expanded_feature_keys[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_feature_names)} features")
            
            # Log top 20 features if there are many
            if len(selected_feature_names) > 20:
                if hasattr(selector, 'scores_'):
                    # For filter methods
                    # Get indices of the top scored features among the selected ones
                    feature_scores = [(i, selector.scores_[i]) for i in selected_indices]
                    sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                    
                    # Report top 20 (or fewer if we have less)
                    top_count = min(20, len(sorted_feature_scores))
                    logger.info(f"Top {top_count} features by importance:")
                    
                    for i in range(top_count):
                        idx, score = sorted_feature_scores[i]
                        feature = expanded_feature_keys[idx]
                        logger.info(f"{feature}: {score:.4f}")
                else:
                    # Just list the first 20 if scores are not available
                    logger.info(f"First 20 selected features: {selected_feature_names[:20]}")
        else:
            logger.info("Using all features (no feature selection)")
            selected_feature_names = expanded_feature_keys
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Prepare classifiers based on user selection
        classifiers = {}
        requested_classifiers = args.feature_classifiers.lower().split(',')
        
        for clf_name in requested_classifiers:
            if clf_name == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                classifiers['Random Forest'] = clf
            elif clf_name == 'svm_rbf':
                clf = SVC(kernel='rbf', probability=True, random_state=42)
                classifiers['SVM (RBF)'] = clf
            elif clf_name == 'svm_linear':
                clf = SVC(kernel='linear', probability=True, random_state=42)
                classifiers['SVM (Linear)'] = clf
            elif clf_name == 'knn':
                clf = KNeighborsClassifier(n_neighbors=5)
                classifiers['KNN'] = clf
            elif clf_name == 'mlp':
                clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                classifiers['MLP'] = clf
            elif clf_name == 'xgboost':
                # Set base_score=0.5 to fix the "base_score must be in (0,1)" error
                clf = XGBClassifier(n_estimators=100, random_state=42, base_score=0.5)
                classifiers['XGBoost'] = clf
            elif clf_name == 'gb':
                clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
                classifiers['Gradient Boosting'] = clf
            elif clf_name == 'logistic':
                clf = LogisticRegression(max_iter=1000, random_state=42)
                classifiers['Logistic Regression'] = clf
                
        if not classifiers:
            logger.warning("No valid classifiers specified. Using Random Forest as default.")
            classifiers['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        logger.info(f"Training {len(classifiers)} classifiers: {', '.join(classifiers.keys())}")
        
        # Hyperparameter optimization if specified
        if args.optimize:
            logger.info("Performing hyperparameter optimization")
            
            optimized_classifiers = {}
            
            # We've already determined if we're working with a very small dataset earlier
            
            for name, clf in classifiers.items():
                logger.info(f"Optimizing {name}")
                
                # Use simplified parameter grids for very small datasets
                if is_very_small_dataset:
                    logger.info(f"Using simplified parameter grid for small dataset (size: {len(y_train)} samples)")
                    if name == 'Random Forest':
                        param_grid = {
                            'n_estimators': [50],
                            'max_depth': [3, None],
                            'min_samples_leaf': [1, 2]
                        }
                    elif 'SVM' in name:
                        param_grid = {
                            'C': [1, 10],
                            'gamma': ['scale', 'auto']
                        }
                    elif name == 'KNN':
                        param_grid = {
                            'n_neighbors': [3, 5],
                            'weights': ['uniform', 'distance']
                        }
                    elif name == 'MLP':
                        param_grid = {
                            'hidden_layer_sizes': [(10,), (20,)],
                            'alpha': [0.001, 0.01]
                        }
                    elif name == 'XGBoost':
                        param_grid = {
                            'n_estimators': [50],
                            'max_depth': [3],
                            'learning_rate': [0.1]
                        }
                    elif name == 'Gradient Boosting':
                        param_grid = {
                            'n_estimators': [50],
                            'max_depth': [3],
                            'learning_rate': [0.1]
                        }
                else:
                    # Standard parameter grids for normal-sized datasets
                    if name == 'Random Forest':
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20, 30],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                    elif 'SVM' in name:
                        param_grid = {
                            'C': [0.1, 1, 10, 100],
                            'gamma': ['scale', 'auto', 0.1, 0.01]
                        }
                    elif name == 'KNN':
                        param_grid = {
                            'n_neighbors': [3, 5, 7, 9, 11],
                            'weights': ['uniform', 'distance'],
                            'p': [1, 2]  # Manhattan or Euclidean
                        }
                    elif name == 'MLP':
                        param_grid = {
                            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                            'alpha': [0.0001, 0.001, 0.01],
                            'learning_rate': ['constant', 'adaptive']
                        }
                    elif name == 'XGBoost':
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                    elif name == 'Gradient Boosting':
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                    elif name == 'Logistic Regression':
                        param_grid = {
                            'C': [0.1, 1, 10, 100],
                            'penalty': ['l1', 'l2'],
                            'solver': ['liblinear', 'saga']
                        }
                
                # Define a variable to track if we have a param grid for this classifier
                has_param_grid = True
                    
                # Initialize param_grid as an empty dictionary if it doesn't exist
                param_grid = {}
                
                # If we don't have a param grid defined for this classifier or it's a special case
                if not param_grid:
                    has_param_grid = False
                    # Default - skip optimization
                    optimized_classifiers[name] = clf
                    continue
                
                # Determine appropriate cross-validation strategy based on dataset size
                # For very small datasets, use fewer folds or even LOO (Leave-One-Out) CV
                cv_strategy = 5  # Default 5-fold CV
                
                # Count samples per class to determine appropriate CV strategy
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts[class_counts > 0])
                
                if min_class_count < 5:
                    # For extremely small datasets (< 5 samples in smallest class)
                    logger.info(f"Very small dataset detected ({min_class_count} samples in smallest class)")
                    logger.info(f"Using 2-fold stratified CV for {name}")
                    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                elif min_class_count < 10:
                    # For small datasets (<10 samples in smallest class)
                    logger.info(f"Small dataset detected ({min_class_count} samples in smallest class)")
                    logger.info(f"Using 3-fold stratified CV for {name}")
                    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    
                # Use GridSearchCV for smaller grids, RandomizedSearchCV for larger ones
                if np.prod([len(v) for v in param_grid.values()]) > 30:
                    search = RandomizedSearchCV(
                        clf, param_grid, n_iter=20, cv=cv_strategy, scoring='f1', 
                        random_state=42, n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        clf, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                
                logger.info(f"Best parameters for {name}: {search.best_params_}")
                logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
                
                optimized_classifiers[name] = search.best_estimator_
            
            # Use optimized classifiers for further evaluation
            classifiers = optimized_classifiers
        
        # Train and evaluate each classifier
        results = {}
        
        for name, clf in classifiers.items():
            logger.info(f"Training and evaluating {name}")
            
            try:
                # Determine appropriate cross-validation strategy based on dataset size
                cv_strategy = 5  # Default 5-fold CV
                
                # Count samples per class to determine appropriate CV strategy
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts[class_counts > 0])
                
                # For extremely small datasets, we'll skip cross-validation
                # and just evaluate on the test set
                skip_cv = False
                
                if min_class_count < 3:
                    # For extremely small datasets (<3 samples in smallest class)
                    # We'll skip cross-validation entirely
                    logger.info(f"Very small dataset detected ({min_class_count} samples in smallest class)")
                    logger.info(f"Skipping cross-validation for {name}")
                    skip_cv = True
                    # Create placeholder for CV scores
                    cv_scores = {
                        'test_accuracy': np.array([0.0]),
                        'test_precision': np.array([0.0]),
                        'test_recall': np.array([0.0]),
                        'test_f1': np.array([0.0]),
                        'test_roc_auc': np.array([0.0])
                    }
                elif min_class_count < 5:
                    # For very small datasets (<5 samples in smallest class)
                    logger.info(f"Very small dataset detected ({min_class_count} samples in smallest class)")
                    logger.info(f"Using 2-fold stratified CV for {name}")
                    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                elif min_class_count < 10:
                    # For small datasets (<10 samples in smallest class)
                    logger.info(f"Small dataset detected ({min_class_count} samples in smallest class)")
                    logger.info(f"Using 3-fold stratified CV for {name}")
                    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                
                # Only perform cross-validation if we have enough samples
                if not skip_cv:
                    # Ensure we use integer labels for cross-validation
                    try:
                        # Cross-validation evaluation with explicit integer labels
                        y_train_int = y_train.astype(np.int32)
                        cv_scores = cross_validate(
                            clf, X_train, y_train_int, 
                            cv=cv_strategy,
                            scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
                        )
                    except Exception as e:
                        logger.warning(f"Cross-validation failed with error: {str(e)}")
                        logger.info("Attempting cross-validation with alternative label format")
                        try:
                            # Try with raveled array if the first approach fails
                            cv_scores = cross_validate(
                                clf, X_train, y_train_int.ravel(), 
                                cv=cv_strategy,
                                scoring=['accuracy', 'f1', 'precision', 'recall']  # Remove roc_auc which may cause issues
                            )
                        except Exception as e2:
                            logger.error(f"Cross-validation failed on second attempt: {str(e2)}")
                            # Set empty scores to prevent errors later
                            cv_scores = {
                                'test_accuracy': np.array([0.0]),
                                'test_f1': np.array([0.0]),
                                'test_precision': np.array([0.0]),
                                'test_recall': np.array([0.0])
                            }
                
                # Ensure labels are integers before model training
                # Many classifiers expect integer labels
                y_train_int = y_train.astype(np.int32)
                y_test_int = y_test.astype(np.int32)
                
                # Train final model on full training set with explicit integer labels
                try:
                    clf.fit(X_train, y_train_int)
                except Exception as e:
                    logger.error(f"Error during model fitting: {str(e)}")
                    # Try alternative approach if the first one fails
                    try:
                        # Some models might require different label format
                        logger.info("Attempting alternative approach with different label format")
                        clf.fit(X_train, y_train_int.ravel())
                    except Exception as e2:
                        logger.error(f"Second attempt also failed: {str(e2)}")
                        raise
                
                # Evaluate on test set
                try:
                    y_pred = clf.predict(X_test)
                    y_pred = y_pred.astype(np.int32)  # Ensure predictions are integers
                    
                    if hasattr(clf, "predict_proba"):
                        y_pred_proba = clf.predict_proba(X_test)[:, 1]
                    else:
                        y_pred_proba = None
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    raise
                
                # Calculate metrics with safe handling for small datasets
                # Ensure both arrays have compatible data types
                y_test_int = y_test_int if 'y_test_int' in locals() else y_test.astype(np.int32)
                y_pred_int = y_pred.astype(np.int32)
                
                try:
                    accuracy = accuracy_score(y_test_int, y_pred_int)
                except Exception as e:
                    logger.warning(f"Error calculating accuracy: {str(e)}")
                    accuracy = 0.0
                
                # Use zero_division=0.0 (must be float) for all metrics to handle small datasets better
                try:
                    precision = precision_score(y_test_int, y_pred_int, zero_division=0.0)
                except Exception as e:
                    logger.warning(f"Error calculating precision: {str(e)}")
                    precision = 0.0
                    
                try:    
                    recall = recall_score(y_test_int, y_pred_int, zero_division=0.0)
                except Exception as e:
                    logger.warning(f"Error calculating recall: {str(e)}")
                    recall = 0.0
                    
                try:
                    f1 = f1_score(y_test_int, y_pred_int, zero_division=0.0)
                except Exception as e:
                    logger.warning(f"Error calculating F1 score: {str(e)}")
                    f1 = 0.0
                
                # Calculate specificity with safe handling
                try:
                    specificity = specificity_score(y_test_int, y_pred_int)
                except Exception as e:
                    logger.warning(f"Error calculating specificity: {str(e)}")
                    specificity = 0.0
                
                # Handle ROC AUC calculation safely
                try:
                    # Use integer labels for y_test
                    if y_pred_proba is not None:
                        roc_auc = roc_auc_score(y_test_int, y_pred_proba)
                    else:
                        logger.warning("No probability predictions available for ROC AUC calculation")
                        roc_auc = None
                except Exception as e:
                    logger.warning(f"Error calculating ROC AUC: {str(e)}")
                    roc_auc = None
                # Ensure we're using integer labels for the confusion matrix
                conf_matrix = confusion_matrix(y_test_int, y_pred_int)
                
                # Log results
                logger.info(f"{name} - Test Results:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall/Sensitivity: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                logger.info(f"  Specificity: {specificity:.4f}")
                if roc_auc is not None:
                    logger.info(f"  ROC AUC: {roc_auc:.4f}")
                logger.info(f"  Confusion Matrix:\n{conf_matrix}")
                
                # Store results
                # Include both raw metrics and formatted metrics for the summary table
                results[name] = {
                    'classifier': clf,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_specificity': specificity,
                    'test_roc_auc': roc_auc,
                    'confusion_matrix': conf_matrix,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    # Formatted metrics for summary table - these keys match what generate_summary_table expects
                    'AC': accuracy * 100,  # Convert to percentage
                    'PR': precision * 100,
                    'SN': recall * 100,
                    'F1': f1 * 100,
                    'SP': specificity * 100,
                    'AUC': roc_auc * 100 if roc_auc is not None else None,
                    'NUM_FEATURES': len(selected_feature_names),
                    'NUM_SELECTED': len(selected_feature_names),
                    'FEATURE_REDUCTION': 0.0 if args.feature_selection == 'none' else 
                                         (1 - len(selected_feature_names)/X.shape[1]) * 100
                }
                
                # Add cross-validation results if available
                if 'skip_cv' in locals() and not skip_cv and 'cv_scores' in locals():
                    try:
                        results[name].update({
                            'cv_accuracy': cv_scores['test_accuracy'].mean(),
                            'cv_precision': cv_scores['test_precision'].mean(), 
                            'cv_recall': cv_scores['test_recall'].mean(),
                            'cv_f1': cv_scores['test_f1'].mean(),
                            'cv_roc_auc': cv_scores['test_roc_auc'].mean(),
                        })
                    except Exception as e:
                        logger.warning(f"Error processing CV scores for {name}: {str(e)}")
                        # Fall back to test metrics
                        results[name].update({
                            'cv_accuracy': accuracy,
                            'cv_precision': precision,
                            'cv_recall': recall,
                            'cv_f1': f1,
                            'cv_roc_auc': roc_auc if roc_auc is not None else 0.0,
                        })
                else:
                    # Use test set metrics as fallback when CV is skipped
                    results[name].update({
                        'cv_accuracy': accuracy,
                        'cv_precision': precision,
                        'cv_recall': recall,
                        'cv_f1': f1,
                        'cv_roc_auc': roc_auc if roc_auc is not None else 0.0,
                    })
                
                # Plot ROC curve if probability estimates are available
                if y_pred_proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {name}')
                    plt.legend(loc='lower right')
                    
                    output_path = f'output/metrics/roc_curve_{name.replace(" ", "_")}.png'
                    plt.savefig(output_path)
                    plt.close()
                    
                    logger.info(f"ROC curve saved to {output_path}")
                
                # Generate feature importance plot if available
                if hasattr(clf, 'feature_importances_'):
                    # Get feature importances
                    importances = clf.feature_importances_
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Take top 30 features or all if less
                    n_top_features = min(30, len(selected_feature_names))
                    top_indices = indices[:n_top_features]
                    
                    plt.figure(figsize=(10, 8))
                    plt.title(f'Top {n_top_features} Feature Importances - {name}')
                    plt.barh(range(n_top_features), importances[top_indices], align='center')
                    plt.yticks(range(n_top_features), [selected_feature_names[i] for i in top_indices])
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    
                    output_path = f'output/metrics/feature_importance_{name.replace(" ", "_")}.png'
                    plt.savefig(output_path)
                    plt.close()
                    
                    logger.info(f"Feature importance plot saved to {output_path}")
                
                # For SVM with linear kernel, plot feature coefficients
                elif name == 'SVM (Linear)' and hasattr(clf, 'coef_'):
                    # Get coefficients
                    coefficients = clf.coef_[0]
                    
                    # Sort features by absolute coefficient value
                    indices = np.argsort(np.abs(coefficients))[::-1]
                    
                    # Take top 30 features or all if less
                    n_top_features = min(30, len(selected_feature_names))
                    top_indices = indices[:n_top_features]
                    
                    plt.figure(figsize=(10, 8))
                    plt.title(f'Top {n_top_features} Feature Coefficients - {name}')
                    plt.barh(range(n_top_features), coefficients[top_indices], align='center')
                    plt.yticks(range(n_top_features), [selected_feature_names[i] for i in top_indices])
                    plt.xlabel('Coefficient')
                    plt.tight_layout()
                    
                    output_path = f'output/metrics/feature_coefficients_{name.replace(" ", "_")}.png'
                    plt.savefig(output_path)
                    plt.close()
                    
                    logger.info(f"Feature coefficients plot saved to {output_path}")
                
                # Save model for ALL classifiers, not just linear SVM
                # Create a unique directory for each classifier with timestamp for versioning
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_model_dir = f'model/feature_based'
                os.makedirs(base_model_dir, exist_ok=True)
                
                # Sanitize classifier name for directory naming
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                model_dir = f'{base_model_dir}/{safe_name}_{timestamp}'
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Creating model directory: {model_dir}")
                
                # Define paths for all model components
                model_path = f'{model_dir}/model.joblib'
                scaler_path = f'{model_dir}/scaler.joblib'
                selector_path = f'{model_dir}/selector.joblib' if selector else None
                metadata_path = f'{model_dir}/metadata.json'
                performance_path = f'{model_dir}/performance.json'
                
                # Save the classifier model using joblib for better performance
                try:
                    dump(clf, model_path)
                    logger.info(f"Model saved to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {str(e)}")
                
                # Save the scaler using joblib
                try:
                    dump(scaler, scaler_path)
                    logger.info(f"Scaler saved to {scaler_path}")
                except Exception as e:
                    logger.error(f"Error saving scaler: {str(e)}")
                    
                # Create a symlink to the latest model
                latest_dir = f'{base_model_dir}/{safe_name}_latest'
                if os.path.exists(latest_dir) and os.path.islink(latest_dir):
                    os.unlink(latest_dir)
                try:
                    # Use relative path for platform independence
                    os.symlink(os.path.basename(model_dir), latest_dir, target_is_directory=True)
                    logger.info(f"Created symlink from {latest_dir} to {model_dir}")
                except Exception as e:
                    # Symlinks might not work on all platforms, so just log the error
                    logger.warning(f"Could not create symlink (might not be supported): {str(e)}")
                    
                # Save selector if available using joblib
                if selector:
                    try:
                        dump(selector, selector_path)
                        logger.info(f"Feature selector saved to {selector_path}")
                    except Exception as e:
                        logger.error(f"Error saving feature selector: {str(e)}")
                
                # Save comprehensive metadata about the model, training data and performance
                metadata = {
                    # Feature information
                    'features': selected_feature_names,
                    'num_features': len(selected_feature_names),
                    'feature_selection_method': args.feature_selection,
                    'feature_set': args.feature_set,
                    
                    # Dataset information
                    'dataset_size': {
                        'total_samples': len(y_train) + len(y_test),
                        'training_samples': len(y_train),
                        'test_samples': len(y_test),
                        'class_counts': {
                            'training': {
                                'bcc': int(np.sum(y_train == 1)),
                                'sk': int(np.sum(y_train == 0))
                            },
                            'test': {
                                'bcc': int(np.sum(y_test == 1)),
                                'sk': int(np.sum(y_test == 0))
                            }
                        }
                    },
                    
                    # Model information
                    'model_type': name,
                    'trained_with_cross_validation': not ('skip_cv' in locals() and skip_cv),
                    'small_dataset_adaptations': is_very_small_dataset,
                    
                    # Training metadata
                    'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'training_parameters': {
                        'feature_selection': args.feature_selection,
                        'feature_set': args.feature_set,
                        'num_selected_features': len(selected_feature_names),
                    },
                    
                    # Performance metrics
                    'test_metrics': {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'specificity': float(specificity),
                        'roc_auc': float(roc_auc) if roc_auc is not None else None,
                        'confusion_matrix': conf_matrix.tolist()
                    }
                }
                
                # Add class parameters if available
                if hasattr(clf, 'get_params'):
                    try:
                        metadata['model_parameters'] = clf.get_params()
                    except:
                        metadata['model_parameters'] = str(clf)
                
                # Save metadata to JSON file
                try:
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Model metadata saved to {metadata_path}")
                except Exception as e:
                    logger.error(f"Error saving metadata: {str(e)}")
                    
                # Also save performance metrics separately for easier access and comparison
                try:
                    with open(performance_path, 'w') as f:
                        json.dump(metadata['test_metrics'], f, indent=2)
                    logger.info(f"Performance metrics saved to {performance_path}")
                except Exception as e:
                    logger.error(f"Error saving performance metrics: {str(e)}")
                
                logger.info(f"Model and metadata saved to {model_dir}")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                # Continue with the next classifier
        
        # Generate summary table for all classifiers
        generate_summary_table(results, logger, table_num=5, 
                             title=f"BCC vs SK Detection - Conventional Features ({args.feature_set}) Comparison")
        
        # Plot learning curves for best classifier if we have any successful results
        if results:
            try:
                best_classifier_name = max(results, key=lambda x: results[x]['test_f1'])
                best_classifier = results[best_classifier_name]['classifier']
                
                logger.info(f"Generating learning curves for best classifier: {best_classifier_name}")
                plot_learning_curve(
                    best_classifier, X_train, y_train, cv=5,
                    title=f"Learning Curve - {best_classifier_name} (Conventional Features)",
                    save_path=f"output/metrics/learning_curve_{best_classifier_name.replace(' ', '_')}.png"
                )
            except Exception as e:
                logger.error(f"Error generating learning curves: {str(e)}")
        else:
            logger.warning("No successful classifier results available for learning curve generation")
        
        # Calculate and report total time
        total_time = time.time() - start_time
        logger.info(f"Feature engineering training completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in train_features: {str(e)}")
        traceback.print_exc()


def classify(args, logger):
    """Classify a single image as Basal-cell Carcinoma (BCC) or Seborrheic Keratosis (SK) using trained models with enhanced feature extraction."""
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
            temp_classifier = BCCSKClassifier()
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
                    
                    # Apply feature selection if available, with error handling for dimension mismatch
                    if feature_selector_path and os.path.exists(feature_selector_path):
                        try:
                            feature_selector = load(feature_selector_path)
                            X_selected = feature_selector.transform(X_scaled)
                            logger.info(f"Applied feature selection: using {X_selected.shape[1]} features")
                            
                            # Make prediction with selected features
                            pred = clf.predict(X_selected)[0]
                        except ValueError as e:
                            logger.warning(f"Feature selection error: {str(e)}")
                            logger.info("Falling back to using all scaled features for prediction")
                            pred = clf.predict(X_scaled)[0]
                    else:
                        # Make prediction without feature selection
                        pred = clf.predict(X_scaled)[0]
                    
                    # Get probability if available
                    prob = None
                    if hasattr(clf, 'predict_proba'):
                        try:
                            # Use selected features for probability if available
                            if feature_selector_path and os.path.exists(feature_selector_path) and 'X_selected' in locals():
                                try:
                                    probs = clf.predict_proba(X_selected)
                                    prob = probs[0, 1]  # Probability of positive class (BCC)
                                    logger.info(f"Using feature-selected data for probability calculation")
                                except ValueError as e:
                                    logger.warning(f"Error using selected features for probability: {str(e)}")
                                    logger.info("Falling back to using all scaled features for probability")
                                    probs = clf.predict_proba(X_scaled)
                                    prob = probs[0, 1]  # Probability of positive class (BCC)
                            else:
                                probs = clf.predict_proba(X_scaled)
                                prob = probs[0, 1]  # Probability of positive class (BCC)
                        except Exception as prob_err:
                            logger.warning(f"Could not get probability for {classifier_name}: {str(prob_err)}")
                            
                    # Determine risk level based on probability
                    risk_level = ""
                    explanation = ""
                    
                    if prob is not None:
                        prob_pct = prob * 100
                        if prob_pct > 75:
                            risk_level = "HIGH"
                            explanation = "High probability of Basal-cell Carcinoma (BCC). Immediate medical consultation recommended."
                        elif prob_pct > 50:
                            risk_level = "MODERATE TO HIGH"
                            explanation = "Elevated probability of Basal-cell Carcinoma (BCC). Prompt medical consultation recommended."
                        elif prob_pct > 25:
                            risk_level = "MODERATE"
                            explanation = "Some features suggesting BCC are present. Medical evaluation advised."
                        else:
                            risk_level = "LOW"
                            explanation = "Low probability of BCC. Likely Seborrheic Keratosis (SK). Regular self-examination advised."
                    
                    # Store result with enhanced information
                    results[classifier_name] = {
                        'prediction': 'Basal-cell Carcinoma (BCC)' if pred == 1 else 'Seborrheic Keratosis (SK)',
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
            
            model_path = os.path.join('model', 'CNN', 'bcc_sk_cnn_model.h5')
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"CNN model file not found at {model_path}. Skipping classification.")
            else:
                try:
                    # Import CNN module only when needed
                    from src.cnn_classifier import CNNBCCSKClassifier
                    
                    # Initialize the CNN classifier
                    cnn_classifier = CNNBCCSKClassifier(
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
                        explanation = "High probability of Basal-cell Carcinoma (BCC). Immediate medical consultation recommended."
                    elif prob_pct > 50:
                        risk_level = "MODERATE TO HIGH"
                        explanation = "Elevated probability of Basal-cell Carcinoma (BCC). Prompt medical consultation recommended."
                    elif prob_pct > 25:
                        risk_level = "MODERATE"
                        explanation = "Some features suggesting BCC are present. Medical evaluation advised."
                    else:
                        risk_level = "LOW"
                        explanation = "Low probability of BCC. Likely Seborrheic Keratosis (SK). Regular self-examination advised."
                    
                    # Store result with enhanced information
                    results['CNN'] = {
                        'prediction': 'Basal-cell Carcinoma (BCC)' if pred_class == 1 else 'Seborrheic Keratosis (SK)',
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
            bcc_count = sum(1 for r in results.values() if r['prediction'] == 'Basal-cell Carcinoma (BCC)')
            sk_count = sum(1 for r in results.values() if r['prediction'] == 'Seborrheic Keratosis (SK)')
            total_count = len(results)
            
            # Format results in a table
            logger.info(f"{'Classifier':<15} | {'Prediction':<30} | {'Probability':<12} | {'Risk Level':<15}")
            logger.info("-" * 100)
            
            for classifier_name, result in results.items():
                prob_str = f"{result['probability']:.2f}%" if result['probability'] is not None else "N/A"
                risk_level = result.get('risk_level', 'N/A')
                logger.info(f"{classifier_name:<15} | {result['prediction']:<30} | {prob_str:<12} | {risk_level:<15}")
                
                # Display explanation for risk level if available
                if 'explanation' in result:
                    logger.info(f"    └─ {result['explanation']}")
            
            logger.info("-" * 60)
            logger.info(f"Summary: {bcc_count}/{total_count} classifiers predict Basal-cell Carcinoma (BCC)")
            logger.info(f"         {sk_count}/{total_count} classifiers predict Seborrheic Keratosis (SK)")
            
            # Overall majority prediction
            if bcc_count > sk_count:
                logger.info("\nOverall prediction: BASAL-CELL CARCINOMA (medical consultation recommended)")
            elif sk_count > bcc_count:
                logger.info("\nOverall prediction: SEBORRHEIC KERATOSIS (SK)")
            else:
                logger.info("\nOverall prediction: INCONCLUSIVE (equal votes)")
            
            # Add calculated metrics (for comparison)
            logger.info("\nPerformance Metrics:")
            logger.info("-" * 60)
            logger.info("NOTE: These are not validation metrics, just individual model confidence indicators.")
            
            for classifier_name, result in results.items():
                if result['probability'] is not None:
                    # Calculate metrics based on probability thresholds
                    confidence = result['probability'] / 100 if result['prediction'] == 'Basal-cell Carcinoma (BCC)' else 1 - (result['probability'] / 100)
                    
                    logger.info(f"{classifier_name}:")
                    logger.info(f"  Confidence: {result['probability']:.2f}%")
                    logger.info(f"  Decision threshold: {'>.5' if result['prediction'] == 'Basal-cell Carcinoma (BCC)' else '≤.5'}")
            
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
        estimator: The classifier object (or BCCSKClassifier instance)
        X: Features
        y: Labels
        cv: Cross-validation folds
        n_jobs: Number of parallel jobs
        train_sizes: Array of training sizes to plot
        title: Plot title
        save_path: Path to save the plot
    """
    # If estimator is a BCCSKClassifier, use its underlying classifier
    if hasattr(estimator, 'classifier'):
        estimator = estimator.classifier
    plt.figure(figsize=(10, 6))
    
    # Check if there are multiple classes in the dataset
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # Handle the special case of single-class datasets
        plt.text(0.5, 0.5, "Cannot generate learning curve: Only one class in dataset",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    try:
        # Use StratifiedKFold to ensure each fold has samples of each class
        if isinstance(cv, int):
            # Get the class counts
            class_counts = np.bincount(y.astype(int))
            # Ensure each class has at least 2 samples per fold on average
            # (minimum 2 folds, maximum original cv value)
            safe_n_splits = min(cv, min(class_counts) // 2)
            safe_n_splits = max(2, safe_n_splits)  # At least 2 folds
            
            # Create StratifiedKFold with safe number of splits
            cv = StratifiedKFold(n_splits=safe_n_splits, shuffle=True, random_state=42)
        
        # Set error_score to 'raise' for debugging, then handle exceptions
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, 
            scoring='accuracy', error_score=np.nan, random_state=42
        )
        
        # Filter out NaN values that might result from failed CV folds
        valid_indices = ~np.isnan(train_scores).any(axis=1) & ~np.isnan(test_scores).any(axis=1)
        
        if not valid_indices.any():
            # No valid scores, likely due to single-class folds
            plt.text(0.5, 0.5, "Learning curve generation failed: Check for balanced classes",
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(title)
        else:
            # Use only valid indices
            valid_train_sizes = train_sizes[valid_indices]
            valid_train_scores = train_scores[valid_indices]
            valid_test_scores = test_scores[valid_indices]
            
            train_scores_mean = np.mean(valid_train_scores, axis=1) * 100
            train_scores_std = np.std(valid_train_scores, axis=1) * 100
            test_scores_mean = np.mean(valid_test_scores, axis=1) * 100
            test_scores_std = np.std(valid_test_scores, axis=1) * 100
            
            plt.fill_between(valid_train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(valid_train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            
            plt.plot(valid_train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(valid_train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            
            plt.title(title)
            plt.xlabel("Training examples")
            plt.ylabel("Accuracy (%)")
            plt.legend(loc="best")
            plt.grid(True)
    
    except Exception as e:
        # Handle exceptions with a meaningful error message in the plot
        error_message = f"Learning curve error: {str(e)}"
        if "classes" in str(e) and "one" in str(e):
            error_message = "Error: Some cross-validation folds have only one class"
        
        plt.text(0.5, 0.5, error_message,
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12, wrap=True)
        plt.title(title)
    
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
        clf: Trained classifier (or BCCSKClassifier instance)
        X: Features
        y: True labels
        name: Classifier name
        save_path: Path to save the plot
    """
    # If clf is a BCCSKClassifier, we need to use the custom predict_proba method
    is_bcc_classifier = hasattr(clf, 'predict_proba') and hasattr(clf, 'classifier')
    plt.figure(figsize=(10, 6))
    
    # Check if there are multiple classes in the dataset
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # Handle the special case of single-class datasets
        plt.text(0.5, 0.5, "Cannot generate calibration curve: Only one class in dataset",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title(f"Calibration Curve for {name}")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    try:
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X)[:, 1]
            
            # Calculate Brier score loss
            brier_score = brier_score_loss(y, prob_pos)
            
            # Plot calibration curve
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=min(10, len(y) // 2))
                
                # Plot perfectly calibrated
                plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
                
                # Plot model calibration curve
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                         label=f"{name} (Brier score: {brier_score:.3f})")
                
                plt.ylabel("Fraction of positives (Empirical)")
                plt.xlabel("Mean predicted probability (Model)")
                plt.title(f'Calibration Curve for {name}')
                plt.legend(loc="best")
                plt.grid(True)
            except Exception as e:
                # Handle specific issues with calibration curve calculation
                plt.text(0.5, 0.5, f"Calibration curve error: {str(e)}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title(f"Calibration Issues for {name}")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
        else:
            plt.text(0.5, 0.5, f"Classifier {name} does not support predict_proba",
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title("Calibration Curve Not Available")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        # General exception handling
        plt.text(0.5, 0.5, f"Error generating calibration curve: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title(f"Calibration Error for {name}")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def plot_feature_importance(clf, X, y, feature_names=None, top_n=20, save_path=None):
    """
    Generate and plot feature importance for a classifier.
    
    Args:
        clf: Trained classifier (or BCCSKClassifier instance)
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    # If clf is a BCCSKClassifier, use its underlying classifier
    if hasattr(clf, 'classifier'):
        clf = clf.classifier
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
        clf: Trained classifier (or BCCSKClassifier instance)
        X: Feature matrix
        y: True labels
        save_path: Path to save the plot
    """
    # If clf is a BCCSKClassifier, we'll use its custom predict_proba method
    is_bcc_classifier = hasattr(clf, 'predict_proba') and hasattr(clf, 'classifier')
    plt.figure(figsize=(10, 6))
    
    # Check if there are multiple classes in the dataset
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # Handle the special case of single-class datasets
        plt.text(0.5, 0.5, "Cannot generate prediction histogram: Only one class in dataset",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("Prediction Histogram")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    try:
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(X)
            pos_probs = probas[:, 1]  # Probability of positive class
            
            # Separate probabilities by true class
            pos_probs_pos = pos_probs[y == 1]
            pos_probs_neg = pos_probs[y == 0]
            
            # Calculate appropriate number of bins
            n_bins = min(20, max(5, min(len(pos_probs_pos), len(pos_probs_neg)) // 2))
            
            # Plot histograms - only plot classes that have samples
            if len(pos_probs_pos) > 0:
                plt.hist(pos_probs_pos, bins=n_bins, alpha=0.6, color='red', 
                        label=f'True BCC (n={len(pos_probs_pos)})')
            if len(pos_probs_neg) > 0:
                plt.hist(pos_probs_neg, bins=n_bins, alpha=0.6, color='green', 
                        label=f'True SK (n={len(pos_probs_neg)})')
            
            plt.xlabel('Predicted Probability of Basal-cell Carcinoma (BCC)')
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
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title("Prediction Histogram Not Available")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        # Handle any exceptions
        plt.text(0.5, 0.5, f"Error generating prediction histogram: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("Prediction Histogram Error")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def plot_f1_threshold_curve(clf, X, y, save_path=None):
    """
    Plot F1 score vs decision threshold.
    
    Args:
        clf: Trained classifier (or BCCSKClassifier instance)
        X: Feature matrix
        y: True labels
        save_path: Path to save the plot
    """
    # If clf is a BCCSKClassifier, we'll use its custom predict_proba method
    is_bcc_classifier = hasattr(clf, 'predict_proba') and hasattr(clf, 'classifier')
    plt.figure(figsize=(10, 6))
    
    # Check if there are multiple classes in the dataset
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # Handle the special case of single-class datasets
        plt.text(0.5, 0.5, "Cannot generate F1 threshold curve: Only one class in dataset",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("F1 Threshold Curve")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    try:
        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X)[:, 1]
            
            # Calculate F1 score for different thresholds
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                
                # Use zero_division="warn" to avoid divide by zero errors
                f1 = f1_score(y, y_pred, zero_division="warn")
                precision = precision_score(y, y_pred, zero_division="warn")
                recall = recall_score(y, y_pred, zero_division="warn")
                
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            # Check if we have valid scores (not all zeros)
            if max(f1_scores) > 0:
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
            else:
                # No valid F1 scores found
                plt.text(0.5, 0.5, "Could not generate meaningful F1 curve: All scores are zero",
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title("F1 Threshold Curve - No Valid Scores")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
        else:
            plt.text(0.5, 0.5, "This model does not support predict_proba",
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title("F1 Score Curve Not Available")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        # Handle any exceptions gracefully
        plt.text(0.5, 0.5, f"Error generating F1 threshold curve: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("F1 Threshold Curve Error")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return None

def generate_summary_table(results, logger, table_num=1, title="BCC vs SK Detection Model Comparison Summary"):
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
    
    # Check if we have any columns in our DataFrame
    if len(df.columns) == 0:
        logger.warning("No metrics to visualize - empty results table.")
        return
        
    # Split into performance metrics and feature metrics for better visualization
    performance_metrics = ['Accuracy (%)', 'AUC ROC (%)', 'AUC PR (%)', 'Sensitivity (%)', 
                          'Specificity (%)', 'Precision (%)', 'F1 Score (%)']
    feature_metrics = ['Total Features', 'Selected Features', 'Feature Reduction (%)']
    
    # Filter columns that exist in our dataframe
    perf_cols = [col for col in performance_metrics if col in df.columns]
    feat_cols = [col for col in feature_metrics if col in df.columns]
    
    # If no performance or feature metrics are found, return early
    if not perf_cols and not feat_cols:
        logger.warning("No valid metrics found in results to visualize.")
        return
    
    # Create performance metrics table
    if perf_cols:
        perf_df = df[perf_cols]
        fig, ax = plt.figure(figsize=(12, len(results) * 0.5 + 2)), plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        # Convert values to strings with 2 decimal places to avoid rounding issues
        formatted_values = [[f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                           for val in row] for row in perf_df.values]
        
        table = ax.table(cellText=formatted_values, 
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
        
        # Convert values to strings with 2 decimal places to avoid rounding issues
        formatted_feat_values = [[f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                               for val in row] for row in feat_df.values]
        
        table = ax.table(cellText=formatted_feat_values, 
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
    
    # Convert values to strings with 2 decimal places to avoid rounding issues
    formatted_all_values = [[f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                          for val in row] for row in df.values]
    
    table = ax.table(cellText=formatted_all_values, 
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
    os.makedirs('data/bcc', exist_ok=True)
    os.makedirs('data/sk', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/images', exist_ok=True)
    os.makedirs('output/metrics', exist_ok=True)
    os.makedirs('output/features', exist_ok=True)
    os.makedirs('output/summaries', exist_ok=True)

    logger.info(f"Running in {args.mode} mode")
    
    if args.mode == 'train':
        logger.info(f"Training graph-based models with classifiers: '{args.classifiers}'")
        # Display CNN configuration if applicable
        if 'cnn' in args.classifiers.lower() or args.classifiers.lower() == 'all':
            logger.info(f"CNN configuration: {args.cnn_model} architecture, " +
                       f"{args.input_size}x{args.input_size} input size, " +
                       f"{args.epochs} epochs, batch size {args.batch_size}")
    elif args.mode == 'train_features':
        logger.info(f"Training with conventional feature engineering approach")
        logger.info(f"Feature set: {args.feature_set}, Feature selection: {args.feature_selection}")
        logger.info(f"Classifiers: {args.feature_classifiers}")
        if args.optimize:
            logger.info("Hyperparameter optimization is enabled")

    # Execute based on mode
    try:
        if args.mode == 'train':
            train(args, logger)
        elif args.mode == 'train_features':
            train_features(args, logger)
        elif args.mode == 'classify':
            classify(args, logger)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()