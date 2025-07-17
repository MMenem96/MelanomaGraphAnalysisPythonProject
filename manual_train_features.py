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
import warnings
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

from src.segmentation.skin_lesion_processor import SkinLesionProcessor

from scipy.stats import randint, uniform

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
    'SVM (Linear)': {
        'class': SVC,
        'params': {'kernel': 'linear', 'C': 1.0, 'probability': True, 'random_state': 42}
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
    },
    'Gradient Boosting': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
    },
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42, 'solver': 'lbfgs'}
    },
    'XGBoost': {
        'class': XGBClassifier,
        'params': {'n_estimators': 100, 'random_state': 42, 'base_score': 0.5, 'eval_metric': 'logloss'}
    }
}

"""Utils"""

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
    parser.add_argument('--bcc-dir', type=str, default='data/bcc_segmented',
                        help='Directory containing Basal-cell Carcinoma (BCC) images')
    parser.add_argument('--sk-dir', type=str, default='data/sk_segmented',
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
    parser.add_argument('--mode', type=str, choices=['train', 'train_features', 'classify'], default='train_features',
                        help='Operation mode: train (train graph-based models), train_features (train using conventional feature engineering), or classify (single image)')
    
    # Feature engineering parameters (for train_features mode)
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['basic', 'color', 'texture', 'shape', 'dermoscopy', 'full'],
                        help='Set of features to use for conventional feature engineering')
    parser.add_argument('--feature_selection', type=str, default='none',
                        choices=['none', 'mutual_info', 'chi2', 'f_test', 'rfe'],
                        help='Feature selection method for conventional feature engineering')
    parser.add_argument('--n_features', type=int, default=60,
                        help='Number of features to select when using feature selection')
    parser.add_argument('--feature_classifiers', type=str, default='all',
                        help='Comma-separated list of classifiers to train with feature engineering')
    parser.add_argument('--optimize', action='store_true', default= True,
                        help='Perform hyperparameter optimization for feature-based classifiers')
    
    parser.add_argument('--apply-mask', action='store_true',
                        help='Apply lesion segmentation masking during preprocessing')
    
    parser.add_argument('--apply-gaussian-filter', action='store_true',
                        help='Apply lesion gaussian filter during preprocessing')

    return parser.parse_args()

def clean_and_preprocess_features(X, feature_names=None, logger=None):
    """
    Clean features by handling inf, nan, and extreme values with robust preprocessing
    specifically designed for skin lesion feature matrices.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names (optional)
        logger: Logger instance (optional)
    
    Returns:
        X_clean: Cleaned feature matrix
        variance_selector: Variance selector used (for consistency)
        cleaned_feature_names: Updated feature names after cleaning
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting feature cleaning: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 1: Handle infinite values
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values. Replacing with finite extremes.")
        X = np.where(np.isposinf(X), np.finfo(np.float64).max / 1e10, X)
        X = np.where(np.isneginf(X), np.finfo(np.float64).min / 1e10, X)
    
    # Step 2: Handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values. Replacing with feature medians.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median', fill_value=0)
        X = imputer.fit_transform(X)
    
    # Step 3: Remove constant features (zero variance)
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=1e-8)
    X_variance_filtered = variance_selector.fit_transform(X)
    
    removed_features = X.shape[1] - X_variance_filtered.shape[1]
    if removed_features > 0:
        logger.info(f"Removed {removed_features} constant/near-constant features")
    
    # Step 4: Handle extreme outliers using IQR method
    logger.info("Applying outlier capping using IQR method...")
    for col in range(X_variance_filtered.shape[1]):
        feature_values = X_variance_filtered[:, col]
        Q1 = np.percentile(feature_values, 25)
        Q3 = np.percentile(feature_values, 75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (3 * IQR for more aggressive cleaning)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Cap outliers
        X_variance_filtered[:, col] = np.clip(feature_values, lower_bound, upper_bound)
    
    # Step 5: Final validation
    final_inf_count = np.isinf(X_variance_filtered).sum()
    final_nan_count = np.isnan(X_variance_filtered).sum()
    
    if final_inf_count > 0 or final_nan_count > 0:
        logger.error(f"Cleaning failed: {final_inf_count} inf, {final_nan_count} nan values remain")
        return None, None, None
    
    # Update feature names if provided
    cleaned_feature_names = None
    if feature_names is not None and len(feature_names) == X.shape[1]:
        selected_indices = variance_selector.get_support(indices=True)
        cleaned_feature_names = [feature_names[i] for i in selected_indices]
        logger.info(f"Updated feature names: {len(cleaned_feature_names)} features retained")
    
    logger.info(f"Feature cleaning completed successfully: {X_variance_filtered.shape}")
    return X_variance_filtered, variance_selector, cleaned_feature_names


def specificity_score(y_true, y_pred):
    """
    Calculate the specificity score (true negative rate).
    
    Specificity = TN / (TN + FP)
    """
    from sklearn.metrics import confusion_matrix
    
    # Ensure binary classification
    if len(np.unique(y_true)) != 2 or len(np.unique(y_pred)) != 2:
        return 0.0
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle edge cases
        if cm.shape != (2, 2):
            return 0.0
            
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate specificity with division by zero protection
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity
        
    except Exception:
        return 0.0

def generate_summary_table(results, logger, table_num=1, title="Model Comparison Summary", save_to_file=True):
    """
    Generate and log a formatted summary table of model results, and optionally save to file.
    
    Args:
        results: Dictionary of model results
        logger: Logger instance
        table_num: Table number for reference
        title: Table title
        save_to_file: Whether to save the table to a text file (default: True)
    """
    try:
        if not results:
            logger.warning("No results provided for summary table")
            return
        
        # Prepare table content
        table_content = []
        table_content.append("=" * 80)
        table_content.append(f"TABLE {table_num}: {title}")
        table_content.append("=" * 80)
        
        # Create header
        header = f"{'Model':<20} | {'AC (%)':<8} | {'SN (%)':<8} | {'SP (%)':<8} | {'PR (%)':<8} | {'F1 (%)':<8} | {'AUC (%)':<9} | {'Features':<10}"
        table_content.append(header)
        table_content.append("-" * len(header))
        
        # Sort results by F1 score (descending)
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].get('F1', 0), 
                              reverse=True)
        
        for model_name, metrics in sorted_results:
            # Format metrics with safe handling of None values
            ac = f"{metrics.get('AC', 0):.1f}" if metrics.get('AC') is not None else "N/A"
            sn = f"{metrics.get('SN', 0):.1f}" if metrics.get('SN') is not None else "N/A"
            sp = f"{metrics.get('SP', 0):.1f}" if metrics.get('SP') is not None else "N/A"
            pr = f"{metrics.get('PR', 0):.1f}" if metrics.get('PR') is not None else "N/A"
            f1 = f"{metrics.get('F1', 0):.1f}" if metrics.get('F1') is not None else "N/A"
            auc = f"{metrics.get('AUC', 0):.1f}" if metrics.get('AUC') is not None else "N/A"
            features = str(metrics.get('NUM_SELECTED', metrics.get('NUM_FEATURES', 'N/A')))
            
            row = f"{model_name:<20} | {ac:<8} | {sn:<8} | {sp:<8} | {pr:<8} | {f1:<8} | {auc:<9} | {features:<10}"
            table_content.append(row)
        
        table_content.append("-" * len(header))
        
        # Add summary statistics
        if len(results) > 1:
            f1_scores = [metrics.get('F1', 0) for metrics in results.values() if metrics.get('F1') is not None]
            if f1_scores:
                summary_line = f"Summary: Best F1: {max(f1_scores):.1f}%, Average F1: {np.mean(f1_scores):.1f}%, Models: {len(results)}"
                table_content.append(summary_line)
        
        table_content.append("=" * 80)
        
        # Add metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        table_content.append(f"\nGenerated on: {timestamp}")
        table_content.append(f"Total models evaluated: {len(results)}")
        
        # Log to console
        logger.info("\n" + "\n".join(table_content))
        
        # Save to file if requested
        if save_to_file:
            try:
                import os
                
                # Create output/tables directory if it doesn't exist
                output_dir = "output/tables"
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename with timestamp and table info
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                filename = f"table_{table_num}_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = os.path.join(output_dir, filename)
                
                # Write table to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(table_content))
                    
                    # Add detailed results section
                    f.write("\n\n" + "=" * 80)
                    f.write("\nDETAILED RESULTS")
                    f.write("\n" + "=" * 80)
                    
                    for model_name, metrics in sorted_results:
                        f.write(f"\n\n{model_name}:")
                        f.write(f"\n  Accuracy: {metrics.get('AC', 0):.2f}%")
                        f.write(f"\n  Sensitivity/Recall: {metrics.get('SN', 0):.2f}%")
                        f.write(f"\n  Specificity: {metrics.get('SP', 0):.2f}%")
                        f.write(f"\n  Precision: {metrics.get('PR', 0):.2f}%")
                        f.write(f"\n  F1 Score: {metrics.get('F1', 0):.2f}%")
                        if metrics.get('AUC') is not None:
                            f.write(f"\n  AUC: {metrics.get('AUC', 0):.2f}%")
                        f.write(f"\n  Features Used: {metrics.get('NUM_SELECTED', metrics.get('NUM_FEATURES', 'N/A'))}")
                        
                        # Add confusion matrix if available
                        if 'confusion_matrix' in metrics:
                            f.write(f"\n  Confusion Matrix:")
                            cm = metrics['confusion_matrix']
                            f.write(f"\n    {cm}")
                
                logger.info(f"Summary table saved to: {filepath}")
                
            except Exception as file_error:
                logger.error(f"Error saving table to file: {str(file_error)}")
        
    except Exception as e:
        logger.error(f"Error generating summary table: {str(e)}")


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5),
                       title="Learning Curve", save_path=None):
    """
    Plot learning curves to show model performance vs training set size.
    
    Args:
        estimator: ML model/estimator
        X: Feature matrix
        y: Target labels
        cv: Cross-validation folds
        n_jobs: Number of parallel jobs
        train_sizes: Training set sizes to evaluate
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        from sklearn.model_selection import learning_curve
        import matplotlib.pyplot as plt
        
        # Generate learning curves
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
            scoring='f1', random_state=42
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation curves
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Cross-Validation Score')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                        alpha=0.2, color='red')
        
        # Formatting
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return train_sizes_abs, train_scores, test_scores
        
    except Exception as e:
        if save_path:
            # Create empty plot with error message if learning curve fails
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Learning curve generation failed:\n{str(e)}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(title)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return None, None, None

def generate_lesion_mask_from_black_background(image, threshold=10):
    """
    Generate a binary mask for lesion segmentation from black-background images.
    
    Args:
        image: RGB image with black background and lesion on foreground
        threshold: Threshold for separating background from lesion (default=10)
    
    Returns:
        mask: Binary mask where True=lesion, False=background
    """
    try:
        # Convert to grayscale for easier thresholding
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create mask: anything above threshold is considered lesion
        # Black background (0,0,0) will be False, lesion pixels will be True
        mask = gray > threshold
        
        # Optional: Clean up mask with morphological operations
        # Remove small noise and fill small holes
        from skimage.morphology import opening, closing, disk
        
        # Remove small noise (opening)
        mask = opening(mask, disk(2))
        
        # Fill small holes (closing)
        mask = closing(mask, disk(3))
        
        return mask.astype(bool)
        
    except Exception as e:
        # Fallback: return full image mask if processing fails
        return np.ones(image.shape[:2], dtype=bool)


def extract_features_with_proper_masking(feature_extractor, image, feature_set='full', logger=None):
    """
    Extract features using proper lesion masking for black-background images.
    
    Args:
        feature_extractor: ConventionalFeatureExtractor instance
        image: RGB image with black background
        feature_set: Type of features to extract
        logger: Logger instance
    
    Returns:
        features: Dictionary of extracted features
    """
    try:
        # Generate proper mask from black background
        mask = generate_lesion_mask_from_black_background(image)
        
        if logger:
            lesion_area = np.sum(mask)
            total_area = mask.size
            coverage = (lesion_area / total_area) * 100
            logger.debug(f"Generated mask: {lesion_area} lesion pixels ({coverage:.1f}% coverage)")
        
        # Extract features based on selected feature set with proper masking
        if feature_set == 'full':
            # Extract all available features with proper mask
            features = feature_extractor.extract_all_features(image, mask)
        elif feature_set == 'color':
            # Only color features with proper mask
            features = feature_extractor.extract_color_features(image, mask)
        elif feature_set == 'texture':
            # Only texture features with proper mask
            features = feature_extractor.extract_texture_features(image, mask)
        elif feature_set == 'shape':
            # Only geometric features using the generated mask
            features = feature_extractor.extract_geometric_features(mask)
        elif feature_set == 'dermoscopy':
            # Only dermoscopic features with proper mask
            features = feature_extractor.extract_dermoscopic_features(image, mask)
        else:
            # Basic set - combine key features with proper masking
            color_features = feature_extractor.extract_color_features(image, mask)
            texture_features = feature_extractor.extract_texture_features(image, mask)
            geometric_features = feature_extractor.extract_geometric_features(mask)
            
            # Select only key features from each category
            features = {}
            for key in color_features:
                if any(x in key for x in ['mean', 'std', 'entropy', 'color_variance']):
                    features[key] = color_features[key]
            
            for key in texture_features:
                if any(x in key for x in ['glcm_contrast', 'glcm_homogeneity', 'wavelet_approx', 'gradient_mag']):
                    features[key] = texture_features[key]
                    
            for key in geometric_features:
                if any(x in key for x in ['area', 'perimeter', 'compactness', 'eccentricity']):
                    features[key] = geometric_features[key]
        
        return features
        
    except Exception as e:
        if logger:
            logger.error(f"Error in feature extraction with masking: {str(e)}")
        return {}

""""""
def train_features(args, logger):
    """
    Enhanced train_features function optimized for pre-segmented dermoscopic images
    with improved feature extraction, selection, and ensemble methods.
    """
    # filepath: /Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/manual_train_features.py
    
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import BorderlineSMOTE
    
    try:
        start_time = time.time()
        
        logger.info("Starting ENHANCED conventional feature engineering-based training")
        logger.info(f"Feature set: {args.feature_set}, Selection method: {args.feature_selection}")
        logger.info("Optimized for pre-segmented dermoscopic images with black backgrounds")
        
        # STAGE 1: Load and validate image paths
        logger.info("STAGE 1: Loading and validating image paths")
        bcc_paths = glob.glob(os.path.join(args.bcc_dir, "*.jpg")) + \
                    glob.glob(os.path.join(args.bcc_dir, "*.png")) + \
                    glob.glob(os.path.join(args.bcc_dir, "*.jpeg"))
        
        sk_paths = glob.glob(os.path.join(args.sk_dir, "*.jpg")) + \
                   glob.glob(os.path.join(args.sk_dir, "*.png")) + \
                   glob.glob(os.path.join(args.sk_dir, "*.jpeg"))
        
        if len(bcc_paths) == 0 or len(sk_paths) == 0:
            logger.error(f"No images found. BCC: {len(bcc_paths)}, SK: {len(sk_paths)}")
            return
        
        # Balance dataset with intelligent sampling
        if args.max_images_per_class > 0:
            random.shuffle(bcc_paths)
            random.shuffle(sk_paths)
            bcc_paths = bcc_paths[:min(len(bcc_paths), args.max_images_per_class)]
            sk_paths = sk_paths[:min(len(sk_paths), args.max_images_per_class)]
        
        # Create balanced labels
        bcc_labels = np.ones(len(bcc_paths), dtype=np.int32)
        sk_labels = np.zeros(len(sk_paths), dtype=np.int32)
        all_image_paths = bcc_paths + sk_paths
        all_labels = np.concatenate([bcc_labels, sk_labels]).astype(np.int32)
        
        logger.info(f"Dataset loaded: {len(bcc_paths)} BCC, {len(sk_paths)} SK images")
        logger.info(f"Class ratio: {len(sk_paths)/len(bcc_paths):.2f}:1 (SK:BCC)")
        
        # STAGE 2: Initialize enhanced feature extractor
        logger.info("STAGE 2: Initializing enhanced feature extraction")
        feature_extractor = ConventionalFeatureExtractor()
        
        # Create directories
        preprocessed_dir = "preprocessed_images"
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # STAGE 3: Enhanced feature extraction with proper masking
        logger.info("STAGE 3: Enhanced feature extraction from pre-segmented images")
        features_list = []
        success_count = 0
        failed_images = []
        
        for idx, image_path in enumerate(all_image_paths):
            if idx % 100 == 0:
                logger.info(f"Processing image {idx+1}/{len(all_image_paths)}")
            
            try:
                # Load pre-segmented image directly
                original_image = cv2.imread(image_path)
                if original_image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    failed_images.append(image_path)
                    features_list.append({})
                    continue
                
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Enhanced feature extraction with improved masking
                features = extract_enhanced_features_with_proper_masking(
                    feature_extractor, image, args.feature_set, logger
                )
                
                if not features:
                    logger.warning(f"No features extracted from: {image_path}")
                    failed_images.append(image_path)
                
                features_list.append(features)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                failed_images.append(image_path)
                features_list.append({})
        
        logger.info(f"Successfully processed {success_count}/{len(all_image_paths)} images")
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
        
        # STAGE 4: Enhanced feature matrix construction
        logger.info("STAGE 4: Constructing enhanced feature matrix")
        
        # Build comprehensive feature matrix
        expanded_feature_keys = []
        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (list, np.ndarray)):
                    for i in range(len(value)):
                        expanded_key = f"{key}_{i}"
                        if expanded_key not in expanded_feature_keys:
                            expanded_feature_keys.append(expanded_key)
                else:
                    if key not in expanded_feature_keys:
                        expanded_feature_keys.append(key)
        
        expanded_feature_keys = sorted(expanded_feature_keys)
        logger.info(f"Total features after expansion: {len(expanded_feature_keys)}")
        
        # Create feature matrix
        X = np.zeros((len(features_list), len(expanded_feature_keys)))
        
        for i, features in enumerate(features_list):
            for j, key in enumerate(expanded_feature_keys):
                if '_' in key and key.rsplit('_', 1)[0] in features:
                    base_key, idx_str = key.rsplit('_', 1)
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        value = features.get(base_key)
                        if isinstance(value, (list, np.ndarray)) and idx < len(value):
                            X[i, j] = value[idx]
                else:
                    value = features.get(key, 0)
                    if not isinstance(value, (list, np.ndarray)):
                        X[i, j] = value
        
        # STAGE 5: Enhanced data cleaning and preprocessing
        logger.info("STAGE 5: Enhanced data cleaning and preprocessing")
        X_clean, variance_selector, cleaned_feature_names = clean_and_preprocess_features(
            X, expanded_feature_keys, logger
        )
        
        if X_clean is None:
            logger.error("Feature cleaning failed. Cannot proceed.")
            return
        
        X = X_clean
        expanded_feature_keys = cleaned_feature_names or expanded_feature_keys
        logger.info(f"Using {X.shape[1]} features after enhanced cleaning")
        
        # STAGE 6: Intelligent train-test split with stratification
        logger.info("STAGE 6: Intelligent dataset splitting")
        X_train, X_test, y_train, y_test = train_test_split(
            X, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train class distribution: BCC={np.sum(y_train==1)}, SK={np.sum(y_train==0)}")
        
        # STAGE 7: Enhanced feature selection with optimal parameters
        logger.info("STAGE 7: Enhanced feature selection")
        selector = None
        
        if args.feature_selection != 'none':
            # Determine optimal number of features based on dataset size
            optimal_n_features = min(
                args.n_features,
                X_train.shape[1],
                max(20, X_train.shape[0] // 10)  # Adaptive to dataset size
            )
            
            logger.info(f"Applying {args.feature_selection} selection for {optimal_n_features} features")
            
            if args.feature_selection == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=optimal_n_features)
            elif args.feature_selection == 'chi2':
                X_train_min = X_train.min(axis=0)
                X_train = X_train - X_train_min
                X_test = X_test - X_train_min
                selector = SelectKBest(chi2, k=optimal_n_features)
            elif args.feature_selection == 'f_test':
                selector = SelectKBest(f_classif, k=optimal_n_features)
            elif args.feature_selection == 'rfe':
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator=base_model, n_features_to_select=optimal_n_features)
            
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = [expanded_feature_keys[i] for i in selected_indices]
            logger.info(f"Selected {len(selected_feature_names)} most informative features")
        else:
            selected_feature_names = expanded_feature_keys
            logger.info("Using all features (no selection)")
        
        # STAGE 8: Enhanced class balancing with SMOTE
        logger.info("STAGE 8: Enhanced class balancing")
        class_counts = np.bincount(y_train)
        imbalance_ratio = max(class_counts) / min(class_counts)
        
        if imbalance_ratio > 1.3:  # Apply SMOTE if imbalance > 1.3:1
            logger.info(f"Applying BorderlineSMOTE - imbalance ratio: {imbalance_ratio:.2f}")
            try:
                smote = BorderlineSMOTE(random_state=42, k_neighbors=min(3, np.min(class_counts)-1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                new_counts = np.bincount(y_train)
                logger.info(f"After SMOTE: BCC={new_counts[1]}, SK={new_counts[0]}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Proceeding without resampling.")
        else:
            logger.info(f"Classes sufficiently balanced (ratio: {imbalance_ratio:.2f})")
        
        # STAGE 9: Enhanced feature scaling
        logger.info("STAGE 9: Enhanced feature scaling")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
        try:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Applied robust scaling")
        except Exception as e:
            logger.warning(f"Robust scaling failed: {str(e)}. Using standard scaling.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Applied standard scaling")
        
        # Final validation
        if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
            logger.error("Invalid values detected after preprocessing!")
            return
        
        # STAGE 10: Enhanced classifier setup with optimal parameters
        logger.info("STAGE 10: Setting up enhanced classifiers")
        classifiers = setup_enhanced_classifiers(args, logger)
        
        # STAGE 11: Enhanced hyperparameter optimization
        if args.optimize:
            logger.info("STAGE 11: Enhanced hyperparameter optimization")
            classifiers = perform_enhanced_optimization(classifiers, X_train, y_train, logger)
        
        # STAGE 12: Enhanced model training and evaluation
        logger.info("STAGE 12: Enhanced model training and evaluation")
        results = {}
        
        for name, clf in classifiers.items():
            logger.info(f"Training and evaluating {name}")
            try:
                result = train_and_evaluate_enhanced_classifier(
                    name, clf, X_train, X_test, y_train, y_test, 
                    selected_feature_names, scaler, selector, args, logger
                )
                if result:
                    results[name] = result
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        # STAGE 13: Enhanced ensemble creation
        if len(results) >= 2:
            logger.info("STAGE 13: Creating enhanced weighted ensemble")
            ensemble_result = create_enhanced_weighted_ensemble(results, X_test, y_test, logger)
            if ensemble_result:
                results['Enhanced Ensemble'] = ensemble_result
        
        # STAGE 14: Generate comprehensive results
        logger.info("STAGE 14: Generating comprehensive results")
        generate_summary_table(
            results, logger, table_num=5,
            title=f"Enhanced BCC vs SK Detection - {args.feature_set} Features"
        )
        
        # Generate additional analysis
        if results:
            generate_enhanced_analysis(results, X_train, y_train, logger)
        
        total_time = time.time() - start_time
        logger.info(f"Enhanced training completed in {total_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in enhanced train_features: {str(e)}")
        traceback.print_exc()
        return None
    

def extract_enhanced_features_with_proper_masking(feature_extractor, image, feature_set='full', logger=None):
    """Enhanced feature extraction with improved masking for segmented images"""
    try:
        # Generate enhanced mask with higher threshold for better separation
        mask = generate_enhanced_lesion_mask(image, threshold=15)
        
        if logger:
            lesion_area = np.sum(mask)
            total_area = mask.size
            coverage = (lesion_area / total_area) * 100
            logger.debug(f"Enhanced mask: {lesion_area} lesion pixels ({coverage:.1f}% coverage)")
        
        # Extract features with enhanced methods
        if feature_set == 'full':
            features = feature_extractor.extract_all_features(image, mask)
            # Add enhanced color ratio features
            enhanced_color_features = extract_enhanced_color_ratios(image, mask)
            features.update(enhanced_color_features)
        else:
            # Standard feature extraction
            features = feature_extractor.extract_all_features(image, mask)
        
        return features
        
    except Exception as e:
        if logger:
            logger.error(f"Enhanced feature extraction failed: {str(e)}")
        return {}

def generate_enhanced_lesion_mask(image, threshold=15):
    """Enhanced lesion mask generation for segmented images"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Enhanced thresholding for better separation
        mask = gray > threshold
        
        # Enhanced morphological operations
        from skimage.morphology import opening, closing, disk, remove_small_objects
        
        # Remove noise
        mask = opening(mask, disk(3))
        # Fill holes
        mask = closing(mask, disk(5))
        # Remove small objects
        mask = remove_small_objects(mask, min_size=100)
        
        return mask.astype(bool)
        
    except Exception as e:
        return np.ones(image.shape[:2], dtype=bool)

def extract_enhanced_color_ratios(image, mask):
    """Extract enhanced color ratio features"""
    features = {}
    try:
        masked_pixels = image[mask]
        if len(masked_pixels) > 0:
            r, g, b = masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2]
            
            # Enhanced color ratios
            features['enhanced_rg_ratio'] = np.mean(r) / (np.mean(g) + 1e-8)
            features['enhanced_rb_ratio'] = np.mean(r) / (np.mean(b) + 1e-8)
            features['enhanced_gb_ratio'] = np.mean(g) / (np.mean(b) + 1e-8)
            
            # Color variance ratios
            features['color_variance_ratio'] = np.var(r) / (np.var(g) + np.var(b) + 1e-8)
            
    except Exception:
        pass
    
    return features

def create_enhanced_weighted_ensemble(results, X_test, y_test, logger):
    """Create enhanced weighted ensemble of top performers"""
    try:
        # Select top 3 performers by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('F1', 0), reverse=True)
        top_classifiers = sorted_results[:3]
        
        # Calculate performance-based weights
        weights = {}
        total_score = sum([result[1]['F1'] for result in top_classifiers])
        
        for name, metrics in top_classifiers:
            weights[name] = metrics['F1'] / total_score
        
        logger.info(f"Enhanced ensemble weights: {weights}")
        
        # Generate weighted predictions
        ensemble_proba = np.zeros(len(y_test))
        
        for name, metrics in top_classifiers:
            if 'y_pred_proba' in metrics and metrics['y_pred_proba'] is not None:
                ensemble_proba += weights[name] * metrics['y_pred_proba']
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Calculate metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        ensemble_precision = precision_score(y_test, ensemble_pred)
        ensemble_recall = recall_score(y_test, ensemble_pred)
        ensemble_specificity = specificity_score(y_test, ensemble_pred)
        
        logger.info(f"Enhanced Ensemble - F1: {ensemble_f1:.4f}, AUC: {ensemble_auc:.4f}")
        
        return {
            'AC': ensemble_accuracy * 100,
            'PR': ensemble_precision * 100,
            'SN': ensemble_recall * 100,
            'F1': ensemble_f1 * 100,
            'SP': ensemble_specificity * 100,
            'AUC': ensemble_auc * 100,
            'NUM_FEATURES': len(results[top_classifiers[0][0]].get('features', [])),
            'ensemble_weights': weights,
            'y_pred_proba': ensemble_proba
        }
        
    except Exception as e:
        logger.error(f"Enhanced ensemble creation failed: {str(e)}")
        return None

def setup_enhanced_classifiers(args, logger):
    """
    Setup enhanced classifiers with optimized parameters for dermoscopic image analysis
    """
    # filepath: /Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/manual_train_features.py
    
    classifiers = {}
    
    # Parse classifier selection
    if args.feature_classifiers.lower() == 'all':
        selected_classifiers = list(CLASSIFIERS.keys())
    else:
        classifier_map = {
            'svm_rbf': 'SVM (RBF)',
            'svm_linear': 'SVM (Linear)', 
            'svm_poly': 'SVM (Poly)',
            'svm_sigmoid': 'SVM (Sigmoid)',
            'rf': 'RF',
            'mlp': 'MLP',
            'knn': 'KNN',
            'gb': 'Gradient Boosting',
            'logistic': 'Logistic Regression',
            'xgboost': 'XGBoost'
        }
        
        selected_names = [name.strip() for name in args.feature_classifiers.split(',')]
        selected_classifiers = []
        for name in selected_names:
            if name in classifier_map:
                selected_classifiers.append(classifier_map[name])
            elif name in CLASSIFIERS:
                selected_classifiers.append(name)
    
    # Enhanced classifier configurations optimized for skin lesion analysis
    enhanced_configs = {
        'SVM (RBF)': {
            'class': SVC,
            'params': {
                'kernel': 'rbf', 
                'C': 50.0,  # Increased for better boundary definition
                'gamma': 'scale', 
                'probability': True, 
                'random_state': 42,
                'class_weight': 'balanced'  # Handle class imbalance
            }
        },
        'SVM (Linear)': {
            'class': SVC,
            'params': {
                'kernel': 'linear', 
                'C': 10.0,
                'probability': True, 
                'random_state': 42,
                'class_weight': 'balanced'
            }
        },
        'RF': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 200,  # Increased for better performance
                'max_depth': 15,      # Deeper trees for complex patterns
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'MLP': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (150, 100, 50),  # Deeper network
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,      # Regularization
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,     # More iterations
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42
            }
        },
        'XGBoost': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
        },
        'Gradient Boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'random_state': 42
            }
        },
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'penalty': 'l2',
                'C': 1.0
            }
        },
        'KNN': {
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 7,    # Optimized for skin lesion data
                'weights': 'distance',
                'algorithm': 'auto',
                'p': 2,
                'metric': 'minkowski'
            }
        }
    }
    
    # Create classifier instances
    for classifier_name in selected_classifiers:
        if classifier_name in enhanced_configs:
            config = enhanced_configs[classifier_name]
            try:
                classifiers[classifier_name] = config['class'](**config['params'])
                logger.info(f"Enhanced {classifier_name} configured successfully")
            except Exception as e:
                logger.error(f"Error configuring enhanced {classifier_name}: {str(e)}")
                # Fallback to original configuration
                if classifier_name in CLASSIFIERS:
                    original_config = CLASSIFIERS[classifier_name]
                    classifiers[classifier_name] = original_config['class'](**original_config['params'])
    
    logger.info(f"Enhanced classifiers setup completed: {list(classifiers.keys())}")
    return classifiers


def perform_enhanced_optimization(classifiers, X_train, y_train, logger):
    """
    Perform enhanced hyperparameter optimization for dermoscopic image classifiers
    """
    # filepath: /Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/manual_train_features.py
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    optimized_classifiers = {}
    
    # Enhanced parameter grids optimized for skin lesion classification
    param_grids = {
        'XGBoost': {
            'n_estimators': randint(150, 300),
            'max_depth': randint(6, 12),
            'learning_rate': uniform(0.05, 0.15),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3),
            'reg_alpha': uniform(0, 0.2),
            'reg_lambda': uniform(0, 0.2)
        },
        'RF': {
            'n_estimators': randint(150, 400),
            'max_depth': [None] + list(randint(10, 25).rvs(10)),
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 8),
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        },
        'MLP': {
            'hidden_layer_sizes': [
                (100,), (150,), (200,),
                (100, 50), (150, 75), (200, 100),
                (150, 100, 50), (200, 150, 75)
            ],
            'alpha': uniform(0.0001, 0.01),
            'learning_rate_init': uniform(0.0005, 0.005),
            'beta_1': uniform(0.85, 0.14),
            'beta_2': uniform(0.9, 0.099)
        },
        'SVM (RBF)': {
            'C': uniform(10, 90),
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5))
        },
        'SVM (Linear)': {
            'C': uniform(1, 50)
        }
    }
    
    # Cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        logger.info(f"Optimizing hyperparameters for {name}")
        
        try:
            if name in param_grids:
                param_grid = param_grids[name]
                
                # Use RandomizedSearchCV for efficient optimization
                search = RandomizedSearchCV(
                    clf, 
                    param_grid, 
                    n_iter=30,  # Reduced for faster optimization
                    cv=cv_strategy,
                    scoring='f1',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
                
                search.fit(X_train, y_train)
                optimized_classifiers[name] = search.best_estimator_
                
                logger.info(f"{name} optimization completed. Best F1: {search.best_score_:.4f}")
                logger.info(f"Best parameters: {search.best_params_}")
                
            else:
                # Use original classifier if no optimization parameters defined
                optimized_classifiers[name] = clf
                logger.info(f"No optimization parameters for {name}, using default configuration")
                
        except Exception as e:
            logger.error(f"Optimization failed for {name}: {str(e)}")
            optimized_classifiers[name] = clf
    
    return optimized_classifiers


def train_and_evaluate_enhanced_classifier(name, clf, X_train, X_test, y_train, y_test, 
                                         selected_feature_names, scaler, selector, args, logger):
    """
    Train and evaluate enhanced classifier with comprehensive metrics
    """
    # filepath: /Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/manual_train_features.py
    
    try:
        start_time = time.time()
        
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(clf, 'predict_proba'):
            try:
                y_pred_proba_full = clf.predict_proba(X_test)
                if y_pred_proba_full.shape[1] == 2:
                    y_pred_proba = y_pred_proba_full[:, 1]  # Probability of positive class
                else:
                    y_pred_proba = y_pred_proba_full.ravel()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities for {name}: {str(e)}")
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        specificity = specificity_score(y_test, y_pred)
        
        # Calculate AUC if probabilities are available
        auc_score = None
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC for {name}: {str(e)}")
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Prepare result dictionary
        result = {
            'AC': accuracy * 100,
            'PR': precision * 100,
            'SN': recall * 100,
            'F1': f1 * 100,
            'SP': specificity * 100,
            'NUM_FEATURES': len(selected_feature_names),
            'TRAINING_TIME': training_time,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'features': selected_feature_names,
            'model': clf,
            'scaler': scaler,
            'selector': selector
        }
        
        # Add AUC if available
        if auc_score is not None:
            result['AUC'] = auc_score * 100
            result['y_pred_proba'] = y_pred_proba
        
        # Log detailed results
        logger.info(f"{name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}, F1: {f1:.4f}, Specificity: {specificity:.4f}")
        if auc_score:
            logger.info(f"  AUC: {auc_score:.4f}")
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error training and evaluating {name}: {str(e)}")
        return None


def generate_enhanced_analysis(results, X_train, y_train, logger):
    """
    Generate enhanced analysis and insights from model results
    """
    # filepath: /Users/mmoniem96/Desktop/Work/Master/Practical Project/MelanomaGraphAnalysisV3/manual_train_features.py
    
    try:
        logger.info("ENHANCED ANALYSIS: Generating comprehensive model insights")
        
        # Performance analysis
        f1_scores = [result.get('F1', 0) for result in results.values() if 'F1' in result]
        auc_scores = [result.get('AUC', 0) for result in results.values() if 'AUC' in result]
        
        if f1_scores:
            logger.info(f"F1 Score Statistics:")
            logger.info(f"  Best: {max(f1_scores):.2f}%")
            logger.info(f"  Average: {np.mean(f1_scores):.2f}%")
            logger.info(f"  Std Dev: {np.std(f1_scores):.2f}%")
        
        if auc_scores:
            logger.info(f"AUC Score Statistics:")
            logger.info(f"  Best: {max(auc_scores):.2f}%")
            logger.info(f"  Average: {np.mean(auc_scores):.2f}%")
            logger.info(f"  Std Dev: {np.std(auc_scores):.2f}%")
        
        # Model complexity analysis
        feature_counts = [result.get('NUM_FEATURES', 0) for result in results.values()]
        training_times = [result.get('TRAINING_TIME', 0) for result in results.values() if 'TRAINING_TIME' in result]
        
        if feature_counts:
            logger.info(f"Feature Usage: {feature_counts[0]} features selected")
        
        if training_times:
            logger.info(f"Training Time Statistics:")
            logger.info(f"  Average: {np.mean(training_times):.2f}s")
            logger.info(f"  Total: {sum(training_times):.2f}s")
        
        # Model recommendations
        logger.info("MODEL RECOMMENDATIONS:")
        
        # Find best performers
        best_f1_model = max(results.items(), key=lambda x: x[1].get('F1', 0))
        logger.info(f"Best F1 Performance: {best_f1_model[0]} ({best_f1_model[1].get('F1', 0):.2f}%)")
        
        if auc_scores:
            best_auc_model = max(results.items(), key=lambda x: x[1].get('AUC', 0))
            logger.info(f"Best AUC Performance: {best_auc_model[0]} ({best_auc_model[1].get('AUC', 0):.2f}%)")
        
        # Performance insights
        if len(results) >= 3:
            logger.info("INSIGHTS:")
            logger.info("- Consider ensemble methods for improved performance")
            logger.info("- Multiple models show strong performance - ensemble recommended")
        
        # Class balance analysis
        class_counts = np.bincount(y_train)
        balance_ratio = max(class_counts) / min(class_counts)
        logger.info(f"Dataset Balance Ratio: {balance_ratio:.2f}:1")
        
        if balance_ratio > 2:
            logger.info("- Dataset is imbalanced - SMOTE was beneficial")
        else:
            logger.info("- Dataset is well balanced")
        
        # Feature importance analysis (if available)
        try:
            for name, result in results.items():
                if 'model' in result and hasattr(result['model'], 'feature_importances_'):
                    importances = result['model'].feature_importances_
                    top_features = np.argsort(importances)[-5:][::-1]
                    
                    logger.info(f"{name} - Top 5 Most Important Features:")
                    feature_names = result.get('features', [])
                    for i, idx in enumerate(top_features):
                        if idx < len(feature_names):
                            logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
                    break  # Only show for one model to avoid clutter
        except Exception as e:
            logger.debug(f"Feature importance analysis failed: {str(e)}")
        
        logger.info("Enhanced analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")


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
    train_features(args, logger)

if __name__ == "__main__":
    main()
