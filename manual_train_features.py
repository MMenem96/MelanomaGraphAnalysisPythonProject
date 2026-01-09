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
from src.tabular_dnn_classifier import TabularDNNClassifier

from src.segmentation.skin_lesion_processor import SkinLesionProcessor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Dictionary of available classifiers
# ...existing imports...

# Dictionary of available classifiers
CLASSIFIERS = {
    'CatBoost': {
        'class': CatBoostClassifier,
        'params': {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 5,
            'random_strength': 1.5,
            'bagging_temperature': 1.0,
            'border_count': 128,
            'random_state': 42,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'task_type': 'CPU',
            'eval_metric': 'F1'
        }
    },
    'LightGBM': {
        'class': LGBMClassifier,
        'params': {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True,
            'feature_pre_filter': False
        }
    },
    'XGBoost': {
        'class': XGBClassifier,
        'params': {
            'n_estimators': 800,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.2,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': 'logloss',
            'enable_categorical': False,
            'tree_method': 'hist'
        }
    },
    'Extra Trees': {
        'class': ExtraTreesClassifier,
        'params': {
            'n_estimators': 500,
            'max_depth': 25,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'RF': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 500,
            'max_depth': 25,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'SVM (RBF)': {
        'class': SVC,
        'params': {
            'C': 100,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'cache_size': 1000,
            'probability': True,  # Enable probability estimates for ROC curves
            'random_state': 42
        }
    },
    'SVM (Linear)': {
        'class': LinearSVC,
        'params': {
            'C': 10,
            'class_weight': 'balanced',
            'max_iter': 5000,
            'dual': False,
            'random_state': 42
        }
    },
    'SVM (Sigmoid)': {
        'class': SVC,
        'params': {
            'C': 50,
            'kernel': 'sigmoid',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'cache_size': 1000,
            'probability': True,
            'random_state': 42
        }
    },
    'SVM (Poly)': {
        'class': SVC,
        'params': {
            'C': 50,
            'kernel': 'poly',
            'degree': 3,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'cache_size': 1000,
            'probability': True,
            'random_state': 42
        }
    },
    'Logistic Regression': {
        'class': CalibratedClassifierCV,
        'params': {
            'estimator': LogisticRegression(
                C=10.0,
                max_iter=3000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42
            ),
            'method': 'sigmoid',
            'cv': 5
        }
    },
    'MLP': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (100, 50),
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }
    },
    'KNN': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean',
            'algorithm': 'auto',
            'n_jobs': -1
        }
    },
    'Gradient Boosting': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }
    },
    'Calibrated SVM': {
        'class': CalibratedClassifierCV,
        'params': {
            'estimator': LinearSVC(
                C=10,
                class_weight='balanced',
                max_iter=5000,
                dual=False,
                random_state=42
            ),
            'method': 'sigmoid',
            'cv': 5
        }
    },
    'Deep DNN': {
        'class': TabularDNNClassifier,
        'params': {
            'input_dim': 511,  # Will be updated dynamically based on actual features
            'hidden_units': [512, 512, 384, 384, 256, 256, 128],
            'dropout_rate': 0.3,
            'l2_reg': 1e-4,
            'activation': 'swish',
            'use_attention': True,
            'learning_rate': 1e-3,
            'batch_size': 128,
            'epochs': 200,
            'patience': 30,
            'focal_loss_alpha': 0.16,  # 16.4% minority class (BCC)
            'focal_loss_gamma': 2.0,
            'mixup_alpha': 0.2,
            'validation_split': 0.2,
            'verbose': 0,  # Set to 1 for progress bar
            'random_state': 42
        }
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

    #Feature file operations
    parser.add_argument('--features-file', type=str, default=None,
                        help='Path to saved features file for fast training')
    parser.add_argument('--save-features-only', action='store_true',
                        help='Only extract and save features, skip model training')
    parser.add_argument('--train-from-features', action='store_true',
                        help='Train models using previously saved features')
    
    #Custom model parameters
    parser.add_argument('--custom-params', type=str, default=None,
                        help='JSON file with custom model parameters')

    # Dataset paths
    parser.add_argument('--bcc-dir', type=str, default='data/bcc_segmented_augmented',
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
    parser.add_argument('--n_features', type=int, default=350,
                        help='Number of features to select when using feature selection')
    parser.add_argument('--feature_classifiers', type=str, default='all',
                        help='Comma-separated list of classifiers to train with feature engineering')
    parser.add_argument('--optimize', action='store_true', default= False,
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

def generate_lesion_mask_from_transparent_background(image, threshold=10):
    """
    Generate a binary mask for lesion segmentation from transparent-background images.
    """
    try:
        # If image has alpha channel (RGBA)
        if image.shape[2] == 4:
            # Use alpha channel directly as mask
            alpha_channel = image[:, :, 3]
            mask = alpha_channel > threshold
            return mask.astype(bool)
        
        # If RGB image with white background
        elif len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # For white background: anything NOT white is lesion
            # White pixels have value ~255, lesion pixels have lower values
            mask = gray < (255 - threshold)
            
            # Clean up mask
            from skimage.morphology import opening, closing, disk, remove_small_objects
            mask = opening(mask, disk(2))
            mask = closing(mask, disk(3))
            mask = remove_small_objects(mask, min_size=100)
            
            return mask.astype(bool)
        
        # Fallback for grayscale
        else:
            mask = image < (255 - threshold)
            return mask.astype(bool)
            
    except Exception as e:
        # Fallback: return full image mask
        return np.ones(image.shape[:2], dtype=bool)

""""""
def load_image_with_transparency_support(image_path):
    """Load PNG with transparency, convert transparent areas to white background."""
    try:
        from PIL import Image
        import numpy as np
        
        # Load with PIL to preserve transparency
        pil_image = Image.open(image_path)
        
        if pil_image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            # Paste lesion onto white background using alpha mask
            background.paste(pil_image, mask=pil_image.split()[-1])
            return np.array(background)
        else:
            return np.array(pil_image.convert('RGB'))
            
    except Exception as e:
        # Fallback to OpenCV if PIL fails
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
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

    #Initializing the lesion segmenter
    segmenter = SkinLesionProcessor() 

    # Explicitly import the train_test_split function to make sure it's in scope
    from sklearn.model_selection import train_test_split
    try:
        # Start timer
        start_time = time.time()
        
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
        
        # Create directory for saving preprocessed images
        preprocessed_dir = "preprocessed_images"
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # Track saved images for sampling
        saved_bcc_count = 0
        saved_sk_count = 0
        max_sk_samples_per_class = 5 
        max_bcc_samples_per_class = 5 
        
        # Extract features from all images
        logger.info("Extracting features from images...")
        features_list = []
        success_count = 0
        
        for idx, image_path in enumerate(all_image_paths):
            if idx % 100 == 0:
                logger.info(f"Processing image {idx+1}/{len(all_image_paths)}")
            
            try:
                original_image = load_image_with_transparency_support(image_path)
                if original_image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue
                 # Step 1: Convert to grayscale for hair detection
                grayscale_image = segmenter.convert_to_grayscale(original_image)
                
                # Step 2: Apply hair detection and removal
                combined_hair_mask, blackhat_image, tophat_image = segmenter.apply_combined_hair_detection(grayscale_image)
                
                # Step 3: Apply inpainting to remove detected hairs
                inpainted_image = segmenter.apply_inpainting(original_image, combined_hair_mask)
                
                # Step 4: Apply Gaussian blur for smoothing
                image = segmenter.apply_gaussian_blur(inpainted_image)
                
                # Save first 5 preprocessed images from each class for visualization
                current_label = all_labels[idx]
                image_filename = os.path.basename(image_path)
                # Remove file extension from original name
                original_name = os.path.splitext(image_filename)[0]
                class_name = "BCC" if current_label == 1 else "SK"
                
                if (current_label == 1 and saved_bcc_count < max_bcc_samples_per_class) or \
                   (current_label == 0 and saved_sk_count < max_sk_samples_per_class):
                    
                    # Convert images back to BGR for saving with OpenCV
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    
                    # Create filename with class prefix and original name
                    sample_number = saved_bcc_count + 1 if current_label == 1 else saved_sk_count + 1
                    
                    # Create side-by-side comparison image
                    height, width = original_bgr.shape[:2]
                    
                    # Create combined image (side by side)
                    combined_img = np.zeros((height, width * 2, 3), dtype=np.uint8)
                    combined_img[:, :width] = original_bgr
                    combined_img[:, width:] = image_bgr
                    
                    # Add text labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_color = (255, 255, 255)  # White text
                    font_thickness = 2
                    
                    # Add "Original" label
                    cv2.putText(combined_img, "Original", (10, 30), font, font_scale, font_color, font_thickness)
                    
                    # Add "Preprocessed" label
                    cv2.putText(combined_img, "Preprocessed", (width + 10, 30), font, font_scale, font_color, font_thickness)
                    
                    # Save combined image
                    combined_filename = f"{class_name}_{sample_number}_comparison_{original_name}.jpg"
                    combined_save_path = os.path.join(preprocessed_dir, combined_filename)
                    cv2.imwrite(combined_save_path, combined_img)
                    
                    # Update counters
                    if current_label == 1:
                        saved_bcc_count += 1
                        # logger.info(f"Saved preprocessed BCC image: {save_filename}")
                    else:
                        saved_sk_count += 1
                        # logger.info(f"Saved preprocessed SK image: {save_filename}")
                
               
                logger.debug(f"Extracting {args.feature_set} features with proper lesion masking")
                mask = generate_lesion_mask_from_transparent_background(image, threshold=10)
                # Validate mask
                if np.sum(mask) < 100:  # Less than 100 pixels
                    logger.warning(f"Mask too small for {image_path}: {np.sum(mask)} pixels")
                    continue

                # Optional: Save mask visualization for debugging
                if saved_bcc_count < 2 or saved_sk_count < 2:
                    mask_vis = (mask * 255).astype(np.uint8)
                    mask_save_path = os.path.join(preprocessed_dir, f"{class_name}_{sample_number}_mask.jpg")
                    cv2.imwrite(mask_save_path, mask_vis)
                    
                features = feature_extractor.extract_all_features(image, mask)
                
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
        
        # Save feature names to file for reference
        os.makedirs(preprocessed_dir, exist_ok=True)  # Ensure directory exists
        feature_names_file = os.path.join(preprocessed_dir, "extracted_feature_names.txt")
        with open(feature_names_file, 'w') as f:
            f.write(f"Feature Extraction Configuration:\n")
            f.write(f"Feature Set: {args.feature_set}\n")
            f.write(f"Feature Selection: {args.feature_selection}\n")
            f.write(f"Total Features: {len(expanded_feature_keys)}\n")
            f.write(f"Extraction Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Complete List of Extracted Features:\n")
            f.write("=" * 50 + "\n")
            for i, feature_name in enumerate(expanded_feature_keys, 1):
                f.write(f"{i:4d}. {feature_name}\n")
        
        logger.info(f"Feature names saved to: {feature_names_file}")
        
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
        
        # Apply robust data cleaning to prevent numerical issues
        logger.info("Cleaning feature matrix to prevent infinite/NaN values...")
        X_clean, variance_selector, cleaned_feature_names = clean_and_preprocess_features(
            X, expanded_feature_keys, logger
        )
        
        if X_clean is None:
            logger.error("Feature cleaning failed. Cannot proceed with training.")
            return
        
        # Update feature matrix and names
        X = X_clean
        if cleaned_feature_names is not None:
            expanded_feature_keys = cleaned_feature_names
        
        logger.info(f"Using {X.shape[1]} features after cleaning")


        #Save extracted features to file**
        feature_metadata = {
            'feature_set': args.feature_set,
            'feature_selection': args.feature_selection,
            'total_images_processed': success_count,
            'bcc_images': len(bcc_paths),
            'sk_images': len(sk_paths),
            'extraction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_feature_count': len(expanded_feature_keys),
            'cleaned_feature_count': X.shape[1],
            'dataset_info': {
                'bcc_dir': args.bcc_dir,
                'sk_dir': args.sk_dir,
                'max_images_per_class': args.max_images_per_class
            }
        }
        
        # Save features to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_filepath = f"output/features/extracted_features_{args.feature_set}_{timestamp}.pkl"
        
        if save_features_to_file(X, all_labels, expanded_feature_keys, feature_metadata, features_filepath):
            logger.info(f"Features successfully saved to {features_filepath}")
        else:
            logger.warning("Failed to save features to file")


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
        
        # Initialize selector to None for later checks
        selector = None
        
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
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                base_model =  XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss',
                    enable_categorical=False,
                    tree_method='hist'
                )
                selector = RFE(estimator=base_model, n_features_to_select=min(args.n_features, X_train.shape[1]), step=5)
            
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
            logger.info(f"Using all {len(expanded_feature_keys)} features (no feature selection)")
            selected_feature_names = expanded_feature_keys
            # Create a dummy selector that includes all features for consistency
            selector = SelectKBest(k='all')
            selector.fit(X_train, y_train)
        
        # Scale features with robust scaling (better for outliers)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
        try:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Applied robust scaling to features")
        except Exception as e:
            logger.warning(f"Robust scaling failed: {str(e)}. Trying standard scaling.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Applied standard scaling to features")
        
        # Final check after scaling
        if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
            logger.error("Infinite or NaN values detected after scaling!")
            return
        
        logger.info(f"Scaled feature ranges: train min={np.min(X_train):.2e}, train max={np.max(X_train):.2e}")
        
        # Prepare classifiers using CLASSIFIERS dictionary (same as fast training)
        classifiers = {}

        # Handle 'all' option
        if args.feature_classifiers.lower() == 'all':
            requested_classifiers = list(CLASSIFIERS.keys())
        else:
            # Map short names to full classifier names
            name_mapping = {
                'rf': 'RF',
                'svm_rbf': 'SVM (RBF)',
                'svm_linear': 'SVM (Linear)',
                'svm_sigmoid': 'SVM (Sigmoid)',
                'svm_poly': 'SVM (Poly)', 
                'knn': 'KNN',
                'mlp': 'MLP',
                'gb': 'Gradient Boosting',
                'logistic': 'Logistic Regression',
                'xgboost': 'XGBoost',
                'catboost': 'CatBoost',
                'lgbm': 'LightGBM',
                'extra_trees': 'Extra Trees',
                'calibrated_svm': 'Calibrated SVM'
            }
            
            requested_short_names = args.feature_classifiers.lower().split(',')
            requested_classifiers = []
            
            for short_name in requested_short_names:
                short_name = short_name.strip()
                if short_name in name_mapping:
                    full_name = name_mapping[short_name]
                    if full_name in CLASSIFIERS:
                        requested_classifiers.append(full_name)

        # Initialize classifiers using CLASSIFIERS dictionary
        for clf_name in requested_classifiers:
            if clf_name in CLASSIFIERS:
                clf_config = CLASSIFIERS[clf_name]
                clf = clf_config['class'](**clf_config['params'])
                classifiers[clf_name] = clf

        if not classifiers:
            logger.warning("No valid classifiers specified. Using Random Forest as default.")
            rf_config = CLASSIFIERS['RF']
            classifiers['RF'] = rf_config['class'](**rf_config['params'])

        logger.info(f"Training {len(classifiers)} classifiers: {', '.join(classifiers.keys())}")
        
        # Hyperparameter optimization if specified
        if args.optimize:
            logger.info("Performing hyperparameter optimization with enhanced grids")
            
            optimized_classifiers = {}
            is_very_small_dataset = len(all_labels) < 30
            
            for name, clf in classifiers.items():
                logger.info(f"Optimizing {name}")
                
                # Enhanced parameter grids optimized for 2257 samples, 511 features
                if is_very_small_dataset:
                    # Simplified grids for small datasets
                    if name == 'RF':
                        param_grid = {'n_estimators': [100], 'max_depth': [5, None], 'min_samples_leaf': [1, 2]}
                    elif 'SVM' in name:
                        param_grid = {'C': [1, 10], 'gamma': ['scale', 'auto']}
                    elif name == 'XGBoost':
                        param_grid = {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]}
                    elif name == 'CatBoost':
                        param_grid = {'iterations': [200], 'depth': [4], 'learning_rate': [0.1]}
                    # For LightGBM specifically
                    elif name == 'LightGBM':
                        param_grid = {
                            'num_leaves': [31, 63, 127],             # Include smaller values
                            'n_estimators': [300, 500, 800],
                            'learning_rate': [0.01, 0.03, 0.05],     # Include 0.03
                            'min_child_samples': [20, 30, 50],       # More aggressive
                            'min_split_gain': [0.0, 0.01, 0.05],     # NEW: test gain thresholds
                            'subsample': [0.7, 0.8, 0.9],            # Include 0.7
                            'colsample_bytree': [0.7, 0.8, 0.9],     # Include 0.7
                            'reg_alpha': [0.1, 0.5, 1.0],            # Stronger regularization
                            'reg_lambda': [1.0, 2.0, 3.0],           # Stronger regularization
                            'class_weight': ['balanced']
                            }
                    else:
                        param_grid = {}
                else:
                    # Optimized grids based on dataset: 2257 samples, 511 features, 83.6% class balance
                    if name == 'CatBoost':
                        param_grid = {
                            'iterations': [800, 1000, 1200],           # Increased for better convergence
                            'depth': [6, 8, 10],                       # Deeper trees for 511 features
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer granularity
                            'l2_leaf_reg': [3, 5, 7, 10],              # Stronger regularization options
                            'random_strength': [1.0, 1.5, 2.0],        # NEW: controls randomness
                            'bagging_temperature': [0.5, 1.0, 1.5],    # NEW: bagging aggressiveness
                            'border_count': [128, 254]                 # NEW: splits for numerical features
                        }

                    elif name == 'LightGBM':
                        param_grid = {
                            'num_leaves': [31, 63, 95, 127],           # Wider range for 511 features
                            'n_estimators': [500, 800, 1000],          # More trees for stability
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer granularity
                            'max_depth': [7, 8, 10, -1],               # Include unlimited depth option
                            'min_child_samples': [15, 20, 25],         # Adjusted for 2257 samples
                            'min_split_gain': [0.0, 0.01, 0.05],       # Pruning threshold
                            'subsample': [0.7, 0.8, 0.9],              # Row sampling
                            'colsample_bytree': [0.7, 0.8, 0.9],       # Feature sampling
                            'reg_alpha': [0.05, 0.1, 0.5],             # L1 regularization
                            'reg_lambda': [0.5, 1.0, 2.0],             # L2 regularization
                            'class_weight': ['balanced'],              # Handle 83.6/16.4 imbalance
                            'min_data_in_leaf': [15, 20, 25]          # NEW: minimum leaf samples
                        }

                    elif name == 'XGBoost':
                        param_grid = {
                            'n_estimators': [600, 800, 1000],          # More trees for 511 features
                            'max_depth': [6, 7, 8, 10],                # Deeper for complex features
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer control
                            'subsample': [0.75, 0.85, 0.95],           # Row sampling
                            'colsample_bytree': [0.75, 0.85, 0.95],    # Feature sampling
                            'gamma': [0.1, 0.2, 0.3],                  # Minimum loss reduction
                            'min_child_weight': [2, 3, 4],             # Minimum sum of instance weight
                            'reg_alpha': [0.05, 0.1, 0.2],             # L1 regularization
                            'reg_lambda': [0.5, 1.0, 1.5],             # L2 regularization
                            'scale_pos_weight': [1, 5, 10]             # NEW: handle class imbalance (83.6/16.4)
                        }

                    elif name == 'RF':
                        param_grid = {
                            'n_estimators': [300, 500, 700],           # More trees for stability
                            'max_depth': [15, 20, 25, None],           # Deeper trees for 511 features
                            'min_samples_split': [2, 3, 5],            # More granular
                            'min_samples_leaf': [1, 2, 3],             # Leaf size control
                            'max_features': ['sqrt', 'log2', 0.8],     # Include 80% of features option
                            'class_weight': ['balanced', 'balanced_subsample'],  # Handle imbalance
                            'max_samples': [0.7, 0.85, None],          # NEW: bootstrap sample size
                            'criterion': ['gini', 'entropy']           # NEW: split criterion
                        }

                    elif name == 'Extra Trees':
                        param_grid = {
                            'n_estimators': [300, 500, 700],
                            'max_depth': [15, 20, 25, None],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 3],
                            'max_features': ['sqrt', 'log2', 0.8],
                            'class_weight': ['balanced'],
                            'bootstrap': [False, True],                # NEW: use bootstrap
                            'criterion': ['gini', 'entropy']
                        }

                    elif 'SVM (RBF)' in name:
                        param_grid = {
                            'C': [10, 50, 100, 200],                   # Higher C for complex boundaries
                            'gamma': ['scale', 'auto', 0.01, 0.05, 0.1],  # More gamma options
                            'class_weight': ['balanced'],              # Essential for imbalance
                            'cache_size': [1000]                       # NEW: speed up computation
                        }

                    elif 'SVM (Linear)' in name:
                        param_grid = {
                            'C': [0.5, 1, 10, 50, 100],               # Wider range
                            'class_weight': ['balanced'],
                            'max_iter': [5000],                        # Ensure convergence
                            'dual': [False]                            # NEW: faster for n_samples > n_features
                        }

                    elif name == 'Logistic Regression':
                        param_grid = {
                            'estimator__C': [0.1, 1.0, 10.0, 50.0],   # Wider range
                            'estimator__max_iter': [3000, 5000],       # Ensure convergence
                            'estimator__solver': ['lbfgs', 'saga'],    # NEW: better solvers
                            'estimator__class_weight': ['balanced'],   # Handle imbalance
                            'method': ['sigmoid', 'isotonic'],
                            'cv': [3, 5]
                        }

                    elif name == 'MLP':
                        param_grid = {
                            'hidden_layer_sizes': [(100,), (100, 50), (150, 75), (200, 100, 50)],  # Larger networks
                            'alpha': [0.0001, 0.001, 0.01, 0.05],     # More regularization options
                            'learning_rate': ['constant', 'adaptive', 'invscaling'],  # NEW: more options
                            'learning_rate_init': [0.001, 0.01],       # NEW: initial learning rate
                            'max_iter': [500, 1000],                   # More iterations
                            'early_stopping': [True],                  # NEW: prevent overfitting
                            'validation_fraction': [0.1]               # NEW: validation set for early stopping
                        }

                    elif name == 'KNN':
                        param_grid = {
                            'n_neighbors': [3, 5, 7, 9, 11],          # More neighbors for 2257 samples
                            'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan', 'minkowski'],  # More metrics
                            'p': [1, 2],                               # NEW: Minkowski parameter
                            'algorithm': ['auto', 'ball_tree']         # NEW: algorithm selection
                        }

                    elif name == 'Gradient Boosting':
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'max_depth': [4, 5, 6, 7],                # Deeper for 511 features
                            'learning_rate': [0.05, 0.1, 0.15],
                            'subsample': [0.8, 0.9, 1.0],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'max_features': ['sqrt', 'log2']
                        }
                    
                    elif name == 'Deep DNN':
                        param_grid = {
                            'hidden_units': [
                                [512, 384, 256, 128],              # Moderate depth (4 residual blocks)
                                [512, 512, 384, 384, 256, 256, 128],  # Deep (7 residual blocks) - DEFAULT
                                [768, 768, 512, 512, 384, 256, 128],  # Wider and deeper
                                [384, 384, 256, 256, 128]          # Shallower (5 residual blocks)
                            ],
                            'dropout_rate': [0.2, 0.3, 0.4],       # Regularization strength
                            'learning_rate': [1e-4, 3e-4, 1e-3],   # Learning rate range
                            'batch_size': [64, 128, 256],          # Batch size options
                            'l2_reg': [1e-5, 1e-4, 1e-3],          # L2 regularization
                            'activation': ['swish'],               # Swish proven best for tabular
                            'use_attention': [True],               # Keep attention enabled
                            'mixup_alpha': [0.0, 0.1, 0.2, 0.3],   # Mixup augmentation strength
                            'focal_loss_alpha': [0.16, 0.20, 0.25], # Class imbalance handling
                            'focal_loss_gamma': [1.5, 2.0, 2.5],   # Focusing parameter
                            'epochs': [200],                       # Fixed epochs (early stopping handles)
                            'patience': [30],                      # Early stopping patience
                            'validation_split': [0.2],             # Fixed validation split
                            'verbose': [0],                        # Silent during grid search
                            'random_state': [42]
                        }
                    else:
                        param_grid = {}
                
                if not param_grid:
                    optimized_classifiers[name] = clf
                    continue
                
                # Adaptive CV strategy
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts[class_counts > 0])
                
                if min_class_count < 5:
                    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                elif min_class_count < 10:
                    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                else:
                    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Use RandomizedSearchCV for large grids
                n_combinations = np.prod([len(v) for v in param_grid.values()])
                
                if n_combinations > 50:
                    search = RandomizedSearchCV(
                        clf, param_grid, n_iter=50, cv=cv_strategy, 
                        scoring='f1', random_state=42, n_jobs=-1, verbose=1
                    )
                else:
                    search = GridSearchCV(
                        clf, param_grid, cv=cv_strategy, 
                        scoring='f1', n_jobs=-1, verbose=1
                    )
                
                search.fit(X_train, y_train)
                
                logger.info(f"Best params for {name}: {search.best_params_}")
                logger.info(f"Best CV F1: {search.best_score_:.4f}")
                
                optimized_classifiers[name] = search.best_estimator_
            
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



                 # 1. Precision-Recall Curve
                if y_pred_proba is not None:
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    
                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
                    ap_score = average_precision_score(y_test, y_pred_proba)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall_vals, precision_vals, label=f'{name} (AP = {ap_score:.3f})')
                    plt.xlabel('Recall (Sensitivity)')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    output_path = f'output/metrics/pr_curve_{name.replace(" ", "_")}.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Precision-Recall curve saved to {output_path}")
                
                # 2. Confusion Matrix Heatmap
                import seaborn as sns
                
                plt.figure(figsize=(6, 5))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['SK', 'BCC'], yticklabels=['SK', 'BCC'])
                plt.title(f'Confusion Matrix - {name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                output_path = f'output/metrics/confusion_matrix_{name.replace(" ", "_")}.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Confusion matrix saved to {output_path}")
                
                # 3. Prediction Probability Distribution
                if y_pred_proba is not None:
                    plt.figure(figsize=(10, 6))
                    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='SK (Class 0)', bins=30, color='blue')
                    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='BCC (Class 1)', bins=30, color='red')
                    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
                    plt.xlabel('Prediction Probability')
                    plt.ylabel('Frequency')
                    plt.title(f'Prediction Probability Distribution - {name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    output_path = f'output/metrics/prob_distribution_{name.replace(" ", "_")}.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Probability distribution saved to {output_path}")

                
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

            # 1. Model Performance Comparison Bar Chart
            if len(results) > 1:
                metrics = ['AC', 'SN', 'SP', 'PR', 'F1']
                model_names = list(results.keys())
                
                fig, ax = plt.subplots(figsize=(12, 8))
                x = np.arange(len(metrics))
                width = 0.8 / len(model_names)
                
                for i, model in enumerate(model_names):
                    values = [results[model].get(metric, 0) for metric in metrics]
                    ax.bar(x + i*width, values, width, label=model, alpha=0.8)
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Score (%)')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(x + width * (len(model_names) - 1) / 2)
                ax.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.ylim(0, 105)
                
                output_path = 'output/metrics/model_comparison.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Model comparison chart saved to {output_path}")
            
            # 2. Combined ROC Curves for All Models
            if len(results) > 1:
                plt.figure(figsize=(10, 8))
                
                for model_name, result in results.items():
                    if result.get('y_pred_proba') is not None:
                        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                        auc_score = result.get('test_roc_auc', 0)
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
                
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves Comparison - All Models')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                
                output_path = 'output/metrics/roc_curves_combined.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Combined ROC curves saved to {output_path}")
            
            # 3. F1 Score Radar Chart (if multiple models)
            if len(results) >= 3:
                try:
                    # Create radar chart for top 3 models by F1 score
                    sorted_models = sorted(results.items(), key=lambda x: x[1].get('F1', 0), reverse=True)[:3]
                    
                    categories = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]  # Complete the circle
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    colors = ['red', 'blue', 'green']
                    for i, (model_name, metrics) in enumerate(sorted_models):
                        values = [metrics.get('AC', 0)/100, metrics.get('SN', 0)/100, 
                                metrics.get('SP', 0)/100, metrics.get('PR', 0)/100, 
                                metrics.get('F1', 0)/100]
                        values += values[:1]  # Complete the circle
                        
                        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
                        ax.fill(angles, values, alpha=0.25, color=colors[i])
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_ylim(0, 1)
                    ax.set_title('Top 3 Models Performance Radar Chart', size=16, pad=20)
                    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                    
                    output_path = 'output/metrics/radar_chart_top3.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Radar chart saved to {output_path}")
                except Exception as e:
                    logger.warning(f"Could not create radar chart: {str(e)}")




        
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


def save_features_to_file(X, y, feature_names, metadata, filepath):
    """
    Save extracted features, labels, and metadata to a file for later use.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        metadata: Dictionary with extraction metadata
        filepath: Path to save the features
    """
    try:
        import pickle
        import numpy as np
        
        # Create the data package
        feature_data = {
            'features': X.astype(np.float32),  # Save space with float32
            'labels': y.astype(np.int32),
            'feature_names': feature_names,
            'metadata': metadata,
            'version': '1.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save using pickle for Python objects
        with open(filepath, 'wb') as f:
            pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save as CSV for external analysis
        csv_path = filepath.replace('.pkl', '_features.csv')
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df['label'] = y
        feature_df.to_csv(csv_path, index=False)
        
        # Save metadata as JSON
        json_path = filepath.replace('.pkl', '_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Features saved to: {filepath}")
        print(f"CSV export saved to: {csv_path}")
        print(f"Metadata saved to: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"Error saving features: {str(e)}")
        return False

def load_features_from_file(filepath):
    """
    Load previously saved features from file.
    
    Args:
        filepath: Path to the saved features file
        
    Returns:
        X, y, feature_names, metadata
    """
    try:
        import pickle
        
        with open(filepath, 'rb') as f:
            feature_data = pickle.load(f)
        
        return (
            feature_data['features'],
            feature_data['labels'],
            feature_data['feature_names'],
            feature_data['metadata']
        )
        
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        return None, None, None, None
    

def train_models_from_features(features_filepath, args, logger, custom_params=None):
    """
    Train models using previously extracted features.
    
    Args:
        features_filepath: Path to saved features file
        args: Command line arguments (for model configuration)
        logger: Logger instance
        custom_params: Optional dictionary with custom model parameters
    """
    try:
        # Load features
        logger.info(f"Loading features from {features_filepath}")
        X, y, feature_names, metadata = load_features_from_file(features_filepath)
        
        if X is None:
            logger.error("Failed to load features")
            return
        
        logger.info(f"Loaded features: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Feature extraction metadata: {metadata.get('extraction_timestamp', 'Unknown')}")
        
        # **ADD FEATURE CLEANING HERE - THIS WAS MISSING!**
        logger.info("Cleaning loaded features to prevent infinite/NaN values...")
        X_clean, variance_selector, cleaned_feature_names = clean_and_preprocess_features(
            X, feature_names, logger
        )
        
        if X_clean is None:
            logger.error("Feature cleaning failed. Cannot proceed with training.")
            return
        
        # Update feature matrix and names
        X = X_clean
        if cleaned_feature_names is not None:
            feature_names = cleaned_feature_names
        
        logger.info(f"Using {X.shape[1]} features after cleaning")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Ensure integer labels
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Apply feature selection if specified
        selector = None
        if args.feature_selection != 'none':
            logger.info(f"Applying feature selection: {args.feature_selection}")
            
            if args.feature_selection == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'chi2':
                X_train_min = X_train.min(axis=0)
                X_train = X_train - X_train_min
                X_test = X_test - X_train_min
                selector = SelectKBest(chi2, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'f_test':
                selector = SelectKBest(f_classif, k=min(args.n_features, X_train.shape[1]))
            elif args.feature_selection == 'rfe':
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                X_train = np.clip(X_train, -1e8, 1e8)
                X_test = np.clip(X_test, -1e8, 1e8)
                base_model =  XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss',
                    enable_categorical=False,
                    tree_method='hist'
                )
                selector = RFE(estimator=base_model, n_features_to_select=min(args.n_features, X_train.shape[1]), step=5)
        
            
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            logger.info(f"Selected {len(selected_feature_names)} features")
            # **NEW: Save feature scores and rankings**
            if hasattr(selector, 'scores_'):
                # Get scores for selected features
                feature_scores = [(feature_names[i], selector.scores_[i]) 
                                 for i in selected_indices]
                
                # Sort by score (descending)
                feature_scores_sorted = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                
                # Save to file
                scores_filename = f"output/features/selected_features_{args.feature_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(scores_filename, 'w') as f:
                    f.write(f"Feature Selection Method: {args.feature_selection}\n")
                    f.write(f"Number of Selected Features: {len(selected_feature_names)}\n")
                    f.write(f"Total Original Features: {len(feature_names)}\n")
                    f.write(f"Selection Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"{'Rank':<6} {'Feature Name':<50} {'Score':<15}\n")
                    f.write("=" * 80 + "\n")
                    
                    for rank, (feature_name, score) in enumerate(feature_scores_sorted, 1):
                        f.write(f"{rank:<6} {feature_name:<50} {score:<15.6f}\n")
                
                logger.info(f"Feature scores saved to: {scores_filename}")
                
                # Log top 20 features
                logger.info(f"Top 20 features by {args.feature_selection} score:")
                for i, (feature, score) in enumerate(feature_scores_sorted[:20], 1):
                    logger.info(f"{i:2d}. {feature}: {score:.4f}")
            else:
                logger.info(f"First 20 selected features: {selected_feature_names[:20]}")

 
        else:
            selected_feature_names = feature_names
        
        # **IMPROVED SCALING WITH BETTER ERROR HANDLING**
        logger.info("Applying feature scaling...")
        
        # Check for remaining issues before scaling
        inf_count = np.isinf(X_train).sum()
        nan_count = np.isnan(X_train).sum()
        
        if inf_count > 0 or nan_count > 0:
            logger.warning(f"Found {inf_count} inf and {nan_count} NaN values before scaling. Applying additional cleaning...")
            
            # Replace any remaining inf values
            X_train = np.where(np.isposinf(X_train), np.finfo(np.float64).max / 1e10, X_train)
            X_train = np.where(np.isneginf(X_train), np.finfo(np.float64).min / 1e10, X_train)
            X_test = np.where(np.isposinf(X_test), np.finfo(np.float64).max / 1e10, X_test)
            X_test = np.where(np.isneginf(X_test), np.finfo(np.float64).min / 1e10, X_test)
            
            # Replace any remaining NaN values with median
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
        
        # Try robust scaling first, then standard scaling
        from sklearn.preprocessing import RobustScaler, StandardScaler
        
        try:
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Applied robust scaling to features")
        except Exception as e:
            logger.warning(f"Robust scaling failed: {str(e)}. Trying standard scaling.")
            try:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                logger.info("Applied standard scaling to features")
            except Exception as e2:
                logger.warning(f"Standard scaling also failed: {str(e2)}. Using min-max scaling.")
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                logger.info("Applied min-max scaling to features")
        
        # Final validation
        if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
            logger.error("Still have infinite/NaN values after cleaning and scaling!")
            logger.info("Applying emergency clipping...")
            X_train = np.clip(X_train, -1e10, 1e10)
            X_test = np.clip(X_test, -1e10, 1e10)
            
            # Replace any NaN with zeros as last resort
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
            logger.info("Applied emergency value cleaning")
        
        logger.info(f"Final scaled feature ranges: train min={np.min(X_train):.2e}, train max={np.max(X_train):.2e}")
        
        # **USE EXISTING CLASSIFIERS DICTIONARY INSTEAD OF RECREATING**
        classifiers = {}
        
        # Handle 'all' option
        if args.feature_classifiers.lower() == 'all':
            requested_classifiers = list(CLASSIFIERS.keys())
        else:
            # Map short names to full classifier names
            name_mapping = {
                'rf': 'RF',
                'svm_rbf': 'SVM (RBF)',
                'svm_linear': 'SVM (Linear)',
                'svm_sigmoid': 'SVM (Sigmoid)',
                'svm_poly': 'SVM (Poly)',
                'knn': 'KNN',
                'mlp': 'MLP',
                'gb': 'Gradient Boosting',
                'logistic': 'Logistic Regression',
                'xgboost': 'XGBoost'
            }
            
            requested_short_names = args.feature_classifiers.lower().split(',')
            requested_classifiers = []
            
            for short_name in requested_short_names:
                if short_name in name_mapping:
                    full_name = name_mapping[short_name]
                    if full_name in CLASSIFIERS:
                        requested_classifiers.append(full_name)
                    else:
                        logger.warning(f"Classifier '{full_name}' not found in CLASSIFIERS dictionary")
                else:
                    logger.warning(f"Unknown classifier short name: '{short_name}'")
        
        # Initialize classifiers using CLASSIFIERS dictionary
        for clf_name in requested_classifiers:
            if clf_name in CLASSIFIERS:
                clf_config = CLASSIFIERS[clf_name]
                
                # Start with default parameters from CLASSIFIERS
                params = clf_config['params'].copy()
                
                # Override with custom parameters if provided
                if custom_params:
                    # Map full names to short names for custom params lookup
                    reverse_mapping = {
                        'RF': 'rf',
                        'SVM (RBF)': 'svm_rbf',
                        'SVM (Linear)': 'svm_linear',
                        'SVM (Sigmoid)': 'svm_sigmoid',
                        'SVM (Poly)': 'svm_poly',
                        'KNN': 'knn',
                        'MLP': 'mlp',
                        'Gradient Boosting': 'gb',
                        'Logistic Regression': 'logistic',
                        'XGBoost': 'xgboost'
                    }
                    
                    short_name = reverse_mapping.get(clf_name)
                    if short_name and short_name in custom_params:
                        logger.info(f"Applying custom parameters for {clf_name}")
                        params.update(custom_params[short_name])
                
                # Create classifier instance
                try:
                    clf = clf_config['class'](**params)
                    classifiers[clf_name] = clf
                    logger.info(f"Initialized {clf_name} with parameters: {params}")
                except Exception as e:
                    logger.error(f"Error initializing {clf_name}: {str(e)}")
            else:
                logger.warning(f"Classifier '{clf_name}' not found in CLASSIFIERS dictionary")
        
        if not classifiers:
            logger.warning("No valid classifiers specified. Using Random Forest as default.")
            rf_config = CLASSIFIERS['RF']
            classifiers['RF'] = rf_config['class'](**rf_config['params'])
        
        logger.info(f"Training {len(classifiers)} classifiers: {', '.join(classifiers.keys())}")
        

        # HYPERPARAMETER OPTIMIZATION 
        if args.optimize:
            logger.info("Performing hyperparameter optimization for loaded features")
            
            optimized_classifiers = {}
            is_very_small_dataset = len(y_train) < 30
            
            for name, clf in classifiers.items():
                logger.info(f"Optimizing {name}")
                
                # Use the SAME parameter grids as in train_features()
                if is_very_small_dataset:
                    # Simplified grids for small datasets
                    if name == 'RF':
                        param_grid = {'n_estimators': [100], 'max_depth': [5, None], 'min_samples_leaf': [1, 2]}
                    elif 'SVM' in name:
                        param_grid = {'C': [1, 10], 'gamma': ['scale', 'auto']}
                    elif name == 'XGBoost':
                        param_grid = {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]}
                    elif name == 'CatBoost':
                        param_grid = {'iterations': [200], 'depth': [4], 'learning_rate': [0.1]}
                    elif name == 'LightGBM':
                        param_grid = {'num_leaves': [31], 'n_estimators': [100], 'learning_rate': [0.1]}
                    else:
                        param_grid = {}
                else:
                    # Optimized grids based on dataset: 2257 samples, 511 features, 83.6% class balance
                    if name == 'CatBoost':
                        param_grid = {
                            'iterations': [800, 1000, 1200],           # Increased for better convergence
                            'depth': [6, 8, 10],                       # Deeper trees for 511 features
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer granularity
                            'l2_leaf_reg': [3, 5, 7, 10],              # Stronger regularization options
                            'random_strength': [1.0, 1.5, 2.0],        # NEW: controls randomness
                            'bagging_temperature': [0.5, 1.0, 1.5],    # NEW: bagging aggressiveness
                            'border_count': [128, 254]                 # NEW: splits for numerical features
                        }

                    elif name == 'LightGBM':
                        param_grid = {
                            'num_leaves': [31, 63, 95, 127],           # Wider range for 511 features
                            'n_estimators': [500, 800, 1000],          # More trees for stability
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer granularity
                            'max_depth': [7, 8, 10, -1],               # Include unlimited depth option
                            'min_child_samples': [15, 20, 25],         # Adjusted for 2257 samples
                            'min_split_gain': [0.0, 0.01, 0.05],       # Pruning threshold
                            'subsample': [0.7, 0.8, 0.9],              # Row sampling
                            'colsample_bytree': [0.7, 0.8, 0.9],       # Feature sampling
                            'reg_alpha': [0.05, 0.1, 0.5],             # L1 regularization
                            'reg_lambda': [0.5, 1.0, 2.0],             # L2 regularization
                            'class_weight': ['balanced'],              # Handle 83.6/16.4 imbalance
                            'min_data_in_leaf': [15, 20, 25]          # NEW: minimum leaf samples
                        }

                    elif name == 'XGBoost':
                        param_grid = {
                            'n_estimators': [600, 800, 1000],          # More trees for 511 features
                            'max_depth': [6, 7, 8, 10],                # Deeper for complex features
                            'learning_rate': [0.03, 0.05, 0.07],       # Finer control
                            'subsample': [0.75, 0.85, 0.95],           # Row sampling
                            'colsample_bytree': [0.75, 0.85, 0.95],    # Feature sampling
                            'gamma': [0.1, 0.2, 0.3],                  # Minimum loss reduction
                            'min_child_weight': [2, 3, 4],             # Minimum sum of instance weight
                            'reg_alpha': [0.05, 0.1, 0.2],             # L1 regularization
                            'reg_lambda': [0.5, 1.0, 1.5],             # L2 regularization
                            'scale_pos_weight': [1, 5, 10]             # NEW: handle class imbalance (83.6/16.4)
                        }

                    elif name == 'RF':
                        param_grid = {
                            'n_estimators': [300, 500, 700],           # More trees for stability
                            'max_depth': [15, 20, 25, None],           # Deeper trees for 511 features
                            'min_samples_split': [2, 3, 5],            # More granular
                            'min_samples_leaf': [1, 2, 3],             # Leaf size control
                            'max_features': ['sqrt', 'log2', 0.8],     # Include 80% of features option
                            'class_weight': ['balanced', 'balanced_subsample'],  # Handle imbalance
                            'max_samples': [0.7, 0.85, None],          # NEW: bootstrap sample size
                            'criterion': ['gini', 'entropy']           # NEW: split criterion
                        }

                    elif name == 'Extra Trees':
                        param_grid = {
                            'n_estimators': [300, 500, 700],
                            'max_depth': [15, 20, 25, None],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 3],
                            'max_features': ['sqrt', 'log2', 0.8],
                            'class_weight': ['balanced'],
                            'bootstrap': [False, True],                # NEW: use bootstrap
                            'criterion': ['gini', 'entropy']
                        }

                    elif 'SVM (RBF)' in name:
                        param_grid = {
                            'C': [10, 50, 100, 200],                   # Higher C for complex boundaries
                            'gamma': ['scale', 'auto', 0.01, 0.05, 0.1],  # More gamma options
                            'class_weight': ['balanced'],              # Essential for imbalance
                            'cache_size': [1000]                       # NEW: speed up computation
                        }

                    elif 'SVM (Linear)' in name:
                        param_grid = {
                            'C': [0.5, 1, 10, 50, 100],               # Wider range
                            'class_weight': ['balanced'],
                            'max_iter': [5000],                        # Ensure convergence
                            'dual': [False]                            # NEW: faster for n_samples > n_features
                        }

                    elif name == 'Logistic Regression':
                        param_grid = {
                            'estimator__C': [0.1, 1.0, 10.0, 50.0],   # Wider range
                            'estimator__max_iter': [3000, 5000],       # Ensure convergence
                            'estimator__solver': ['lbfgs', 'saga'],    # NEW: better solvers
                            'estimator__class_weight': ['balanced'],   # Handle imbalance
                            'method': ['sigmoid', 'isotonic'],
                            'cv': [3, 5]
                        }

                    elif name == 'MLP':
                        param_grid = {
                            'hidden_layer_sizes': [(100,), (100, 50), (150, 75), (200, 100, 50)],  # Larger networks
                            'alpha': [0.0001, 0.001, 0.01, 0.05],     # More regularization options
                            'learning_rate': ['constant', 'adaptive', 'invscaling'],  # NEW: more options
                            'learning_rate_init': [0.001, 0.01],       # NEW: initial learning rate
                            'max_iter': [500, 1000],                   # More iterations
                            'early_stopping': [True],                  # NEW: prevent overfitting
                            'validation_fraction': [0.1]               # NEW: validation set for early stopping
                        }

                    elif name == 'KNN':
                        param_grid = {
                            'n_neighbors': [3, 5, 7, 9, 11],          # More neighbors for 2257 samples
                            'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan', 'minkowski'],  # More metrics
                            'p': [1, 2],                               # NEW: Minkowski parameter
                            'algorithm': ['auto', 'ball_tree']         # NEW: algorithm selection
                        }

                    elif name == 'Gradient Boosting':
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'max_depth': [4, 5, 6, 7],                # Deeper for 511 features
                            'learning_rate': [0.05, 0.1, 0.15],
                            'subsample': [0.8, 0.9, 1.0],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'max_features': ['sqrt', 'log2']
                        }
                    
                    elif name == 'Deep DNN':
                        param_grid = {
                            'hidden_units': [
                                [512, 384, 256, 128],              # Moderate depth (4 residual blocks)
                                [512, 512, 384, 384, 256, 256, 128],  # Deep (7 residual blocks) - DEFAULT
                                [768, 768, 512, 512, 384, 256, 128],  # Wider and deeper
                                [384, 384, 256, 256, 128]          # Shallower (5 residual blocks)
                            ],
                            'dropout_rate': [0.2, 0.3, 0.4],       # Regularization strength
                            'learning_rate': [1e-4, 3e-4, 1e-3],   # Learning rate range
                            'batch_size': [64, 128, 256],          # Batch size options
                            'l2_reg': [1e-5, 1e-4, 1e-3],          # L2 regularization
                            'activation': ['swish'],               # Swish proven best for tabular
                            'use_attention': [True],               # Keep attention enabled
                            'mixup_alpha': [0.0, 0.1, 0.2, 0.3],   # Mixup augmentation strength
                            'focal_loss_alpha': [0.16, 0.20, 0.25], # Class imbalance handling
                            'focal_loss_gamma': [1.5, 2.0, 2.5],   # Focusing parameter
                            'epochs': [200],                       # Fixed epochs (early stopping handles)
                            'patience': [30],                      # Early stopping patience
                            'validation_split': [0.2],             # Fixed validation split
                            'verbose': [0],                        # Silent during grid search
                            'random_state': [42]
                        }
                    else:
                        param_grid = {}
                
                if not param_grid:
                    optimized_classifiers[name] = clf
                    continue
                
                # Adaptive CV strategy
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts[class_counts > 0])
                
                if min_class_count < 5:
                    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                elif min_class_count < 10:
                    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                else:
                    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Use RandomizedSearchCV for large grids
                n_combinations = np.prod([len(v) for v in param_grid.values()])
                
                if n_combinations > 50:
                    search = RandomizedSearchCV(
                        clf, param_grid, n_iter=50, cv=cv_strategy, 
                        scoring='f1', random_state=42, n_jobs=-1, verbose=1
                    )
                else:
                    search = GridSearchCV(
                        clf, param_grid, cv=cv_strategy, 
                        scoring='f1', n_jobs=-1, verbose=1
                    )
                
                search.fit(X_train, y_train)
                
                logger.info(f"Best params for {name}: {search.best_params_}")
                logger.info(f"Best CV F1: {search.best_score_:.4f}")
                
                optimized_classifiers[name] = search.best_estimator_
            
            classifiers = optimized_classifiers
        else:
            logger.info("Skipping hyperparameter optimization (--optimize not specified)")


        # Train and evaluate models (rest remains the same)
        results = {}
        
        for name, clf in classifiers.items():
            logger.info(f"Training and evaluating {name}")
            
            try:
                # Determine CV strategy
                class_counts = np.bincount(y_train)
                min_class_count = min(class_counts[class_counts > 0])
                
                if min_class_count < 5:
                    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                elif min_class_count < 10:
                    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                else:
                    cv_strategy = 5
                
                # Cross-validation
                cv_scores = cross_validate(
                    clf, X_train, y_train, 
                    cv=cv_strategy,
                    scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
                )
                
                # Train final model
                clf.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0.0)
                recall = recall_score(y_test, y_pred, zero_division=0.0)
                f1 = f1_score(y_test, y_pred, zero_division=0.0)
                specificity = specificity_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Log results
                logger.info(f"{name} - Test Results:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                logger.info(f"  Specificity: {specificity:.4f}")
                if roc_auc is not None:
                    logger.info(f"  ROC AUC: {roc_auc:.4f}")
                
                # Store results
                results[name] = {
                    'AC': accuracy * 100,
                    'PR': precision * 100,
                    'SN': recall * 100,
                    'F1': f1 * 100,
                    'SP': specificity * 100,
                    'AUC': roc_auc * 100 if roc_auc is not None else None,
                    'NUM_FEATURES': len(selected_feature_names),
                    'NUM_SELECTED': len(selected_feature_names)
                }
    # === ADD ALL VISUALIZATION CODE FROM train_features() HERE ===
                
                # 1. ROC Curve
                if y_pred_proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {name}')
                    plt.legend(loc='lower right')
                    
                    output_path = f'output/metrics/roc_curve_{name.replace(" ", "_")}_fast.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"ROC curve saved to {output_path}")
                
                # 2. Precision-Recall Curve
                if y_pred_proba is not None:
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    
                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
                    ap_score = average_precision_score(y_test, y_pred_proba)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall_vals, precision_vals, label=f'{name} (AP = {ap_score:.3f})')
                    plt.xlabel('Recall (Sensitivity)')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    output_path = f'output/metrics/pr_curve_{name.replace(" ", "_")}_fast.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Precision-Recall curve saved to {output_path}")
                
                # 3. Confusion Matrix Heatmap
                import seaborn as sns
                
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['SK', 'BCC'], yticklabels=['SK', 'BCC'])
                plt.title(f'Confusion Matrix - {name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                output_path = f'output/metrics/confusion_matrix_{name.replace(" ", "_")}_fast.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Confusion matrix saved to {output_path}")
                
                # 4. Prediction Probability Distribution
                if y_pred_proba is not None:
                    plt.figure(figsize=(10, 6))
                    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='SK (Class 0)', bins=30, color='blue')
                    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='BCC (Class 1)', bins=30, color='red')
                    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
                    plt.xlabel('Prediction Probability')
                    plt.ylabel('Frequency')
                    plt.title(f'Prediction Probability Distribution - {name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    output_path = f'output/metrics/prob_distribution_{name.replace(" ", "_")}_fast.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Probability distribution saved to {output_path}")
                
                # 5. Feature Importance
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    n_top_features = min(30, len(selected_feature_names))
                    top_indices = indices[:n_top_features]
                    
                    plt.figure(figsize=(10, 8))
                    plt.title(f'Top {n_top_features} Feature Importances - {name}')
                    plt.barh(range(n_top_features), importances[top_indices], align='center')
                    plt.yticks(range(n_top_features), [selected_feature_names[i] for i in top_indices])
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    
                    output_path = f'output/metrics/feature_importance_{name.replace(" ", "_")}_fast.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Feature importance plot saved to {output_path}")

                    # Save model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                    model_dir = f'model/feature_based_fast/{safe_name}_{timestamp}'
                    os.makedirs(model_dir, exist_ok=True)
                
                dump(clf, f'{model_dir}/model.joblib')
                dump(scaler, f'{model_dir}/scaler.joblib')
                if selector:
                    dump(selector, f'{model_dir}/selector.joblib')
                
                # Save classifier configuration for reproducibility
                config_path = f'{model_dir}/classifier_config.json'
                clf_config = CLASSIFIERS[clf_name]  # Get config from CLASSIFIERS dict
                with open(config_path, 'w') as f:
                    json.dump({
                        'name': name,
                        'class': clf_config['class'].__name__,
                        'parameters': clf.get_params() if hasattr(clf, 'get_params') else str(clf)
                    }, f, indent=2)
                
                logger.info(f"Model saved to {model_dir}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        # === ADD COMBINED VISUALIZATIONS AFTER ALL MODELS TRAINED ===
        
        # 1. Combined ROC Curves
        if len(results) > 1:
            plt.figure(figsize=(10, 8))
            
            for model_name, result in results.items():
                if result.get('y_pred_proba') is not None:
                    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                    auc_score = result.get('AUC', 0) / 100
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison - All Models (Fast Training)')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            output_path = 'output/metrics/roc_curves_combined_fast.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Combined ROC curves saved to {output_path}")
        
        # 2. Model Performance Comparison Bar Chart
        if len(results) > 1:
            metrics = ['AC', 'SN', 'SP', 'PR', 'F1']
            model_names = list(results.keys())
            
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(metrics))
            width = 0.8 / len(model_names)
            
            for i, model in enumerate(model_names):
                values = [results[model].get(metric, 0) for metric in metrics]
                ax.bar(x + i*width, values, width, label=model, alpha=0.8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score (%)')
            ax.set_title('Model Performance Comparison (Fast Training)')
            ax.set_xticks(x + width * (len(model_names) - 1) / 2)
            ax.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.ylim(0, 105)
            
            output_path = 'output/metrics/model_comparison_fast.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Model comparison chart saved to {output_path}")


        # Generate summary
        if results:
            generate_summary_table(results, logger, table_num=6, 
                                 title="Fast Model Training Results")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in train_models_from_features: {str(e)}")
        traceback.print_exc()
        return None


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
    os.makedirs('output/features', exist_ok=True)
    os.makedirs('model/feature_based_fast', exist_ok=True)

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

        # Load custom parameters if provided
    custom_params = None
    if args.custom_params:
        try:
            with open(args.custom_params, 'r') as f:
                custom_params = json.load(f)
            logger.info(f"Loaded custom parameters from {args.custom_params}")
        except Exception as e:
            logger.warning(f"Failed to load custom parameters: {str(e)}")
    
    if args.train_from_features:
        # Train models from saved features
        if not args.features_file:
            logger.error("--features-file required when using --train-from-features")
            return
        
        logger.info(f"Training models from saved features: {args.features_file}")
        train_models_from_features(args.features_file, args, logger, custom_params)
        
    else:
        # Normal training (extract features and train models)
        train_features(args, logger)        

if __name__ == "__main__":
    main()
