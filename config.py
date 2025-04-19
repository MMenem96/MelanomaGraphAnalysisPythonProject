"""
Configuration settings for the melanoma detection system.
"""

import os
import logging

# Default parameters
DEFAULT_N_SEGMENTS = 20
DEFAULT_COMPACTNESS = 10
DEFAULT_CONNECTIVITY_THRESHOLD = 0.5
DEFAULT_CLASSIFIER = 'svm'
DEFAULT_MODEL_PATH = 'models/melanoma_model.joblib'
DEFAULT_OUTPUT_DIR = 'output'
DEFAULT_UPLOAD_FOLDER = 'uploads'

# Image parameters
MAX_IMAGE_SIZE = (750, 750)  # Standard size as per paper
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Feature extraction parameters
COLOR_FEATURE_COUNT = 150
TEXTURE_FEATURE_COUNT = 144
SHAPE_FEATURE_COUNT = 100
GRAPH_LOCAL_FEATURE_COUNT = 120  # 6 metrics x 20 nodes
GRAPH_GLOBAL_FEATURE_COUNT = 5
SPECTRAL_FEATURE_COUNT = 24

TOTAL_FEATURE_COUNT = (
    COLOR_FEATURE_COUNT + 
    TEXTURE_FEATURE_COUNT + 
    SHAPE_FEATURE_COUNT + 
    GRAPH_LOCAL_FEATURE_COUNT + 
    GRAPH_GLOBAL_FEATURE_COUNT + 
    SPECTRAL_FEATURE_COUNT
)

# Model hyperparameters
SVM_PARAMS = {
    'C': 10.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'class_weight': 'balanced'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced'
}

# Logging configuration
def setup_logging(log_level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('melanoma_detection.log', mode='a')
        ]
    )
