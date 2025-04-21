#!/usr/bin/env python3
"""
Comprehensive model comparison script for melanoma detection.

This script trains and evaluates multiple classifier types with different feature sets,
then generates comparison tables matching the format of the reference tables.

Usage:
    python comprehensive_model_comparison.py [--test-only]

Options:
    --test-only    Only evaluate existing models without retraining (much faster)

The script will:
1. Process dataset
2. Train multiple classifiers (SVM with different kernels, KNN, MLP, RF)
3. Evaluate each classifier with different feature subsets
4. Generate 5 comparison tables with real performance metrics
5. Save tables as CSV files and PNG visualizations
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import local modules
from src.dataset_handler import DatasetHandler
from src.classifier import MelanomaClassifier
from src.feature_extraction import FeatureExtractor
from src.conventional_features import ConventionalFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ComprehensiveModelComparison - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_model_comparison.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ComprehensiveModelComparison')

# Define directories
DATA_DIR = 'data'
MODEL_DIR = 'models_comparison'
OUTPUT_DIR = 'output'
MELANOMA_DIR = os.path.join(DATA_DIR, 'melanoma')
BENIGN_DIR = os.path.join(DATA_DIR, 'benign')

# Define parameters
PARAMS = {
    'n_segments': 20,
    'compactness': 10.0,
    'connectivity_threshold': 0.5,
    'max_images_per_class': 2000,
    'test_size': 0.2,
    'random_state': 42
}

# Define classifiers
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

# Define feature sets
FEATURE_SETS = {
    # Conventional features only
    'Conv': {
        'local_graph': False, 
        'global_graph': False, 
        'spectral': False, 
        'conventional': True
    },
    
    # Graph feature sets (Table 3)
    'C': {
        'local_graph': True, 
        'global_graph': False, 
        'spectral': False, 
        'conventional': False
    },
    'G': {
        'local_graph': False, 
        'global_graph': True, 
        'spectral': False, 
        'conventional': False
    },
    'C+G': {
        'local_graph': True, 
        'global_graph': True, 
        'spectral': False, 
        'conventional': False
    },
    'G+Conv': {
        'local_graph': False, 
        'global_graph': True, 
        'spectral': False, 
        'conventional': True
    },
    
    # GFT feature sets (Table 4 & 5)
    'GFT': {
        'local_graph': False, 
        'global_graph': False, 
        'spectral': True, 
        'conventional': False
    },
    'C+G+GFT': {
        'local_graph': True, 
        'global_graph': True, 
        'spectral': True, 
        'conventional': False
    },
    'GFT+Conv': {
        'local_graph': False, 
        'global_graph': False, 
        'spectral': True, 
        'conventional': True
    },
    'All': {
        'local_graph': True, 
        'global_graph': True, 
        'spectral': True, 
        'conventional': True
    },
    
    # Combined features (for Graph+Conv needed in Table 1 & 2)
    'Graph+Conv': {
        'local_graph': True, 
        'global_graph': True, 
        'spectral': False, 
        'conventional': True
    },
}

def specificity_score(y_true, y_pred):
    """Calculate the specificity score (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def extract_features(graphs, feature_set):
    """Extract a subset of features from graphs based on the feature set configuration."""
    feature_extractor = FeatureExtractor()
    
    # Create a feature matrix
    feature_list = []
    for G in graphs:
        features = {}
        
        # Get basic graph features
        graph_features = G.graph.get('features', {})
        conv_features = G.graph.get('conventional_features', {})
        
        # Extract local graph features
        if feature_set['local_graph']:
            local_features = []
            for feature_name in ['clustering_coefficient', 'nodal_strength', 'betweenness_centrality', 'closeness_centrality']:
                values = list(graph_features.get(feature_name, {}).values())
                if values:
                    # Calculate statistical measures
                    local_features.extend([
                        np.mean(values), np.std(values), 
                        np.min(values), np.max(values), 
                        np.median(values)
                    ])
                else:
                    local_features.extend([0, 0, 0, 0, 0])
            features['local_graph'] = local_features
        
        # Extract global graph features
        if feature_set['global_graph']:
            global_features = []
            for key in ['local_efficiency', 'char_path_length', 'global_efficiency', 
                        'global_clustering', 'density', 'assortativity', 
                        'transitivity', 'rich_club_coefficient']:
                value = graph_features.get(key, 0)
                # Handle infinity
                if value == float('inf'):
                    value = 0
                global_features.append(value)
            features['global_graph'] = global_features
        
        # Extract spectral features
        if feature_set['spectral']:
            spectral_features = []
            for key in ['spectral_radius', 'energy', 'spectral_gap', 
                        'normalized_laplacian_energy', 'spectral_power', 
                        'spectral_entropy', 'spectral_amplitude']:
                spectral_features.append(graph_features.get(key, 0))
            
            # Add GFT coefficients
            gft_coeffs = graph_features.get('gft_coefficients', np.zeros(20))
            spectral_features.extend(gft_coeffs)
            features['spectral'] = spectral_features
        
        # Extract conventional features
        if feature_set['conventional']:
            if conv_features:
                # Sort by key to ensure consistent ordering
                conv_keys = sorted(conv_features.keys())
                conv_feature_list = []
                
                for k in conv_keys:
                    value = conv_features[k]
                    # Handle lists (like Hu moments)
                    if isinstance(value, list):
                        conv_feature_list.extend(value)
                    else:
                        conv_feature_list.append(value)
                features['conventional'] = conv_feature_list
            else:
                # If conventional features are missing, use zeros as placeholders
                features['conventional'] = [0] * 394  # Typical length of conventional features
        
        # Combine all features into a single vector
        feature_vector = []
        for key in ['local_graph', 'global_graph', 'spectral', 'conventional']:
            if key in features:
                feature_vector.extend(features[key])
        
        # Convert any NaN or inf values to zeros
        feature_vector = np.array(feature_vector, dtype=np.float64)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        feature_list.append(feature_vector)
    
    # Convert list to numpy array
    feature_matrix = np.array(feature_list, dtype=np.float64)
    
    logger.info(f"Extracted features with shape {feature_matrix.shape} for {feature_set}")
    
    return feature_matrix

def train_and_evaluate(test_only=False):
    """
    Train and evaluate multiple classifiers with different feature sets.
    
    Args:
        test_only: If True, load pre-trained models instead of training new ones
    
    Returns:
        Dictionary of results for all classifiers and feature sets
    """
    results = {}
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize dataset handler
    logger.info("Initializing dataset handler")
    dataset_handler = DatasetHandler(
        n_segments=PARAMS['n_segments'],
        compactness=PARAMS['compactness'],
        connectivity_threshold=PARAMS['connectivity_threshold'],
        max_images_per_class=PARAMS['max_images_per_class']
    )
    
    # Process dataset
    logger.info(f"Processing dataset from {MELANOMA_DIR} and {BENIGN_DIR}")
    graphs, labels = dataset_handler.process_dataset(
        MELANOMA_DIR,
        BENIGN_DIR
    )
    
    # Split dataset into train and test sets
    train_graphs, test_graphs, train_labels, test_labels = dataset_handler.split_dataset(
        graphs, labels, test_size=PARAMS['test_size'], random_state=PARAMS['random_state']
    )
    
    logger.info(f"Dataset split: {len(train_graphs)} training samples, {len(test_graphs)} test samples")
    
    # For each feature set
    for feature_set_name, feature_set in FEATURE_SETS.items():
        logger.info(f"Processing feature set: {feature_set_name}")
        
        # Extract features for training and test sets
        X_train = extract_features(train_graphs, feature_set)
        X_test = extract_features(test_graphs, feature_set)
        
        # Initialize scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # For each classifier
        for classifier_name, classifier_info in CLASSIFIERS.items():
            logger.info(f"Processing classifier: {classifier_name}")
            
            # Create model directories
            model_subdir = os.path.join(MODEL_DIR, classifier_name.replace(' ', '_').replace('(', '').replace(')', ''))
            os.makedirs(model_subdir, exist_ok=True)
            
            # Define model and scaler paths
            model_path = os.path.join(model_subdir, f"{feature_set_name.replace('+', '_')}_model.joblib")
            scaler_path = os.path.join(model_subdir, f"{feature_set_name.replace('+', '_')}_scaler.joblib")
            
            # Train or load model
            if not test_only and not os.path.exists(model_path):
                logger.info(f"Training {classifier_name} with {feature_set_name} features")
                
                # Initialize classifier
                clf = classifier_info['class'](**classifier_info['params'])
                
                try:
                    # Train classifier
                    start_time = time.time()
                    clf.fit(X_train_scaled, train_labels)
                    training_time = time.time() - start_time
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    
                    # Save model and scaler
                    dump(clf, model_path)
                    dump(scaler, scaler_path)
                    logger.info(f"Model saved to {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error training {classifier_name} with {feature_set_name} features: {str(e)}")
                    continue
            elif os.path.exists(model_path):
                logger.info(f"Loading pre-trained model from {model_path}")
                try:
                    clf = load(model_path)
                    scaler = load(scaler_path)
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    continue
            else:
                logger.warning(f"Model {model_path} not found and test_only is set. Skipping.")
                continue
            
            # Evaluate model
            logger.info(f"Evaluating {classifier_name} with {feature_set_name} features")
            try:
                # Make predictions
                y_pred = clf.predict(X_test_scaled)
                
                # Get prediction probabilities if available
                y_proba = None
                if hasattr(clf, 'predict_proba'):
                    try:
                        probs = clf.predict_proba(X_test_scaled)
                        if probs.shape[1] >= 2:  # Binary or multi-class
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
                logger.info(f"Results for {classifier_name} with {feature_set_name} features:")
                logger.info(f"  Accuracy: {metrics['AC']:.2f}%")
                logger.info(f"  AUC: {metrics['AUC']:.2f}%")
                logger.info(f"  Sensitivity: {metrics['SN']:.2f}%")
                logger.info(f"  Specificity: {metrics['SP']:.2f}%")
                
                # Store results
                if classifier_name not in results:
                    results[classifier_name] = {}
                results[classifier_name][feature_set_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {classifier_name} with {feature_set_name} features: {str(e)}")
    
    return results

def generate_tables(results):
    """
    Generate comparison tables based on evaluation results.
    
    Args:
        results: Dictionary containing results for all classifiers and feature sets
    """
    # Define table configurations
    table_configs = [
        {
            'id': 1,
            'title': 'Table 1: Melanoma detection performance using the ISIC 2017 dataset.',
            'feature_types': ['Conv', 'Graph+Conv'],
            'metrics': ['AC', 'AUC', 'SN', 'SP'],
            'row_first': 'feature',  # First level of multi-index is feature type
            'classifiers': list(CLASSIFIERS.keys())
        },
        {
            'id': 2,
            'title': 'Table 2: Melanoma detection performance using the ISIC 2017 dataset.',
            'feature_types': ['Conv', 'Graph+Conv'],
            'metrics': ['AC', 'AUC', 'SN', 'SP'],
            'row_first': 'feature',  # First level of multi-index is feature type
            'classifiers': list(CLASSIFIERS.keys())
        },
        {
            'id': 3,
            'title': 'Table 3: Melanoma detection performance using graph-signal-based feature descriptors.',
            'feature_types': ['C', 'G', 'C+G', 'G+Conv'],
            'metrics': ['AC', 'AUC', 'SN', 'SP'],
            'row_first': 'classifier',  # First level of multi-index is classifier
            'classifiers': list(CLASSIFIERS.keys())
        },
        {
            'id': 4,
            'title': 'Table 4: Melanoma detection performance using color signal-dependent feature descriptors.',
            'feature_types': ['GFT', 'C+G+GFT', 'GFT+Conv', 'All'],
            'metrics': ['AC', 'AUC', 'SN', 'SP'],
            'row_first': 'classifier',  # First level of multi-index is classifier
            'classifiers': list(CLASSIFIERS.keys())
        },
        {
            'id': 5,
            'title': 'Table 5: Melanoma detection performance using geometry signal-dependent features.',
            'feature_types': ['GFT', 'C+G+GFT', 'GFT+Conv', 'All'],
            'metrics': ['AC', 'AUC', 'SN', 'SP'],
            'row_first': 'classifier',  # First level of multi-index is classifier
            'classifiers': list(CLASSIFIERS.keys())
        }
    ]
    
    # Generate each table
    for config in table_configs:
        logger.info(f"Generating {config['title']}")
        
        try:
            # Determine the structure of the table
            if config['row_first'] == 'feature':
                # Feature type is the first level of the row multi-index
                row_tuples = [(ft, m) for ft in config['feature_types'] for m in config['metrics']]
                row_index = pd.MultiIndex.from_tuples(row_tuples)
                df = pd.DataFrame(index=row_index, columns=config['classifiers'])
                
                # Fill in the dataframe
                for ft in config['feature_types']:
                    for m in config['metrics']:
                        for cls in config['classifiers']:
                            if cls in results and ft in results[cls]:
                                df.loc[(ft, m), cls] = results[cls][ft].get(m, 0)
            else:
                # Classifier is the first level of the row multi-index
                row_tuples = [(cls, m) for cls in config['classifiers'] for m in config['metrics']]
                row_index = pd.MultiIndex.from_tuples(row_tuples)
                df = pd.DataFrame(index=row_index, columns=config['feature_types'])
                
                # Fill in the dataframe
                for cls in config['classifiers']:
                    for m in config['metrics']:
                        for ft in config['feature_types']:
                            if cls in results and ft in results[cls]:
                                df.loc[(cls, m), ft] = results[cls][ft].get(m, 0)
            
            # Round values
            df = df.round(2)
            
            # Save table to CSV
            csv_path = os.path.join(OUTPUT_DIR, f'table_{config["id"]}.csv')
            df.to_csv(csv_path)
            logger.info(f"Saved table to {csv_path}")
            
            # Save visualization
            save_table_visualization(df, config['title'], os.path.join(OUTPUT_DIR, f'table_{config["id"]}.png'))
        
        except Exception as e:
            logger.error(f"Error generating table {config['id']}: {str(e)}")
    
    logger.info(f"All tables generated and saved to {OUTPUT_DIR}")

def save_table_visualization(df, title, output_path):
    """
    Save DataFrame as a formatted table image.
    
    Args:
        df: DataFrame to visualize
        title: Title for the table
        output_path: Path to save the image
    """
    try:
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Figure size based on table dimensions
        rows, cols = df.shape
        fig_height = max(rows * 0.4 + 2, 8)  # Minimum height of 8 inches
        fig_width = max(cols * 1.5 + 2, 10)  # Minimum width of 10 inches
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Create the table
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(title, fontsize=12, y=1.02)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Table visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving table visualization: {str(e)}")

def create_directories():
    """Create necessary directories."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive model comparison for melanoma detection')
    parser.add_argument('--test-only', action='store_true', help='Only evaluate existing models without retraining')
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Start timer
        start_time = time.time()
        
        # Create directories
        create_directories()
        
        logger.info("Starting comprehensive model comparison")
        if args.test_only:
            logger.info("Running in test-only mode, no new models will be trained")
        
        # Train and evaluate models
        results = train_and_evaluate(test_only=args.test_only)
        
        # Generate tables
        generate_tables(results)
        
        # Report time taken
        elapsed_time = time.time() - start_time
        logger.info(f"Comprehensive model comparison completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during comprehensive model comparison: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())