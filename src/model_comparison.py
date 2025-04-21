"""
Model comparison utility for the melanoma detection system.
This module creates comparison tables for different classifiers and feature sets.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

# Define specificity score (not in sklearn)
def specificity_score(y_true, y_pred):
    """Calculate the specificity score (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Set up logging
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, output_dir='output'):
        """Initialize model comparison with output directory for results."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = defaultdict(dict)
        
    def evaluate_classifier(self, name, y_true, y_pred, y_proba=None, feature_type='All'):
        """
        Evaluate a classifier and store results.
        
        Args:
            name: Classifier name (e.g., 'SVM (RBF)')
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            feature_type: Type of features used (e.g., 'Conv', 'Graph', 'Conv+Graph')
        """
        try:
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred) * 100
            prec = precision_score(y_true, y_pred) * 100
            rec = recall_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred) * 100
            
            # Calculate specificity (true negative rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = (tn / (tn + fp)) * 100
            
            # Calculate AUC if probabilities are provided
            auc = 0
            if y_proba is not None:
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    # For multi-class, use the positive class probability
                    auc = roc_auc_score(y_true, y_proba[:, 1]) * 100
                else:
                    # For binary, use the provided probability
                    auc = roc_auc_score(y_true, y_proba) * 100
            
            # Store results
            if feature_type not in self.results[name]:
                self.results[name][feature_type] = {}
                
            self.results[name][feature_type]['AC'] = acc
            self.results[name][feature_type]['AUC'] = auc
            self.results[name][feature_type]['SN'] = rec
            self.results[name][feature_type]['SP'] = spec
            
            logger.info(f"Metrics for {name} with {feature_type} features:")
            logger.info(f"  Accuracy: {acc:.2f}%")
            logger.info(f"  AUC: {auc:.2f}%")
            logger.info(f"  Sensitivity (Recall): {rec:.2f}%")
            logger.info(f"  Specificity: {spec:.2f}%")
            
            return {
                'accuracy': acc,
                'auc': auc,
                'sensitivity': rec,
                'specificity': spec,
                'precision': prec,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classifier {name}: {e}")
            return None
            
    def add_comparison_result(self, classifier_name, feature_type, metrics):
        """
        Add pre-computed comparison result.
        
        Args:
            classifier_name: Name of the classifier
            feature_type: Type of features used
            metrics: Dictionary containing metrics (AC, AUC, SN, SP)
        """
        if feature_type not in self.results[classifier_name]:
            self.results[classifier_name][feature_type] = {}
            
        for metric_name, value in metrics.items():
            self.results[classifier_name][feature_type][metric_name] = value
    
    def generate_table_1(self, output_path=None):
        """
        Generate Table 1: Melanoma detection performance using the ISIC 2017 dataset.
        Compares different classifiers with conventional and graph features.
        
        Table format as in the paper:
                    SVM     SVM         SVM      KNN      MLP      RF
                    (RBF)   (Sigmoid)   (Poly)
        Conv   AC    -       -           -        -        -       -
               AUC    -       -           -        -        -       -
               SN     -       -           -        -        -       -
               SP     -       -           -        -        -       -
        Graph  AC     -       -           -        -        -       -
        +Conv  AUC    -       -           -        -        -       -
               SN     -       -           -        -        -       -
               SP     -       -           -        -        -       -
        """
        try:
            # Define classifiers and metrics in the order they should appear
            classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
            feature_types = ['Conv', 'Graph+Conv']
            metrics = ['AC', 'AUC', 'SN', 'SP']
            
            # Create multi-index for rows
            row_tuples = [(ft, m) for ft in feature_types for m in metrics]
            row_index = pd.MultiIndex.from_tuples(row_tuples)
            
            # Create DataFrame
            df = pd.DataFrame(index=row_index, columns=classifiers)
            
            # Fill DataFrame with results
            for cls in classifiers:
                if cls in self.results:
                    for ft in feature_types:
                        if ft in self.results[cls]:
                            for m in metrics:
                                if m in self.results[cls][ft]:
                                    df.loc[(ft, m), cls] = self.results[cls][ft][m]
            
            # Format the table
            df = df.round(2)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path)
                logger.info(f"Table 1 saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating Table 1: {e}")
            return None
            
    def generate_table_2(self, output_path=None):
        """
        Generate Table 2: Melanoma detection performance using the ISIC 2017 dataset.
        Shows performance for seven classifiers and two feature types.
        
        Table format as in the paper:
                              SVM     SVM         SVM      KNN      MLP      RF
                              (RBF)   (Sigmoid)   (Poly)
        Conv         AC        -       -           -        -        -       -
                     AUC       -       -           -        -        -       -
                     SN        -       -           -        -        -       -
                     SP        -       -           -        -        -       -
        Graph+Conv   AC        -       -           -        -        -       -
                     AUC       -       -           -        -        -       -
                     SN        -       -           -        -        -       -
                     SP        -       -           -        -        -       -
        """
        try:
            # Define classifiers and metrics in the order they should appear
            classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
            feature_types = ['Conv', 'Graph+Conv']
            metrics = ['AC', 'AUC', 'SN', 'SP']
            
            # This is similar to Table 1 but organized a bit differently
            # Create multi-index for rows
            row_tuples = [(ft, m) for ft in feature_types for m in metrics]
            row_index = pd.MultiIndex.from_tuples(row_tuples)
            
            # Create DataFrame
            df = pd.DataFrame(index=row_index, columns=classifiers)
            
            # Fill DataFrame with results
            for cls in classifiers:
                if cls in self.results:
                    for ft in feature_types:
                        if ft in self.results[cls]:
                            for m in metrics:
                                if m in self.results[cls][ft]:
                                    df.loc[(ft, m), cls] = self.results[cls][ft][m]
            
            # Format the table
            df = df.round(2)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path)
                logger.info(f"Table 2 saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating Table 2: {e}")
            return None
            
    def generate_table_3(self, output_path=None):
        """
        Generate Table 3: Melanoma detection performance using graph-signal-based feature descriptors.
        
        Table format:
                           C       G       C+G     G+GFT+Conv
        SVM      AC        -       -        -         -
        (RBF)    AUC       -       -        -         -
                 SN        -       -        -         -
                 SP        -       -        -         -
        SVM      AC        -       -        -         -
        (Sigmoid)AUC       -       -        -         -
                 SN        -       -        -         -
                 SP        -       -        -         -
        ...
        """
        try:
            # Define classifiers and metrics in the order they should appear
            classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
            feature_types = ['C', 'G', 'C+G', 'G+Conv']
            metrics = ['AC', 'AUC', 'SN', 'SP']
            
            # Create multi-index for rows
            row_tuples = [(cls, m) for cls in classifiers for m in metrics]
            row_index = pd.MultiIndex.from_tuples(row_tuples)
            
            # Create DataFrame
            df = pd.DataFrame(index=row_index, columns=feature_types)
            
            # Fill DataFrame with results
            for cls in classifiers:
                if cls in self.results:
                    for ft in feature_types:
                        if ft in self.results[cls]:
                            for m in metrics:
                                if m in self.results[cls][ft]:
                                    df.loc[(cls, m), ft] = self.results[cls][ft][m]
            
            # Format the table
            df = df.round(2)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path)
                logger.info(f"Table 3 saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating Table 3: {e}")
            return None
            
    def generate_table_4(self, output_path=None):
        """
        Generate Table 4: Melanoma detection performance using color signal-dependent feature descriptors.
        
        Table format:
                           GFT     C+G+GFT     GFT+Conv     All
        SVM      AC        -         -            -          -
        (RBF)    AUC       -         -            -          -
                 SN        -         -            -          -
                 SP        -         -            -          -
        ...
        """
        try:
            # Define classifiers and metrics in the order they should appear
            classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
            feature_types = ['GFT', 'C+G+GFT', 'GFT+Conv', 'All']
            metrics = ['AC', 'AUC', 'SN', 'SP']
            
            # Create multi-index for rows
            row_tuples = [(cls, m) for cls in classifiers for m in metrics]
            row_index = pd.MultiIndex.from_tuples(row_tuples)
            
            # Create DataFrame
            df = pd.DataFrame(index=row_index, columns=feature_types)
            
            # Fill DataFrame with results
            for cls in classifiers:
                if cls in self.results:
                    for ft in feature_types:
                        if ft in self.results[cls]:
                            for m in metrics:
                                if m in self.results[cls][ft]:
                                    df.loc[(cls, m), ft] = self.results[cls][ft][m]
            
            # Format the table
            df = df.round(2)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path)
                logger.info(f"Table 4 saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating Table 4: {e}")
            return None
            
    def generate_table_5(self, output_path=None):
        """
        Generate Table 5: Melanoma detection performance using geometry signal-dependent features.
        
        Table format:
                           GFT     C+G+GFT     GFT+Conv     All
        SVM      AC        -         -            -          -
        (RBF)    AUC       -         -            -          -
                 SN        -         -            -          -
                 SP        -         -            -          -
        ...
        """
        try:
            # Define classifiers and metrics in the order they should appear
            classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
            feature_types = ['GFT', 'C+G+GFT', 'GFT+Conv', 'All']
            metrics = ['AC', 'AUC', 'SN', 'SP']
            
            # Create multi-index for rows
            row_tuples = [(cls, m) for cls in classifiers for m in metrics]
            row_index = pd.MultiIndex.from_tuples(row_tuples)
            
            # Create DataFrame
            df = pd.DataFrame(index=row_index, columns=feature_types)
            
            # Fill DataFrame with results
            for cls in classifiers:
                if cls in self.results:
                    for ft in feature_types:
                        if ft in self.results[cls]:
                            for m in metrics:
                                if m in self.results[cls][ft]:
                                    df.loc[(cls, m), ft] = self.results[cls][ft][m]
            
            # Format the table
            df = df.round(2)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path)
                logger.info(f"Table 5 saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating Table 5: {e}")
            return None
            
    def visualize_tables(self, output_dir=None):
        """
        Generate visualizations of all tables.
        
        Args:
            output_dir: Directory to save visualizations (defaults to self.output_dir)
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and visualize tables
        tables = [
            (self.generate_table_1(), "Table 1: Melanoma detection performance using the ISIC 2017 dataset."),
            (self.generate_table_2(), "Table 2: Melanoma detection performance using the ISIC 2017 dataset."),
            (self.generate_table_3(), "Table 3: Melanoma detection performance using graph-signal-based feature descriptors."),
            (self.generate_table_4(), "Table 4: Melanoma detection performance using color signal-dependent feature descriptors."),
            (self.generate_table_5(), "Table 5: Melanoma detection performance using geometry signal-dependent features.")
        ]
        
        for i, (table, title) in enumerate(tables, 1):
            if table is not None:
                self._save_table_as_image(table, os.path.join(output_dir, f"table_{i}.png"), title)
    
    def _save_table_as_image(self, df, output_path, title):
        """
        Save DataFrame as a formatted table image.
        
        Args:
            df: DataFrame to visualize
            output_path: Path to save the image
            title: Title for the table
        """
        try:
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))
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
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Add title
            plt.title(title, fontsize=12, pad=20)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Table visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving table as image: {e}")