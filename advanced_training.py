"""
Advanced Trainer for Skin Lesion Classification
Optimized for achieving 98%+ accuracy using ensemble methods and advanced preprocessing

STANDALONE VERSION - Run directly with: python advanced_training.py --features_file your_features.pkl
SUPPORTS: CSV and PKL files with automatic detection
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFECV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Advanced boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN


class AdvancedSkinLesionTrainer:
    """
    Advanced trainer for skin lesion classification with state-of-the-art techniques
    for achieving 98%+ accuracy.
    
    Features:
    - Advanced feature selection and engineering
    - Multiple resampling strategies for class imbalance
    - Aggressive hyperparameter optimization
    - Stacked ensemble models
    - Comprehensive evaluation metrics
    - Support for both CSV and PKL files
    - Preserves feature names throughout pipeline
    """
    
    def __init__(self, output_dir='output/advanced_training', random_state=42):
        """
        Initialize the advanced trainer.
        
        Args:
            output_dir: Directory for saving outputs
            random_state: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/models', exist_ok=True)
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.output_dir}/reports', exist_ok=True)
        
        # Initialize components
        self.scaler = None
        self.selector = None
        self.resampler = None
        self.best_model = None
        self.ensemble = None
        self.feature_names = None  # Store feature names
        self.selected_feature_names = None  # Store selected feature names
        
        self.logger.info("AdvancedSkinLesionTrainer initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger = logging.getLogger('AdvancedTrainer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler
        os.makedirs(self.output_dir, exist_ok=True)
        fh = logging.FileHandler(f'{self.output_dir}/training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def clean_features(self, X_df, feature_names=None):
        """
        Advanced feature cleaning with multiple strategies.
        Preserves DataFrame structure and feature names.
        
        Args:
            X_df: Feature DataFrame or array
            feature_names: List of feature names (optional if X_df is DataFrame)
            
        Returns:
            X_clean: Cleaned feature DataFrame
            cleaned_feature_names: Updated feature names
        """
        # Convert to DataFrame if needed
        if not isinstance(X_df, pd.DataFrame):
            if feature_names is not None:
                X_df = pd.DataFrame(X_df, columns=feature_names)
            else:
                X_df = pd.DataFrame(X_df, columns=[f'feature_{i}' for i in range(X_df.shape[1])])
        
        self.logger.info(f"Cleaning features: {X_df.shape}")
        
        # 1. Handle infinite values
        inf_mask = np.isinf(X_df.values)
        if inf_mask.any():
            self.logger.warning(f"Found {inf_mask.sum()} infinite values")
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Handle NaN values with median imputation
        nan_count = X_df.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values")
            X_df = X_df.fillna(X_df.median())
        
        # 3. Remove zero-variance features
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=1e-8)
        variance_mask = variance_selector.fit_transform(X_df.values).shape[1] == X_df.shape[1]
        
        if not variance_mask:
            variance_support = variance_selector.get_support()
            removed = (~variance_support).sum()
            self.logger.info(f"Removing {removed} zero-variance features")
            X_df = X_df.loc[:, variance_support]
        
        # 4. Remove highly correlated features (>0.95)
        if X_df.shape[1] > 1:
            corr_matrix = X_df.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            
            if to_drop:
                self.logger.info(f"Removing {len(to_drop)} highly correlated features")
                X_df = X_df.drop(columns=to_drop)
        
        # 5. Cap outliers using IQR method (3x IQR)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            X_df[col] = X_df[col].clip(lower, upper)
        
        self.logger.info(f"Feature cleaning complete: {X_df.shape}")
        
        return X_df, X_df.columns.tolist()
    
    def select_features(self, X_train_df, y_train, X_test_df, n_features=100, method='mutual_info'):
        """
        Advanced feature selection using multiple methods.
        Preserves DataFrame structure and feature names.
        
        Args:
            X_train_df, y_train: Training data (DataFrame)
            X_test_df: Test data (DataFrame)
            n_features: Number of features to select
            method: Selection method ('mutual_info', 'rfe', 'recursive')
            
        Returns:
            X_train_selected: Selected training features (DataFrame)
            X_test_selected: Selected test features (DataFrame)
            selected_feature_names: List of selected feature names
        """
        self.logger.info(f"Feature selection using {method}, selecting {n_features} features")
        
        n_features = min(n_features, X_train_df.shape[1])
        
        if method == 'mutual_info':
            # Mutual information for non-linear relationships
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train_df.values, y_train)
            X_test_selected = selector.transform(X_test_df.values)
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = X_train_df.columns[selected_indices].tolist()
            
        elif method == 'rfe':
            # Recursive Feature Elimination with XGBoost
            base_estimator = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            
            selector = RFECV(
                estimator=base_estimator,
                step=1,
                cv=StratifiedKFold(5),
                scoring='f1',
                n_jobs=-1
            )
            
            X_train_selected = selector.fit_transform(X_train_df.values, y_train)
            X_test_selected = selector.transform(X_test_df.values)
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = X_train_df.columns[selected_indices].tolist()
            
            self.logger.info(f"RFE selected {len(selected_indices)} features")
            
        elif method == 'recursive':
            # Custom recursive feature importance
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train_df.values, y_train)
            
            importances = rf.feature_importances_
            selected_indices = np.argsort(importances)[-n_features:]
            selected_feature_names = X_train_df.columns[selected_indices].tolist()
            
            X_train_selected = X_train_df.iloc[:, selected_indices].values
            X_test_selected = X_test_df.iloc[:, selected_indices].values
            
        else:
            # Default: use all features
            X_train_selected = X_train_df.values
            X_test_selected = X_test_df.values
            selected_feature_names = X_train_df.columns.tolist()
        
        self.selector = selector if 'selector' in locals() else None
        self.selected_feature_names = selected_feature_names
        
        self.logger.info(f"Selected {len(selected_feature_names)} features")
        
        # Convert back to DataFrames with feature names
        X_train_df_selected = pd.DataFrame(X_train_selected, columns=selected_feature_names)
        X_test_df_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names)
        
        return X_train_df_selected, X_test_df_selected, selected_feature_names
    
    def handle_imbalance(self, X_train_df, y_train, strategy='smote_tomek'):
        """
        Handle class imbalance with multiple strategies.
        Preserves DataFrame structure.
        
        Args:
            X_train_df: Training data (DataFrame)
            y_train: Training labels
            strategy: Resampling strategy
                
        Returns:
            X_resampled_df: Resampled data (DataFrame)
            y_resampled: Resampled labels
        """
        if strategy == 'none':
            return X_train_df, y_train
        
        self.logger.info(f"Handling class imbalance using {strategy}")
        
        class_counts = np.bincount(y_train)
        self.logger.info(f"Original class distribution: {class_counts}")
        
        if strategy == 'smote':
            resampler = SMOTE(random_state=self.random_state)
        elif strategy == 'adasyn':
            resampler = ADASYN(random_state=self.random_state)
        elif strategy == 'smote_tomek':
            resampler = SMOTETomek(random_state=self.random_state)
        elif strategy == 'smote_enn':
            resampler = SMOTEENN(random_state=self.random_state)
        elif strategy == 'borderline':
            resampler = BorderlineSMOTE(random_state=self.random_state)
        else:
            self.logger.warning(f"Unknown strategy {strategy}, using SMOTE")
            resampler = SMOTE(random_state=self.random_state)
        
        try:
            X_resampled, y_resampled = resampler.fit_resample(X_train_df.values, y_train)
            
            new_class_counts = np.bincount(y_resampled)
            self.logger.info(f"Resampled class distribution: {new_class_counts}")
            
            self.resampler = resampler
            
            # Convert back to DataFrame with same feature names
            X_resampled_df = pd.DataFrame(X_resampled, columns=X_train_df.columns)
            
            return X_resampled_df, y_resampled
            
        except Exception as e:
            self.logger.error(f"Resampling failed: {str(e)}")
            return X_train_df, y_train
    
    def create_base_models(self):
        """
        Create base models with optimized hyperparameters for 98%+ accuracy.
        
        Returns:
            Dictionary of base models
        """
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric='auc',
                n_jobs=-1
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
                force_row_wise=True  # Suppresses feature name warnings
            ),
            
            'CatBoost': CatBoostClassifier(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5.0,
                loss_function='Logloss',
                eval_metric='AUC',
                random_state=self.random_state,
                verbose=False
            ),
            
            'RandomForest': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                random_state=self.random_state
            )
        }
        
        return models
    
    def create_stacked_ensemble(self, base_models):
        """
        Create a stacked ensemble of base models.
        
        Args:
            base_models: Dictionary of base models
            
        Returns:
            Stacked classifier
        """
        self.logger.info("Creating stacked ensemble")
        
        # Base estimators as list of tuples
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner (final estimator)
        final_estimator = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=self.random_state,
            eval_metric='auc'
        )
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_clf
    
    def create_voting_ensemble(self, base_models):
        """
        Create a soft voting ensemble.
        
        Args:
            base_models: Dictionary of base models
            
        Returns:
            Voting classifier
        """
        self.logger.info("Creating voting ensemble")
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_clf
    
    def optimize_hyperparameters(self, model, X_train_df, y_train, model_name):
        """
        Optimize hyperparameters for a single model.
        Uses DataFrame to preserve feature names.
        
        Args:
            model: Model to optimize
            X_train_df: Training data (DataFrame)
            y_train: Training labels
            model_name: Name of the model
            
        Returns:
            Optimized model
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}")
        
        # Define parameter grids for each model type
        param_grids = {
            'XGBoost': {
                'n_estimators': [400, 600, 800],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'gamma': [0, 0.1, 0.5],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            },
            
            'LightGBM': {
                'n_estimators': [400, 600, 800],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            
            'CatBoost': {
                'iterations': [500, 800, 1000],
                'learning_rate': [0.03, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [3, 5, 7]
            },
            
            'RandomForest': {
                'n_estimators': [300, 500, 700],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            
            'ExtraTrees': {
                'n_estimators': [300, 500, 700],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            
            'GradientBoosting': {
                'n_estimators': [200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9]
            }
        }
        
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            self.logger.warning(f"No parameter grid for {model_name}, returning original model")
            return model
        
        # Calculate number of combinations
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        
        # Use RandomizedSearchCV for large grids
        if n_combinations > 50:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=50,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        else:
            search = GridSearchCV(
                model,
                param_grid,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train_df, y_train)
        
        self.logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        self.logger.info(f"Best CV F1 score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def evaluate_model(self, model, X_test_df, y_test, model_name):
        """
        Comprehensive model evaluation.
        Uses DataFrame to preserve feature names.
        
        Args:
            model: Trained model
            X_test_df: Test data (DataFrame)
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test_df)
        y_pred_proba = model.predict_proba(X_test_df)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Specificity
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Log results
        self.logger.info(f"\n{model_name} Test Results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        if 'roc_auc' in metrics:
            self.logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def plot_results(self, results, y_test):
        """
        Generate comprehensive visualizations.
        
        Args:
            results: Dictionary of results for each model
            y_test: True labels
        """
        self.logger.info("Generating visualizations")
        
        # 1. Model Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            values = [results[model_name][metric] * 100 for metric in metrics_to_plot]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Combined ROC Curves
        if any('y_pred_proba' in results[m] for m in results):
            plt.figure(figsize=(10, 8))
            
            for model_name, result in results.items():
                if 'y_pred_proba' in result:
                    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                    roc_auc = result.get('roc_auc', 0)
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/roc_curves_all.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Confusion Matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, result) in enumerate(results.items()):
            if idx >= 6:
                break
            
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['SK', 'BCC'], yticklabels=['SK', 'BCC'])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(len(results), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {self.output_dir}/plots/")
    
    def train_complete_pipeline(self, X, y, feature_names=None, 
                               optimize=True, use_ensemble=True,
                               resampling_strategy='smote_tomek',
                               n_features=100):
        """
        Complete training pipeline with all optimizations.
        Uses DataFrames throughout to preserve feature names.
        
        Args:
            X: Feature matrix (array or DataFrame)
            y: Labels
            feature_names: List of feature names (optional if X is DataFrame)
            optimize: Whether to optimize hyperparameters
            use_ensemble: Whether to create ensemble models
            resampling_strategy: Strategy for handling imbalance
            n_features: Number of features to select
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("="*80)
        self.logger.info("ADVANCED TRAINING PIPELINE FOR 98%+ ACCURACY")
        self.logger.info("="*80)
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self.feature_names = X.columns.tolist()
        
        # 1. Clean features
        X_clean_df, cleaned_names = self.clean_features(X, self.feature_names)
        
        # 2. Split data (keeping DataFrame structure)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_clean_df, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"Train set: {X_train_df.shape}, Test set: {X_test_df.shape}")
        
        # 3. Feature selection (keeping DataFrame structure)
        X_train_selected_df, X_test_selected_df, selected_feature_names = self.select_features(
            X_train_df, y_train, X_test_df, n_features=n_features, method='mutual_info'
        )
        
        # 4. Handle class imbalance (keeping DataFrame structure)
        X_train_resampled_df, y_train_resampled = self.handle_imbalance(
            X_train_selected_df, y_train, strategy=resampling_strategy
        )
        
        # 5. Scale features (keeping DataFrame structure)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_resampled_df)
        X_test_scaled = self.scaler.transform(X_test_selected_df)
        
        # Convert back to DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_resampled_df.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_selected_df.columns)
        
        self.logger.info("Feature preprocessing complete")
        
        # 6. Train base models
        base_models = self.create_base_models()
        results = {}
        trained_models = {}
        
        for name, model in base_models.items():
            self.logger.info(f"\nTraining {name}...")
            
            # Optimize if requested
            if optimize:
                model = self.optimize_hyperparameters(
                    model, X_train_scaled_df, y_train_resampled, name
                )
            
            # Train
            model.fit(X_train_scaled_df, y_train_resampled)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test_scaled_df, y_test, name)
            
            # Store predictions for ensemble
            y_pred_proba = model.predict_proba(X_test_scaled_df)[:, 1] if hasattr(model, 'predict_proba') else None
            metrics['y_pred_proba'] = y_pred_proba
            
            results[name] = metrics
            trained_models[name] = model
        
        # 7. Create ensemble models if requested
        if use_ensemble:
            self.logger.info("\n" + "="*80)
            self.logger.info("CREATING ENSEMBLE MODELS")
            self.logger.info("="*80)
            
            # Stacking ensemble
            stacking_clf = self.create_stacked_ensemble(trained_models)
            self.logger.info("Training stacking ensemble...")
            stacking_clf.fit(X_train_scaled_df, y_train_resampled)
            results['Stacking Ensemble'] = self.evaluate_model(
                stacking_clf, X_test_scaled_df, y_test, 'Stacking Ensemble'
            )
            y_pred_proba = stacking_clf.predict_proba(X_test_scaled_df)[:, 1]
            results['Stacking Ensemble']['y_pred_proba'] = y_pred_proba
            trained_models['Stacking Ensemble'] = stacking_clf
            
            # Voting ensemble
            voting_clf = self.create_voting_ensemble(trained_models)
            self.logger.info("Training voting ensemble...")
            voting_clf.fit(X_train_scaled_df, y_train_resampled)
            results['Voting Ensemble'] = self.evaluate_model(
                voting_clf, X_test_scaled_df, y_test, 'Voting Ensemble'
            )
            y_pred_proba = voting_clf.predict_proba(X_test_scaled_df)[:, 1]
            results['Voting Ensemble']['y_pred_proba'] = y_pred_proba
            trained_models['Voting Ensemble'] = voting_clf
        
        # 8. Select best model
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = trained_models[best_model_name]
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"BEST MODEL: {best_model_name}")
        self.logger.info(f"F1 Score: {results[best_model_name]['f1']:.4f}")
        self.logger.info("="*80)
        
        # 9. Generate visualizations
        self.plot_results(results, y_test)
        
        # 10. Save models and results
        self._save_results(trained_models, results, best_model_name)
        
        # 11. Generate final report
        self._generate_report(results, best_model_name)
        
        return {
            'results': results,
            'models': trained_models,
            'best_model_name': best_model_name,
            'best_model': self.best_model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names
        }
    
    def _save_results(self, models, results, best_model_name):
        """Save trained models and results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        dump(self.best_model, f'{self.output_dir}/models/best_model_{timestamp}.joblib')
        dump(self.scaler, f'{self.output_dir}/models/scaler_{timestamp}.joblib')
        if self.selector:
            dump(self.selector, f'{self.output_dir}/models/selector_{timestamp}.joblib')
        
        # Save feature names
        if self.feature_names:
            dump(self.feature_names, f'{self.output_dir}/models/feature_names_{timestamp}.joblib')
        if self.selected_feature_names:
            dump(self.selected_feature_names, f'{self.output_dir}/models/selected_feature_names_{timestamp}.joblib')
        
        # Save all models
        for name, model in models.items():
            safe_name = name.replace(' ', '_').lower()
            dump(model, f'{self.output_dir}/models/{safe_name}_{timestamp}.joblib')
        
        # Save results as JSON
        results_json = {}
        for name, metrics in results.items():
            results_json[name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else 
                   v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
                if k != 'y_pred_proba'  # Skip probability predictions
            }
        
        with open(f'{self.output_dir}/reports/results_{timestamp}.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        self.logger.info(f"Models and results saved to {self.output_dir}/")
    
    def _generate_report(self, results, best_model_name):
        """Generate comprehensive text report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_lines = [
            "="*80,
            "ADVANCED SKIN LESION CLASSIFICATION - TRAINING REPORT",
            "="*80,
            f"Generated: {timestamp}",
            "",
            "="*80,
            "SUMMARY",
            "="*80,
            f"Best Model: {best_model_name}",
            f"Best F1 Score: {results[best_model_name]['f1']:.4f}",
            f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}",
            "",
            "="*80,
            "DETAILED RESULTS",
            "="*80,
            ""
        ]
        
        # Table header
        report_lines.append(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Specificity':<12} {'AUC':<10}")
        report_lines.append("-"*95)
        
        # Sort by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for name, metrics in sorted_results:
            acc = f"{metrics['accuracy']*100:.2f}%"
            prec = f"{metrics['precision']*100:.2f}%"
            rec = f"{metrics['recall']*100:.2f}%"
            f1 = f"{metrics['f1']*100:.2f}%"
            spec = f"{metrics['specificity']*100:.2f}%"
            auc_val = f"{metrics.get('roc_auc', 0)*100:.2f}%" if 'roc_auc' in metrics else "N/A"
            
            report_lines.append(f"{name:<25} {acc:<10} {prec:<10} {rec:<10} {f1:<10} {spec:<12} {auc_val:<10}")
        
        report_lines.extend([
            "",
            "="*80,
            "DETAILED METRICS FOR BEST MODEL",
            "="*80,
            f"Model: {best_model_name}",
            f"Accuracy: {results[best_model_name]['accuracy']:.4f}",
            f"Precision: {results[best_model_name]['precision']:.4f}",
            f"Recall/Sensitivity: {results[best_model_name]['recall']:.4f}",
            f"Specificity: {results[best_model_name]['specificity']:.4f}",
            f"F1 Score: {results[best_model_name]['f1']:.4f}",
        ])
        
        if 'roc_auc' in results[best_model_name]:
            report_lines.append(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        report_lines.extend([
            "",
            "Confusion Matrix:",
            str(results[best_model_name]['confusion_matrix']),
            "",
            "="*80
        ])
        
        # Save report
        report_file = f'{self.output_dir}/reports/training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to logger
        self.logger.info('\n' + '\n'.join(report_lines))
        
        self.logger.info(f"Report saved to {report_file}")


def main():
    """
    Main function to run advanced training from command line.
    
    Usage:
        # For CSV files:
        python advanced_training.py --features_file features.csv --optimize --n_features 250
        
        # For PKL files:
        python advanced_training.py --features_file features.pkl --optimize --n_features 250
    """
    parser = argparse.ArgumentParser(
        description='Advanced Skin Lesion Classification Training for 98%+ Accuracy'
    )
    
    # Data arguments
    parser.add_argument('--features_file', type=str, required=True,
                       help='Path to features file (CSV or PKL)')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column (default: label)')
    
    # Training arguments
    parser.add_argument('--optimize', action='store_true',
                       help='Enable hyperparameter optimization (takes longer but better results)')
    parser.add_argument('--no_ensemble', action='store_true',
                       help='Skip ensemble models (faster but may be less accurate)')
    parser.add_argument('--n_features', type=int, default=150,
                       help='Number of features to select (default: 150)')
    parser.add_argument('--resampling', type=str, default='smote_tomek',
                       choices=['smote', 'adasyn', 'smote_tomek', 'smote_enn', 'borderline', 'none'],
                       help='Resampling strategy for class imbalance (default: smote_tomek)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/advanced_training',
                       help='Directory for saving outputs (default: output/advanced_training)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" " * 15 + "🎯 ADVANCED SKIN LESION CLASSIFICATION TRAINING 🎯")
    print(" " * 20 + "Target: 98%+ Accuracy with Ensemble Methods")
    print("="*80 + "\n")
    
# Load data
    print(f"📂 Loading features from: {args.features_file}")
    try:
        # Detect file type and load accordingly
        if args.features_file.endswith('.pkl') or args.features_file.endswith('.pickle'):
            print("   File type: PKL (Pickle)")
            data = pd.read_pickle(args.features_file)
            
            # Handle dictionary PKL files
            if isinstance(data, dict):
                print("   Detected dictionary format, extracting DataFrame...")
                # Try common keys
                if 'features' in data:
                    df = data['features']
                elif 'data' in data:
                    df = data['data']
                elif 'X' in data:
                    df = data['X']
                else:
                    # Use first DataFrame found
                    for key, value in data.items():
                        if isinstance(value, pd.DataFrame):
                            df = value
                            print(f"   Using key: '{key}'")
                            break
            else:
                df = data
                
        elif args.features_file.endswith('.csv'):
            print("   File type: CSV")
            df = pd.read_csv(args.features_file)
        else:
            print("❌ Unsupported file format. Use .csv or .pkl files")
            return
        
    except Exception as e:
        print(f"❌ Failed to load data: {str(e)}")
        return
        
        print(f"✅ Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")
    
    # Check if label column exists
    if args.label_column not in df.columns:
        print(f"❌ Label column '{args.label_column}' not found!")
        print(f"Available columns: {', '.join(df.columns[:10])}...")
        return
    
    # Separate features and labels (keep as DataFrame)
    X_df = df.drop(args.label_column, axis=1)
    y = df[args.label_column].values
    feature_names = X_df.columns.tolist()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Features: {X_df.shape[1]}")
    print(f"   Samples: {X_df.shape[0]}")
    print(f"   Class 0 (SK): {np.sum(y == 0)}")
    print(f"   Class 1 (BCC): {np.sum(y == 1)}")
    print(f"   Imbalance ratio: {max(np.sum(y == 0), np.sum(y == 1)) / min(np.sum(y == 0), np.sum(y == 1)):.2f}:1")
    
    # Initialize trainer
    print(f"\n🚀 Initializing Advanced Trainer...")
    trainer = AdvancedSkinLesionTrainer(
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    # Print configuration
    print(f"\n⚙️ Training Configuration:")
    print(f"   Hyperparameter Optimization: {'✅ Enabled' if args.optimize else '❌ Disabled'}")
    print(f"   Ensemble Models: {'✅ Enabled' if not args.no_ensemble else '❌ Disabled'}")
    print(f"   Feature Selection: Top {args.n_features} features")
    print(f"   Resampling Strategy: {args.resampling}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Feature Names Preserved: ✅ Yes (DataFrame mode)")
    
    # Train complete pipeline
    print(f"\n🎓 Starting Training Pipeline...\n")
    
    results = trainer.train_complete_pipeline(
        X=X_df,  # Pass DataFrame directly
        y=y,
        feature_names=feature_names,
        optimize=args.optimize,
        use_ensemble=not args.no_ensemble,
        resampling_strategy=args.resampling,
        n_features=args.n_features
    )
    
    # Print final summary
    print("\n" + "="*80)
    print(" " * 25 + "🏆 TRAINING COMPLETE! 🏆")
    print("="*80)
    print(f"\n✨ Best Model: {results['best_model_name']}")
    print(f"\n📈 Performance Metrics:")
    
    best_results = results['results'][results['best_model_name']]
    print(f"   Accuracy:    {best_results['accuracy']*100:.2f}%")
    print(f"   Precision:   {best_results['precision']*100:.2f}%")
    print(f"   Recall:      {best_results['recall']*100:.2f}%")
    print(f"   F1 Score:    {best_results['f1']*100:.2f}%")
    print(f"   Specificity: {best_results['specificity']*100:.2f}%")
    if 'roc_auc' in best_results:
        print(f"   ROC AUC:     {best_results['roc_auc']*100:.2f}%")
    
    print(f"\n💾 Outputs saved to: {args.output_dir}/")
    print(f"   Models:  {args.output_dir}/models/")
    print(f"   Plots:   {args.output_dir}/plots/")
    print(f"   Reports: {args.output_dir}/reports/")
    
    print("\n" + "="*80)
    print("✅ All done! Check the reports directory for detailed results.")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Run main training pipeline
    main()