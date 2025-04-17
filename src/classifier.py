import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
import logging
import os
from joblib import dump, load
from src.evaluation import ModelEvaluator  # Import ModelEvaluator
class MelanomaClassifier:
    def __init__(self, classifier_type='svm'):
        """Initialize the classifier with specified type."""
        self.logger = logging.getLogger(__name__)
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.feature_selector = None
        # Initialize classifier based on type with parameters from paper
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                C=10.0,  # Higher C value for better regularization as per paper
                gamma='scale',  # Using scale for better generalization
                probability=True,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=200,  # Increased number of trees as per paper
                max_depth=None,  # Allow full depth for complex feature relationships
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    def prepare_features(self, graphs):
        """Convert graph features to a feature matrix."""
        try:
            if not graphs:
                return np.array([]).reshape(0, 1)  # Return empty 2D array
            feature_vectors = []
            for G in graphs:
                # Extract all graph-based features
                graph_features = G.graph['features']
                
                # Extract all conventional features
                conventional_features = G.graph.get('conventional_features', {})
                # Combine all graph-based features into a vector
                graph_feature_vector = np.concatenate([
                    self._get_statistical_features(graph_features['clustering_coefficient']),
                    self._get_statistical_features(graph_features['nodal_strength']),
                    self._get_statistical_features(graph_features['betweenness_centrality']),
                    self._get_statistical_features(graph_features['closeness_centrality']),
                    [graph_features.get('local_efficiency', 0.0)],
                    [graph_features.get('char_path_length', float('inf')) if graph_features.get('char_path_length', float('inf')) != float('inf') else 0.0],
                    [graph_features.get('global_efficiency', 0.0)],
                    [graph_features.get('global_clustering', 0.0)],
                    [graph_features.get('density', 0.0)],
                    [graph_features.get('assortativity', 0.0)],
                    [graph_features.get('transitivity', 0.0)],
                    [graph_features.get('rich_club_coefficient', 0.0)],
                    [graph_features.get('spectral_radius', 0.0)],
                    [graph_features.get('energy', 0.0)],
                    [graph_features.get('spectral_gap', 0.0)],
                    [graph_features.get('normalized_laplacian_energy', 0.0)],
                    [graph_features.get('spectral_power', 0.0)],
                    [graph_features.get('spectral_entropy', 0.0)],
                    [graph_features.get('spectral_amplitude', 0.0)],
                    graph_features.get('gft_coefficients', np.zeros(20))  # GFT coefficients
                ])
                
                # Convert conventional features dictionary to vector
                if conventional_features:
                    # Sort by key to ensure consistent ordering
                    conv_feature_keys = sorted(conventional_features.keys())
                    conv_feature_vector = []
                    
                    for k in conv_feature_keys:
                        value = conventional_features[k]
                        # Handle Hu moments which are lists
                        if isinstance(value, list):
                            conv_feature_vector.extend(value)
                        else:
                            conv_feature_vector.append(value)
                            
                    conv_feature_vector = np.array(conv_feature_vector)
                else:
                    conv_feature_vector = np.array([])
                
                # Combine graph-based and conventional features
                if len(conv_feature_vector) > 0:
                    feature_vector = np.concatenate([graph_feature_vector, conv_feature_vector])
                else:
                    feature_vector = graph_feature_vector
                feature_vectors.append(feature_vector)
            return np.array(feature_vectors)
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
    def select_features(self, X, y, method='mutual_info', n_features=100):
        """Select the most relevant features."""
        try:
            if method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
            elif method == 'f_classif':
                selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
            else:
                self.logger.warning(f"Unknown feature selection method: {method}, using mutual_info")
                selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
                
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            
            self.logger.info(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
            
            return X_selected
                
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return X
    def optimize_hyperparameters(self, X, y, cv=5):
        """Perform hyperparameter optimization."""
        try:
            if self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            elif self.classifier_type == 'rf':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                return self.classifier
                
            grid_search = GridSearchCV(
                self.classifier, param_grid, cv=cv, 
                scoring='f1', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.classifier = grid_search.best_estimator_
            
            return self.classifier
                
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return self.classifier
    def train_evaluate(self, X, y):
        """Train and evaluate the classifier using stratified k-fold cross-validation."""
        try:
            if len(X) == 0:
                raise ValueError("No training data provided")
            # Check minimum required samples per class
            min_required_samples = 3  # Minimum required samples per class for reliable CV
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = np.min(class_counts)
            if min_samples < min_required_samples:
                raise ValueError(
                    f"Need at least {min_required_samples} samples per class for reliable "
                    f"training. Got {min_samples} for some classes."
                )
            # Determine number of folds based on dataset size
            n_folds = min(5, min_samples)  # Use 5-fold CV when possible
            self.logger.info(f"Using {n_folds}-fold cross-validation")
            # Apply feature selection if we have enough features
            if X.shape[1] > 100:  # Only select features if we have many
                self.logger.info(f"Applying feature selection to reduce from {X.shape[1]} features")
                X = self.select_features(X, y, method='mutual_info', n_features=100)
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Optimize hyperparameters if dataset is large enough
            if len(X) >= 20:
                self.logger.info("Optimizing hyperparameters...")
                self.optimize_hyperparameters(X_scaled, y, cv=n_folds)
            # Define scoring metrics as per paper
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score)
            }
            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            # Perform cross-validation
            cv_results = cross_validate(
                self.classifier, X_scaled, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )
            # Calculate and format results
            results = {
                'accuracy': np.mean(cv_results['test_accuracy']),
                'precision': np.mean(cv_results['test_precision']),
                'recall': np.mean(cv_results['test_recall']),
                'f1': np.mean(cv_results['test_f1']),
                'std_accuracy': np.std(cv_results['test_accuracy']),
                'std_precision': np.std(cv_results['test_precision']),
                'std_recall': np.std(cv_results['test_recall']),
                'std_f1': np.std(cv_results['test_f1'])
            }
            # Train final model on all data
            self.classifier.fit(X_scaled, y)
            # Save the model and scaler
            os.makedirs('model', exist_ok=True)
            dump(self.classifier, 'model/melanoma_classifier.joblib')
            dump(self.scaler, 'model/scaler.joblib')
            
            # Save feature selector if used
            if self.feature_selector:
                dump(self.feature_selector, 'model/feature_selector.joblib')
            # Perform comprehensive evaluation if dataset is sufficient
            if len(X) >= 10:  # Only perform comprehensive evaluation with enough samples
                self.logger.info("Performing comprehensive model evaluation...")
                evaluation_results = self.evaluate_model(X_scaled, y, cv=n_folds)
                results['evaluation'] = evaluation_results
            return results
        except Exception as e:
            self.logger.error(f"Error in training and evaluation: {str(e)}")
            raise
    def evaluate_model(self, X, y, cv=5):
        """Perform comprehensive evaluation of the trained model."""
        try:
            self.logger.info("Starting model evaluation...")
            
            # Make sure model and scaler are available
            if not hasattr(self, 'classifier') or not hasattr(self, 'scaler'):
                raise ValueError("Model and scaler must be initialized before evaluation.")
                
            # Initialize evaluator
            evaluator = ModelEvaluator()
            
            # Perform evaluation
            evaluation_results = evaluator.evaluate_classifier(self.classifier, X, y, cv=cv)
            
            # Log summary results
            cv_summary = evaluation_results['cv_results']['summary']
            self.logger.info("Cross-validation summary results:")
            for metric, values in cv_summary.items():
                self.logger.info(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    def load_model(self, model_path, scaler_path, feature_selector_path=None):
        """Load a trained model and scaler."""
        try:
            self.classifier = load(model_path)
            self.scaler = load(scaler_path)
            
            if feature_selector_path and os.path.exists(feature_selector_path):
                self.feature_selector = load(feature_selector_path)
                
            self.logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    def _get_statistical_features(self, feature_dict):
        """Extract statistical features from a dictionary of node values."""
        values = np.array(list(feature_dict.values()))
        return np.array([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values)
        ])