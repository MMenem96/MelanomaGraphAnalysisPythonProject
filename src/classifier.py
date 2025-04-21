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
        # Always keep feature_selector as None to ensure we use all features
        self.feature_selector = None
        self.logger.info("Initializing classifier without feature selection to ensure dimension consistency")
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
            
            # Check if all feature vectors have the same length
            feature_lengths = [len(fv) for fv in feature_vectors]
            if len(set(feature_lengths)) > 1:
                # Inconsistent feature lengths detected
                max_len = max(feature_lengths)
                self.logger.warning(f"Inconsistent feature vector lengths detected. Padding to length {max_len}")
                
                # Pad shorter feature vectors with zeros
                for i, fv in enumerate(feature_vectors):
                    if len(fv) < max_len:
                        feature_vectors[i] = np.pad(fv, (0, max_len - len(fv)), 'constant')
            
            try:
                # Convert to numpy array with explicit dtype
                features_array = np.array(feature_vectors, dtype=np.float64)
                
                # Check for and handle NaN values
                if np.isnan(features_array).any():
                    self.logger.warning(f"NaN values detected in feature array. Replacing with zeros.")
                    # Replace NaN with zeros
                    features_array = np.nan_to_num(features_array, nan=0.0)
                
                return features_array
            except ValueError as e:
                # If still failing, try a more robust conversion
                self.logger.warning(f"Standard conversion failed: {str(e)}. Trying alternate method.")
                # Create empty array and fill it
                result = np.zeros((len(feature_vectors), len(feature_vectors[0])), dtype=np.float64)
                for i, fv in enumerate(feature_vectors):
                    # Convert any potential NaN values to zeros
                    fv_array = np.array(fv, dtype=np.float64)
                    fv_array = np.nan_to_num(fv_array, nan=0.0)
                    result[i, :] = fv_array
                return result
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise

    def _get_statistical_features(self, feature_dict):
        """Calculate statistical measures from dictionary of node features."""
        try:
            values = list(feature_dict.values())
            if not values:
                return np.zeros(5)
                
            # Handle inf values by replacing with large finite number
            values = [v if v != float('inf') else 1e10 for v in values]
            
            # Calculate five statistical measures as specified in paper
            mean = np.mean(values)
            std = np.std(values)
            minimum = np.min(values)
            maximum = np.max(values)
            median = np.median(values)
            
            return np.array([mean, std, minimum, maximum, median])
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical features: {str(e)}")
            return np.zeros(5)

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
                self.logger.warning(f"No hyperparameter grid defined for {self.classifier_type}")
                return self.classifier
                
            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1'
            }
            
            # Create and fit grid search
            grid_search = GridSearchCV(
                self.classifier,
                param_grid,
                cv=cv,
                scoring=scoring,
                refit='f1',  # Optimize for F1 score
                n_jobs=-1,   # Use all available cores
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Update classifier with best parameters
            self.classifier = grid_search.best_estimator_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best F1 score: {grid_search.best_score_:.3f}")
            
            return self.classifier
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return self.classifier

    def train_evaluate(self, X, y, cv=5):
        """Train and evaluate the model using cross-validation."""
        try:
            # Count samples in each class to determine appropriate CV
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Log total number of features
            self.logger.info(f"Training with {X.shape[1]} features")
            
            # Adjust CV to be at most the minimum class count
            if min_class_count < cv:
                self.logger.warning(f"Reducing CV folds from {cv} to {min_class_count} due to limited samples in smallest class")
                cv = min_class_count
            
            # Ensure at least 2 folds
            if cv < 2:
                cv = 2
                self.logger.warning(f"Using minimum 2-fold CV due to very limited samples")
                
            # Check for NaN values in input features before scaling
            if np.isnan(X).any():
                self.logger.warning("NaN values found in input features. Replacing with zeros.")
                X = np.nan_to_num(X, nan=0.0)
                
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Double-check for NaN values after scaling (some scalers can introduce NaNs)
            if np.isnan(X_scaled).any():
                self.logger.warning("NaN values found after scaling. Replacing with zeros.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
            # Define scoring metrics
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score)
            }
            
            # Perform cross-validation
            cv_results = cross_validate(
                self.classifier,
                X_scaled,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True
            )
            
            # Train final model on all data
            self.classifier.fit(X_scaled, y)
            
            # Return aggregated results
            results = {
                'accuracy': np.mean(cv_results['test_accuracy']),
                'precision': np.mean(cv_results['test_precision']),
                'recall': np.mean(cv_results['test_recall']),
                'f1': np.mean(cv_results['test_f1']),
                'std_accuracy': np.std(cv_results['test_accuracy']),
                'std_precision': np.std(cv_results['test_precision']),
                'std_recall': np.std(cv_results['test_recall']),
                'std_f1': np.std(cv_results['test_f1']),
                'estimators': cv_results['estimator']
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in train_evaluate: {str(e)}")
            raise

    def evaluate_model(self, X, y):
        """Evaluate the model on a test set and generate detailed report."""
        try:
            # Create evaluator
            evaluator = ModelEvaluator(output_dir='output')
            
            # Log number of features in evaluation
            self.logger.info(f"Evaluating with {X.shape[1]} features")
            
            # Check for NaN values in input features before scaling
            if np.isnan(X).any():
                self.logger.warning("NaN values found in evaluation features. Replacing with zeros.")
                X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Double-check for NaN values after scaling
            if np.isnan(X_scaled).any():
                self.logger.warning("NaN values found after scaling in evaluation. Replacing with zeros.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
            # We're no longer using feature selection to ensure consistency
            # The feature_selector will be None for new models
            if self.feature_selector is not None:
                # Just log a warning but don't apply the transformation
                self.logger.warning("Feature selector exists but will not be used to ensure dimension consistency")
            
            # Perform comprehensive evaluation
            results = evaluator.evaluate_classifier(self.classifier, X_scaled, y)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def save_model(self, output_dir='model'):
        """Save trained model and preprocessing components."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(output_dir, 'melanoma_classifier.joblib')
            dump(self.classifier, model_path)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            dump(self.scaler, scaler_path)
            
            # Save feature selector if available
            if self.feature_selector is not None:
                selector_path = os.path.join(output_dir, 'feature_selector.joblib')
                dump(self.feature_selector, selector_path)
            
            self.logger.info(f"Model saved to {model_path}")
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'feature_selector_path': os.path.join(output_dir, 'feature_selector.joblib') if self.feature_selector else None
            }
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path, scaler_path, feature_selector_path=None):
        """Load trained model and preprocessing components."""
        try:
            # Load model
            self.classifier = load(model_path)
            
            # Load scaler
            self.scaler = load(scaler_path)
            
            # Load feature selector if path provided
            if feature_selector_path and os.path.exists(feature_selector_path):
                self.feature_selector = load(feature_selector_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise