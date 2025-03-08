import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import logging
import os
from joblib import dump

class MelanomaClassifier:
    def __init__(self, classifier_type='svm'):
        """Initialize the classifier with specified type."""
        self.logger = logging.getLogger(__name__)
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()

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
                # Extract all features
                features = G.graph['features']

                # Combine all features into a single vector as per paper specifications
                feature_vector = np.concatenate([
                    self._get_statistical_features(features['clustering_coefficient']),
                    self._get_statistical_features(features['nodal_strength']),
                    self._get_statistical_features(features['betweenness_centrality']),
                    self._get_statistical_features(features['closeness_centrality']),
                    [features['local_efficiency']],
                    [features['char_path_length']],
                    [features['global_efficiency']],
                    [features['global_clustering']],
                    [features['density']],
                    [features['assortativity']],
                    [features['spectral_radius']],
                    [features['energy']],
                    [features['spectral_gap']],
                    [features['normalized_laplacian_energy']],
                    features['gft_coefficients']  # First k GFT coefficients
                ])

                feature_vectors.append(feature_vector)

            return np.array(feature_vectors)

        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise

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

            # Define scoring metrics as per paper
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score)
            }

            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

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

            return results

        except Exception as e:
            self.logger.error(f"Error in training and evaluation: {str(e)}")
            raise

    def _get_statistical_features(self, feature_dict):
        """Extract statistical features from a dictionary of node values."""
        values = np.array(list(feature_dict.values()))
        return np.array([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values)
        ])