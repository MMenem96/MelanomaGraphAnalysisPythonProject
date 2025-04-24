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
import functools

# Define global feature selector functions that can be pickled safely
# These need to be at the module level (outside any class) to be picklable

def global_feature_score_func(X, y, selected_indices=None):
    """Global function for feature scoring that can be pickled."""
    if selected_indices is None:
        return np.ones(X.shape[1])
    
    scores = np.zeros(X.shape[1])
    scores[selected_indices] = 1.0
    return scores

# Create a SelectKBest subclass that can be safely pickled
class PicklableSelectKBest(SelectKBest):
    """A version of SelectKBest that can be safely pickled."""
    
    def __init__(self, k=10, selected_indices=None, n_features=None):
        # We use f_classif as a placeholder, but we'll override it
        super().__init__(f_classif, k=k)
        self.selected_indices = selected_indices
        self.n_features = n_features
        
        # Initialize scores_ if we have both selected_indices and n_features
        if selected_indices is not None and n_features is not None:
            self._init_scores(n_features)
    
    def _init_scores(self, n_features):
        """Initialize the scores_ attribute with zeros and ones for selected indices."""
        self.scores_ = np.zeros(n_features)
        if hasattr(self, 'selected_indices') and self.selected_indices is not None:
            # Set scores to 1.0 for selected features
            for idx in self.selected_indices:
                if idx < n_features:
                    self.scores_[idx] = 1.0
    
    def fit(self, X, y=None):
        """Fit the SelectKBest model."""
        if not hasattr(self, 'scores_') or self.scores_ is None:
            # If scores_ not initialized, do it now
            self._init_scores(X.shape[1])
        return super().fit(X, y)
        
    def _get_support_mask(self):
        """Return a boolean mask where selected features are True."""
        if hasattr(self, 'selected_indices') and self.selected_indices is not None:
            if not hasattr(self, 'scores_') or self.scores_ is None:
                # If we don't have scores_ yet, we need to create it with the right shape
                n_features = max(self.selected_indices) + 1 if len(self.selected_indices) > 0 else 1
                self._init_scores(n_features)
            
            # Create the mask directly from selected_indices
            mask = np.zeros(self.scores_.shape, dtype=bool)
            for idx in self.selected_indices:
                if idx < len(mask):
                    mask[idx] = True
            return mask
        
        return super()._get_support_mask()

class BCCSKClassifier:
    def __init__(self, classifier_type='svm'):
        """
        Initialize the classifier with optimized parameters for BCC vs SK detection.
        
        Args:
            classifier_type: Type of classifier to use ('svm_rbf', 'svm_sigmoid', 'svm_poly', 'rf', 'knn', 'mlp')
        """
        self.logger = logging.getLogger(__name__)
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.logger.info(f"Initializing optimized {classifier_type} classifier for BCC vs SK detection")
        
        # Initialize classifier with optimized parameters for BCC vs SK detection
        if classifier_type == 'svm' or classifier_type == 'svm_rbf':
            # RBF kernel is generally the most effective for complex, non-linear separations
            # like those in skin lesion classification
            self.classifier = SVC(
                kernel='rbf',
                C=1.0,                # More conservative C for better generalization
                gamma='auto',         # Automatically determine gamma for better adaptability
                probability=True,     # Required for ROC curve and probability estimates
                random_state=42,
                class_weight='balanced',  # Critical for handling class imbalance
                cache_size=1000,      # Larger cache for faster computation
                max_iter=10000        # Ensure convergence
            )
        elif classifier_type == 'svm_sigmoid':
            # Sigmoid kernel can sometimes capture more complex relationships
            self.classifier = SVC(
                kernel='sigmoid',
                C=1.0,
                gamma='auto',
                coef0=0.1,            # Sigmoid-specific parameter, optimized for skin imagery
                probability=True,
                random_state=42,
                class_weight='balanced',
                cache_size=1000,
                max_iter=10000
            )
        elif classifier_type == 'svm_poly':
            # Polynomial kernel can model more complex boundaries
            self.classifier = SVC(
                kernel='poly',
                degree=3,             # Cubic polynomial usually works well for medical imaging
                C=1.0,
                gamma='auto',
                coef0=0.1,
                probability=True,
                random_state=42,
                class_weight='balanced',
                cache_size=1000,
                max_iter=10000
            )
        elif classifier_type == 'rf':
            # Random Forests often perform well on high-dimensional medical image data
            self.classifier = RandomForestClassifier(
                n_estimators=300,          # More trees for better ensemble learning
                max_depth=None,            # No limit on depth for complex medical patterns
                min_samples_split=5,       # More conservative to prevent overfitting 
                min_samples_leaf=2,        # More conservative to prevent overfitting
                max_features='sqrt',       # Standard practice for reducing overfitting
                class_weight='balanced',   # Critical for imbalanced skin lesion datasets
                bootstrap=True,            # Enable bootstrapping for robust ensembles
                oob_score=True,            # Use out-of-bag samples for validation
                random_state=42,
                n_jobs=-1                  # Use all available cores
            )
        elif classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            # KNN works well for cases where local patterns matter
            # Default parameters that will be adjusted dynamically during training
            self.classifier = KNeighborsClassifier(
                n_neighbors=5,             # Start with a more conservative value
                weights='distance',        # Weight by inverse of distance for better local focus
                algorithm='auto',          # Let sklearn choose the most efficient algorithm
                leaf_size=30,              # Default value
                p=2,                       # Euclidean distance (L2 norm)
                metric='minkowski',        # Standard distance metric
                n_jobs=-1                  # Use all available cores
            )
            # We'll adjust n_neighbors later based on dataset size
        elif classifier_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            # Neural networks can capture complex, non-linear relationships in image data
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),  # More complex architecture for medical imaging
                activation='relu',                  # ReLU generally works well
                solver='adam',                      # Adam optimizer for better convergence
                alpha=0.001,                        # Slightly higher regularization to prevent overfitting
                batch_size='auto',
                learning_rate='adaptive',           # Adapt learning rate for faster convergence
                learning_rate_init=0.001,           # Standard starting rate
                max_iter=1000,                      # More iterations to ensure convergence
                tol=1e-4,                           # Standard tolerance
                early_stopping=True,                # Stop if validation score doesn't improve
                validation_fraction=0.2,            # Use 20% for validation
                beta_1=0.9, beta_2=0.999,           # Adam optimizer parameters
                epsilon=1e-8,                       # Adam optimizer parameters
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
                
                # Extract new specialized dermoscopic features - highly discriminative for BCC vs SK
                dermoscopic_features = G.graph.get('dermoscopic_features', {})
                
                # Extract local features aggregated with statistical measures
                # Each local feature contributes 5 statistical metrics
                local_feature_stats = []
                
                # 1. Clustering coefficient (crucial for detecting border irregularities)
                if 'clustering_coefficient' in graph_features:
                    local_feature_stats.append(self._get_statistical_features(graph_features['clustering_coefficient']))
                
                # 2. Nodal strength (connectivity pattern)
                if 'nodal_strength' in graph_features:
                    local_feature_stats.append(self._get_statistical_features(graph_features['nodal_strength']))
                
                # 3. Betweenness centrality (identifies bridge nodes)
                if 'betweenness_centrality' in graph_features:
                    local_feature_stats.append(self._get_statistical_features(graph_features['betweenness_centrality']))
                
                # 4. Closeness centrality (density variations)
                if 'closeness_centrality' in graph_features:
                    local_feature_stats.append(self._get_statistical_features(graph_features['closeness_centrality']))
                
                # 5. Eigenvector centrality (node importance)
                if 'eigenvector_centrality' in graph_features:
                    local_feature_stats.append(self._get_statistical_features(graph_features['eigenvector_centrality']))
                
                # If any local features exist, concatenate them
                if local_feature_stats:
                    local_features = np.concatenate(local_feature_stats)
                else:
                    # No local features found, use empty array
                    local_features = np.array([])
                
                # Extract global features as scalars
                global_features = np.array([
                    # 1. Global efficiency (information flow metric)
                    graph_features.get('global_efficiency', 0.0),
                    # 2. Network density (connection density)
                    graph_features.get('density', 0.0),
                    # 3. Global clustering (organization structure)
                    graph_features.get('global_clustering', 0.0),
                    # 4. Modularity (community structure)
                    graph_features.get('modularity', 0.0)
                ])
                
                # Extract spectral features 
                spectral_features = np.array([
                    # 1. Spectral gap (network connectivity strength)
                    graph_features.get('spectral_gap', 0.0),
                    # 2. Energy (overall energy in network structure)
                    graph_features.get('energy', 0.0),
                    # 3. Normalized Laplacian Energy (structural irregularity)
                    graph_features.get('normalized_laplacian_energy', 0.0),
                    # 4. Spectral mean (average eigenvalue)
                    graph_features.get('spectral_mean', 0.0),
                    # 5. Spectral standard deviation (eigenvalue spread)
                    graph_features.get('spectral_std', 0.0),
                    # 6. Spectral skewness (eigenvalue distribution asymmetry)
                    graph_features.get('spectral_skewness', 0.0),
                    # 7. Spectral kurtosis (eigenvalue distribution peakedness)
                    graph_features.get('spectral_kurtosis', 0.0)
                ])
                
                # Extract GFT coefficients (now 8 coefficients instead of 20)
                gft_coeffs = graph_features.get('gft_coefficients', np.zeros(8))
                
                # Combine all feature types into a single vector
                feature_components = []
                if len(local_features) > 0:
                    feature_components.append(local_features)
                feature_components.append(global_features)
                feature_components.append(spectral_features)
                feature_components.append(gft_coeffs)
                
                # Concatenate all features
                graph_feature_vector = np.concatenate(feature_components)
                
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
                
                # Convert dermoscopic features to vector (highly important for BCC vs SK classification)
                if dermoscopic_features:
                    # Sort by key to ensure consistent ordering
                    dermo_feature_keys = sorted(dermoscopic_features.keys())
                    dermo_feature_vector = []
                    
                    for k in dermo_feature_keys:
                        value = dermoscopic_features[k]
                        # Handle any list features
                        if isinstance(value, list):
                            dermo_feature_vector.extend(value)
                        else:
                            dermo_feature_vector.append(value)
                            
                    dermo_feature_vector = np.array(dermo_feature_vector)
                    
                    # Log dermoscopic feature information for transparency
                    self.logger.debug(f"Added {len(dermo_feature_vector)} dermoscopic features: {dermo_feature_keys}")
                else:
                    dermo_feature_vector = np.array([])
                
                # Combine all three feature types
                feature_components = [graph_feature_vector]
                if len(conv_feature_vector) > 0:
                    feature_components.append(conv_feature_vector)
                if len(dermo_feature_vector) > 0:
                    feature_components.append(dermo_feature_vector)
                    
                feature_vector = np.concatenate(feature_components)
                    
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
        """
        Select the most relevant features for classification using advanced techniques.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Feature selection method ('mutual_info', 'f_classif', or 'combined')
            n_features: Maximum number of features to select
            
        Returns:
            X_selected: Matrix with selected features
        """
        try:
            original_n_features = X.shape[1]
            
            # First, we need to check and remove constant features that cause warnings
            # Find columns where all values are the same (constant features)
            variance = np.var(X, axis=0)
            non_constant_features = variance > 1e-10  # Small threshold to account for numerical precision
            
            if not all(non_constant_features):
                constant_indices = np.where(~non_constant_features)[0]
                self.logger.warning(f"Found {len(constant_indices)} constant features at indices {constant_indices}")
                self.logger.info("Removing constant features to avoid warnings")
                
                # Only keep non-constant features
                X = X[:, non_constant_features]
                
                # If we've removed all features, return with warning
                if X.shape[1] == 0:
                    self.logger.warning("All features are constant! Cannot perform feature selection.")
                    return np.zeros((X.shape[0], 0))
            
            # Continue with standard feature selection
            # Ensure we're working with appropriate feature sizes
            max_features = min(n_features, X.shape[1])
            
            # Default to at least 30% of features or 50, whichever is greater
            min_features = max(int(X.shape[1] * 0.3), min(50, X.shape[1]))
            k = max(min_features, max_features)
            
            # For small datasets, use a higher percentage of features
            if X.shape[0] < 50:  # Less than 50 samples
                k = min(int(X.shape[1] * 0.7), X.shape[1])
                self.logger.info(f"Small dataset detected. Using {k} features.")
            
            # Ensure k is not larger than the number of features
            k = min(k, X.shape[1])
            
            # Check if all classes are the same
            if len(np.unique(y)) == 1:
                self.logger.warning("Only one class detected. Cannot perform standard feature selection.")
                # Return a custom feature selector that selects all features
                selected_indices = np.arange(X.shape[1])
                selector = PicklableSelectKBest(k=len(selected_indices), 
                                               selected_indices=selected_indices,
                                               n_features=X.shape[1])
                selector.fit(X, y)
                self.feature_selector = selector
                return X
            
            # Now proceed with the chosen feature selection method
            if method == 'mutual_info':
                # Using mutual information for feature selection (best for detecting non-linear relationships)
                # Wrap in try/except to handle potential errors
                try:
                    selector = SelectKBest(mutual_info_classif, k=k)
                except Exception as e:
                    self.logger.warning(f"Error using mutual_info: {str(e)}. Falling back to all features.")
                    selected_indices = np.arange(X.shape[1])
                    selector = PicklableSelectKBest(k=len(selected_indices), 
                                                   selected_indices=selected_indices,
                                                   n_features=X.shape[1])
            elif method == 'f_classif':
                # Using ANOVA F-statistic for feature selection (best for linear relationships)
                try:
                    selector = SelectKBest(f_classif, k=k)
                except Exception as e:
                    self.logger.warning(f"Error using f_classif: {str(e)}. Falling back to all features.")
                    selected_indices = np.arange(X.shape[1])
                    selector = PicklableSelectKBest(k=len(selected_indices), 
                                                   selected_indices=selected_indices,
                                                   n_features=X.shape[1])
            elif method == 'combined':
                # Use both methods and take the union of their selected features
                try:
                    # First with mutual information
                    mi_k = min(k//2 + 1, X.shape[1])
                    mi_selector = SelectKBest(mutual_info_classif, k=mi_k)
                    mi_selector.fit(X, y)
                    mi_indices = np.where(mi_selector.get_support())[0]
                except Exception as e:
                    self.logger.warning(f"Error in mutual_info part: {str(e)}. Using random subset.")
                    mi_k = min(k//2 + 1, X.shape[1])
                    mi_indices = np.random.choice(X.shape[1], size=mi_k, replace=False)
                
                try:
                    # Then with F-statistic
                    f_k = min(k//2 + 1, X.shape[1])
                    f_selector = SelectKBest(f_classif, k=f_k)
                    f_selector.fit(X, y)
                    f_indices = np.where(f_selector.get_support())[0]
                except Exception as e:
                    self.logger.warning(f"Error in f_classif part: {str(e)}. Using random subset.")
                    f_k = min(k//2 + 1, X.shape[1])
                    f_indices = np.random.choice(X.shape[1], size=f_k, replace=False)
                
                # Combine unique indices
                combined_indices = np.union1d(mi_indices, f_indices)
                
                # Use our custom picklable selector class
                selector = PicklableSelectKBest(k=len(combined_indices), 
                                               selected_indices=combined_indices,
                                               n_features=X.shape[1])
                
                # Fit the selector to initialize it properly
                selector.fit(X, y)
                
                # Get the selected features
                X_selected = X[:, combined_indices]
                self.feature_selector = selector
                
                self.logger.info(f"Combined feature selection: {len(combined_indices)} features")
                
                # If we originally removed constant features, we need to map back to original indices
                if not all(non_constant_features):
                    # Map the selected indices back to the original feature space
                    original_indices = np.where(non_constant_features)[0][combined_indices]
                    self.logger.info(f"Mapped back to original indices, considering removed constant features")
                    
                    # Create a new selector with the original indices
                    self.feature_selector = PicklableSelectKBest(k=len(original_indices), 
                                                               selected_indices=original_indices,
                                                               n_features=original_n_features)
                
                return X_selected
            else:
                self.logger.warning(f"Unknown feature selection method: {method}, using mutual_info")
                try:
                    selector = SelectKBest(mutual_info_classif, k=k)
                except Exception as e:
                    self.logger.warning(f"Error using mutual_info: {str(e)}. Falling back to all features.")
                    selected_indices = np.arange(X.shape[1])
                    selector = PicklableSelectKBest(k=len(selected_indices), 
                                                   selected_indices=selected_indices,
                                                   n_features=X.shape[1])
            
            # Fit the selector and transform the data with error handling
            try:
                X_selected = selector.fit_transform(X, y)
                self.feature_selector = selector
            except Exception as e:
                self.logger.warning(f"Error in feature selection fitting: {str(e)}. Using all features.")
                # Fall back to using all features
                selected_indices = np.arange(X.shape[1])
                selector = PicklableSelectKBest(k=len(selected_indices), 
                                              selected_indices=selected_indices,
                                              n_features=X.shape[1])
                selector.fit(X, y)
                self.feature_selector = selector
                X_selected = X
            
            # Log feature importance information for better understanding
            if hasattr(selector, 'scores_') and selector.scores_ is not None:
                try:
                    # Get indices of selected features
                    selected_indices = np.where(selector.get_support())[0]
                    
                    # Get scores of selected features (handling NaNs)
                    selected_scores = selector.scores_[selected_indices]
                    # Replace NaNs with 0s for safe sorting
                    selected_scores = np.nan_to_num(selected_scores)
                    
                    # Log top 5 features (or less if fewer are available)
                    n_to_show = min(5, len(selected_indices))
                    if n_to_show > 0:
                        top_indices = np.argsort(selected_scores)[-n_to_show:][::-1]  # Descending order
                        top_features = selected_indices[top_indices]
                        top_scores = selected_scores[top_indices]
                        
                        feature_info = '\n'.join([f"Feature {idx}: Score {score:.4f}" 
                                                for idx, score in zip(top_features, top_scores)])
                        self.logger.info(f"Top {n_to_show} selected features:\n{feature_info}")
                except Exception as e:
                    self.logger.warning(f"Error logging feature importance: {str(e)}")
            
            # If we originally removed constant features, we need to map back to original indices
            if not all(non_constant_features) and hasattr(selector, 'get_support'):
                # Get the indices that were selected from the non-constant feature set
                current_selected = np.where(selector.get_support())[0]
                
                # Map these back to the original feature space
                original_indices = np.where(non_constant_features)[0][current_selected]
                self.logger.info(f"Mapped back to original indices, considering removed constant features")
                
                # Create a new selector with the original indices
                self.feature_selector = PicklableSelectKBest(k=len(original_indices), 
                                                           selected_indices=original_indices,
                                                           n_features=original_n_features)
            
            self.logger.info(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
            return X_selected
                
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            # Return original features if selection fails
            return X

    def optimize_hyperparameters(self, X, y, cv=5):
        """Perform hyperparameter optimization."""
        try:
            # Adjust hyperparameter grids based on dataset size
            total_samples = len(y)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Log dataset characteristics for hyperparameter context
            self.logger.info(f"Optimizing hyperparameters for dataset with {total_samples} samples")
            self.logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            if self.classifier_type == 'svm' or self.classifier_type == 'svm_rbf':
                # For larger datasets, we can use a more focused grid
                if total_samples > 5000:
                    param_grid = {
                        'C': [1, 10, 100],
                        'gamma': ['scale', 'auto'],
                        'kernel': ['rbf']
                    }
                else:
                    param_grid = {
                        'C': [0.1, 1, 10, 100],
                        'gamma': ['scale', 'auto', 0.01, 0.1],
                        'kernel': ['rbf']
                    }
            elif self.classifier_type == 'svm_sigmoid':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['sigmoid'],
                    'coef0': [0.0, 0.1, 0.5]
                }
            elif self.classifier_type == 'svm_poly':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['poly'],
                    'degree': [2, 3, 4],
                    'coef0': [0.0, 0.1, 0.5]
                }
            elif self.classifier_type == 'rf':
                # For very large datasets, use more focused parameters
                if total_samples > 5000:
                    param_grid = {
                        'n_estimators': [200, 300],
                        'max_depth': [None, 30, 50],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                else:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
            elif self.classifier_type == 'knn':
                # Dynamically adjust n_neighbors based on dataset size
                if total_samples >= 1000:
                    # For large datasets, higher k values work better
                    n_neighbors_choices = [5, 7, 9, 11]
                elif total_samples >= 100:
                    # For medium datasets
                    n_neighbors_choices = [3, 5, 7]
                else:
                    # For small datasets, use limited neighbors
                    # Ensure we don't exceed class size
                    max_safe_k = max(3, min(5, min_class_count // 2))
                    n_neighbors_choices = list(range(1, max_safe_k + 1))
                
                param_grid = {
                    'n_neighbors': n_neighbors_choices,
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1=Manhattan, 2=Euclidean
                }
            elif self.classifier_type == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
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
            total_samples = len(y)
            
            # Log total number of features and samples
            self.logger.info(f"Training with {X.shape[1]} features on {total_samples} samples")
            self.logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            # For KNN classifier: Adjust n_neighbors based on dataset size
            # to avoid the "Expected n_neighbors <= n_samples_fit" error
            if self.classifier_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                
                # Calculate safe n_neighbors value (at most 1/5 of smallest class, minimum 3)
                safe_neighbors = max(3, min(5, min_class_count // 5))
                
                if total_samples >= 1000:
                    # For large datasets, we can use higher k value
                    k_neighbors = 7
                elif total_samples >= 100:
                    # For medium datasets
                    k_neighbors = 5
                else:
                    # For small datasets, use the safe value
                    k_neighbors = safe_neighbors
                
                self.logger.info(f"Adjusting KNN n_neighbors to {k_neighbors} based on dataset size")
                self.classifier.set_params(n_neighbors=k_neighbors)
            
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
            
            # For very small datasets with potential class imbalance issues:
            if total_samples < 30:
                self.logger.warning("Small dataset detected. Using stratified KFold with shuffle to ensure class balance")
                # Use stratified k-fold with shuffle to ensure each fold has at least one sample of each class
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                cv_splitter = cv
                
            # Add error handling for cross-validation
            try:
                # Perform cross-validation with error handling
                cv_results = cross_validate(
                    self.classifier,
                    X_scaled,
                    y,
                    cv=cv_splitter,
                    scoring=scoring,
                    return_train_score=True,
                    return_estimator=True,
                    error_score='raise'  # Raise errors to catch them specifically
                )
            except ValueError as e:
                if "The number of classes has to be greater than one" in str(e):
                    self.logger.warning("Cross-validation failed due to single-class folds. Reducing CV folds and retrying.")
                    # If we get a single-class error, reduce CV to minimum and use highest shuffle seed
                    cv = 2
                    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=999)
                    
                    cv_results = cross_validate(
                        self.classifier,
                        X_scaled,
                        y, 
                        cv=cv_splitter,
                        scoring=scoring,
                        return_train_score=True,
                        return_estimator=True,
                        error_score=0  # Return 0 for any failed folds
                    )
                else:
                    # Re-raise other errors
                    raise
            
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
            
            # Create classifier-specific directory
            classifier_dir = self.classifier_type.upper()
            
            # Map internal classifier types to proper directory names for consistency
            if self.classifier_type == 'svm':
                classifier_dir = 'SVM_RBF'
            elif self.classifier_type == 'svm_sigmoid':
                classifier_dir = 'SVM_Sigmoid'
            elif self.classifier_type == 'svm_poly':
                classifier_dir = 'SVM_Poly'
            elif self.classifier_type == 'rf':
                classifier_dir = 'RF'
            elif self.classifier_type == 'knn':
                classifier_dir = 'KNN'
            elif self.classifier_type == 'mlp':
                classifier_dir = 'MLP'
                
            # Create classifier-specific directory
            model_dir = os.path.join(output_dir, classifier_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, 'model.joblib')
            dump(self.classifier, model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            dump(self.scaler, scaler_path)
            
            # Save feature selector using our custom picklable class
            if self.feature_selector is not None:
                try:
                    # Save the feature selector
                    selector_path = os.path.join(model_dir, 'feature_selector.joblib')
                    
                    # For maximum compatibility, check if it's our custom class
                    if isinstance(self.feature_selector, PicklableSelectKBest):
                        # If it's our custom class, it should pickle fine
                        dump(self.feature_selector, selector_path)
                        self.logger.info(f"Saved pickable feature selector to {selector_path}")
                    else:
                        # If it's the standard selector, extract the support mask
                        # and create a new PicklableSelectKBest
                        if hasattr(self.feature_selector, 'get_support'):
                            # Get the indices that were selected
                            support = self.feature_selector.get_support()
                            selected_indices = np.where(support)[0]
                            
                            # Create a simplified version that will pickle safely
                            # Determine the n_features from the scores_ if available
                            n_features = None
                            if hasattr(self.feature_selector, 'scores_'):
                                n_features = len(self.feature_selector.scores_)
                            
                            picklable_selector = PicklableSelectKBest(
                                k=len(selected_indices),
                                selected_indices=selected_indices,
                                n_features=n_features
                            )
                            
                            # Save this simplified selector
                            dump(picklable_selector, selector_path)
                            self.logger.info(f"Saved converted feature selector to {selector_path}")
                        else:
                            self.logger.warning("Feature selector missing expected attributes, not saving")
                            
                except Exception as e:
                    self.logger.warning(f"Could not save feature selector: {str(e)}. Will continue without it.")
            
            self.logger.info(f"Model saved to {model_path}")
            
            # Also save a copy in the main model directory for backward compatibility
            backup_model_path = os.path.join(output_dir, 'bcc_sk_classifier.joblib')
            backup_scaler_path = os.path.join(output_dir, 'scaler.joblib')
            dump(self.classifier, backup_model_path)
            dump(self.scaler, backup_scaler_path)
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'feature_selector_path': os.path.join(model_dir, 'feature_selector.joblib') if self.feature_selector else None
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