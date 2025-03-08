import networkx as nx
import numpy as np
from scipy import linalg
import logging
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        """Initialize FeatureExtractor with logging."""
        self.logger = logging.getLogger(__name__)

    def extract_local_features(self, G):
        """Extract local graph measures as specified in the paper."""
        features = {}

        try:
            # Local efficiency (LE) as defined in the paper
            features['local_efficiency'] = nx.local_efficiency(G)

            # Local clustering coefficient (LCC) with weight normalization
            clustering = nx.clustering(G, weight='weight')
            features['clustering_coefficient'] = {k: v if not np.isnan(v) else 0.0 
                                                  for k, v in clustering.items()}

            # Nodal strength (NS) with minimum value constraint
            features['nodal_strength'] = {k: max(v, 1e-10) 
                                           for k, v in dict(G.degree(weight='weight')).items()}

            # Nodal betweenness centrality (NBC) with weighted paths
            features['betweenness_centrality'] = nx.betweenness_centrality(G, 
                                                                             weight=lambda u, v, d: 1.0/max(d['weight'], 1e-10))

            # Closeness centrality (CC) with weighted paths
            features['closeness_centrality'] = nx.closeness_centrality(G, 
                                                                        distance=lambda u, v, d: 1.0/max(d['weight'], 1e-10))

            # Eccentricity (Ecc) - maximum distance to any other node
            if nx.is_connected(G):
                features['eccentricity'] = nx.eccentricity(G)
            else:
                features['eccentricity'] = {node: float('inf') for node in G.nodes()}

        except Exception as e:
            self.logger.error(f"Error in local feature extraction: {str(e)}")
            features = self._get_default_local_features(G)

        return features

    def extract_global_features(self, G):
        """Extract global graph measures as specified in the paper."""
        features = {}

        try:
            # Characteristic path length with proper handling of disconnected graphs
            if nx.is_connected(G):
                features['char_path_length'] = nx.average_shortest_path_length(G, weight='weight')
            else:
                features['char_path_length'] = float('inf')

            # Global efficiency as defined in paper
            features['global_efficiency'] = nx.global_efficiency(G)

            # Global clustering coefficient with weight consideration
            try:
                features['global_clustering'] = nx.average_clustering(G, weight='weight')
            except ZeroDivisionError:
                features['global_clustering'] = 0.0

            # Graph density
            features['density'] = nx.density(G)

            # Assortativity coefficient with weight consideration
            try:
                features['assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
            except (ValueError, ZeroDivisionError):
                features['assortativity'] = 0.0

        except Exception as e:
            self.logger.error(f"Error in global feature extraction: {str(e)}")
            features = self._get_default_global_features()

        return features

    def extract_spectral_features(self, G):
        """Extract spectral features using graph Fourier transform as per paper specifications."""
        try:
            # Get normalized Laplacian matrix as specified in paper
            L = nx.normalized_laplacian_matrix(G).toarray()

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = linalg.eigh(L)

            # Sort by eigenvalues (ascending order as per paper)
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Compute spectral features as defined in the paper
            features = {
                'spectral_radius': float(np.max(np.abs(eigenvals))),
                'energy': float(np.sum(np.abs(eigenvals))),
                'spectral_gap': float(eigenvals[1] - eigenvals[0]) if len(eigenvals) > 1 else 0.0,
                'normalized_laplacian_energy': float(np.sum(np.abs(eigenvals - 2)))
            }

            # Add graph Fourier transform features with proper normalization
            features['gft_coefficients'] = self._compute_gft_features(G, eigenvecs)

        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {str(e)}")
            features = self._get_default_spectral_features()

        return features

    def _compute_gft_features(self, G, eigenvecs):
        """Compute GFT features for node signals as specified in paper."""
        try:
            # Get node features as signals (using LAB color values as primary signal)
            node_signals = []
            for n in G.nodes():
                if 'features' in G.nodes[n]:
                    # Extract LAB color values (first 3 features)
                    signal = G.nodes[n]['features'][:3]
                    node_signals.append(signal)
                else:
                    node_signals.append(np.zeros(3))

            node_signals = np.array(node_signals)

            # Apply GFT using eigenvectors (as per paper equations)
            gft_coeffs = np.abs(np.dot(node_signals.T, eigenvecs))

            # Take first k coefficients as features (k=10 as specified in paper)
            k = min(10, gft_coeffs.shape[1])
            return gft_coeffs[:, :k].flatten()

        except Exception as e:
            self.logger.error(f"Error computing GFT features: {str(e)}")
            return np.zeros(10)  # Default size for GFT features

    def _get_default_local_features(self, G):
        """Return default values for local features."""
        return {
            'local_efficiency': 0.0,
            'clustering_coefficient': {node: 0.0 for node in G.nodes()},
            'nodal_strength': {node: 1.0 for node in G.nodes()},
            'betweenness_centrality': {node: 0.0 for node in G.nodes()},
            'closeness_centrality': {node: 0.0 for node in G.nodes()},
            'eccentricity': {node: float('inf') for node in G.nodes()}
        }

    def _get_default_global_features(self):
        """Return default values for global features."""
        return {
            'char_path_length': float('inf'),
            'global_efficiency': 0.0,
            'global_clustering': 0.0,
            'density': 0.0,
            'assortativity': 0.0
        }

    def _get_default_spectral_features(self):
        """Return default values for spectral features."""
        return {
            'spectral_radius': 0.0,
            'energy': 0.0,
            'spectral_gap': 0.0,
            'normalized_laplacian_energy': 0.0,
            'gft_coefficients': np.zeros(10)
        }

    def create_feature_matrix(self, graphs, labels=None):
        """Create a labeled feature matrix from graph features.

        Args:
            graphs: List of networkx graphs with extracted features
            labels: Optional array of labels (0 for benign, 1 for melanoma)

        Returns:
            pandas DataFrame with labeled features and diagnosis
        """
        try:
            feature_vectors = []
            feature_names = []

            # Process first graph to get feature names
            if graphs:
                first_graph = graphs[0]
                features = first_graph.graph['features']

                # Local feature names with statistics
                local_features = ['clustering_coefficient', 'nodal_strength', 
                                'betweenness_centrality', 'closeness_centrality']
                stats = ['mean', 'std', 'min', 'max']

                for feat in local_features:
                    for stat in stats:
                        feature_names.append(f"{feat}_{stat}")

                # Global feature names
                global_features = ['local_efficiency', 'char_path_length', 
                                 'global_efficiency', 'global_clustering',
                                 'density', 'assortativity']
                feature_names.extend(global_features)

                # Spectral feature names
                spectral_features = ['spectral_radius', 'energy', 'spectral_gap',
                                   'normalized_laplacian_energy']
                feature_names.extend(spectral_features)

                # GFT coefficient names
                n_gft = len(features['gft_coefficients'])
                gft_names = [f'gft_coef_{i+1}' for i in range(n_gft)]
                feature_names.extend(gft_names)

            # Extract features from all graphs
            for G in graphs:
                features = G.graph['features']
                feature_vector = []

                # Add local features with statistics
                for feat in local_features:
                    values = np.array(list(features[feat].values()))
                    feature_vector.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ])

                # Add global features
                for feat in global_features:
                    feature_vector.append(features[feat])

                # Add spectral features
                for feat in spectral_features:
                    feature_vector.append(features[feat])

                # Add GFT coefficients
                feature_vector.extend(features['gft_coefficients'])

                feature_vectors.append(feature_vector)

            # Create DataFrame
            df = pd.DataFrame(feature_vectors, columns=feature_names)

            # Add diagnosis column if labels are provided
            if labels is not None:
                df['diagnosis'] = ['melanoma' if label == 1 else 'benign' 
                                 for label in labels]

            return df

        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {str(e)}")
            raise RuntimeError(f"Error creating feature matrix: {str(e)}")