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
                # For disconnected graphs, compute average over all connected components
                components = list(nx.connected_components(G))
                if components:
                    path_lengths = []
                    node_weights = []
                    for component in components:
                        sg = G.subgraph(component)
                        if len(sg) > 1:  # Only consider components with at least 2 nodes
                            pl = nx.average_shortest_path_length(sg, weight='weight')
                            path_lengths.append(pl)
                            node_weights.append(len(sg))
                    
                    if path_lengths:
                        # Weighted average by component size
                        features['char_path_length'] = np.average(path_lengths, weights=node_weights)
                    else:
                        features['char_path_length'] = float('inf')
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
                
            # Additional global features from the paper:
            # Transitivity - fraction of all possible triangles present in G
            features['transitivity'] = nx.transitivity(G)
            
            # Rich club coefficient - tendency of high-degree nodes to connect to each other
            # Compute for multiple k values and take average
            rich_club_values = []
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 0
            k_values = range(1, min(6, max_degree + 1))  # Use up to 5 k values
            
            for k in k_values:
                try:
                    rich_k = nx.rich_club_coefficient(G, normalized=False).get(k, 0)
                    rich_club_values.append(rich_k)
                except:
                    rich_club_values.append(0)
                    
            features['rich_club_coefficient'] = np.mean(rich_club_values) if rich_club_values else 0

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
            gft_coeffs = self._compute_gft_features(G, eigenvecs)
            features['gft_coefficients'] = gft_coeffs

            # Add new spectral features from the paper
            # Spectral power - average squared magnitude of GFT coefficients
            features['spectral_power'] = float(np.mean(gft_coeffs**2))
            
            # Spectral entropy - distribution of energy across frequencies
            # Normalize coefficients to get probability-like distribution
            norm_coeffs = np.abs(gft_coeffs) / (np.sum(np.abs(gft_coeffs)) + 1e-10)
            entropy = -np.sum(norm_coeffs * np.log2(norm_coeffs + 1e-10))
            features['spectral_entropy'] = float(entropy)
            
            # Spectral amplitude - maximum coefficient
            features['spectral_amplitude'] = float(np.max(np.abs(gft_coeffs)))

        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {str(e)}")
            features = self._get_default_spectral_features()

        return features

    def _compute_gft_features(self, G, eigenvecs):
        """Compute GFT features for node signals as specified in paper."""
        try:
            # Get node features as signals (using multiple color channels)
            # Initialize signals with 3 channels (Lab color)
            node_signals = np.zeros((len(G.nodes()), 3))
            
            for i, n in enumerate(G.nodes()):
                if 'features' in G.nodes[n]:
                    # Extract LAB color values (first 3 features)
                    node_signals[i] = G.nodes[n]['features'][:3]
            
            # Apply GFT using eigenvectors as per paper equations
            # For each channel, compute the GFT separately
            gft_coeffs = []
            for channel in range(node_signals.shape[1]):
                channel_signal = node_signals[:, channel]
                # Transform signal to frequency domain
                channel_gft = np.abs(np.dot(channel_signal, eigenvecs))
                gft_coeffs.append(channel_gft)
            
            # Combine GFT coefficients from all channels
            combined_gft = np.concatenate(gft_coeffs)
            
            # Take first k coefficients as features (k=20 as specified in paper)
            k = min(20, len(combined_gft))
            return combined_gft[:k].astype(float)

        except Exception as e:
            self.logger.error(f"Error computing GFT features: {str(e)}")
            return np.zeros(20)  # Default size for GFT features

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
            'assortativity': 0.0,
            'transitivity': 0.0,
            'rich_club_coefficient': 0.0
        }

    def _get_default_spectral_features(self):
        """Return default values for spectral features."""
        return {
            'spectral_radius': 0.0,
            'energy': 0.0,
            'spectral_gap': 0.0,
            'normalized_laplacian_energy': 0.0,
            'spectral_power': 0.0,
            'spectral_entropy': 0.0,
            'spectral_amplitude': 0.0,
            'gft_coefficients': np.zeros(20)
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
                graph_features = first_graph.graph['features']
                conventional_features = first_graph.graph.get('conventional_features', {})

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
                                 'density', 'assortativity', 'transitivity',
                                 'rich_club_coefficient']
                feature_names.extend(global_features)

                # Spectral feature names
                spectral_features = ['spectral_radius', 'energy', 'spectral_gap',
                                   'normalized_laplacian_energy', 'spectral_power',
                                   'spectral_entropy', 'spectral_amplitude']
                feature_names.extend(spectral_features)

                # GFT coefficient names
                n_gft = len(graph_features['gft_coefficients'])
                gft_names = [f'gft_coef_{i+1}' for i in range(n_gft)]
                feature_names.extend(gft_names)
                
                # Add conventional feature names if present
                conv_feature_names = []
                if conventional_features:
                    conv_feature_names = sorted(conventional_features.keys())
                    feature_names.extend(conv_feature_names)

            # Extract features from all graphs
            for G in graphs:
                graph_features = G.graph['features']
                conventional_features = G.graph.get('conventional_features', {})
                feature_vector = []

                # Add local features with statistics
                for feat in local_features:
                    values = np.array(list(graph_features[feat].values()))
                    feature_vector.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ])

                # Add global features
                for feat in global_features:
                    feature_vector.append(graph_features.get(feat, 0.0))

                # Add spectral features
                for feat in spectral_features:
                    feature_vector.append(graph_features.get(feat, 0.0))

                # Add GFT coefficients
                feature_vector.extend(graph_features['gft_coefficients'])
                
                # Add conventional features if present
                if conventional_features:
                    for name in conv_feature_names:
                        value = conventional_features.get(name, 0.0)
                        # Handle Hu moments which are lists
                        if isinstance(value, list):
                            feature_vector.extend(value)
                            # Update feature names to reflect multiple values from lists
                            if G == first_graph:  # Only update on first graph
                                # Remove the original list feature name
                                if name in feature_names:
                                    feature_names.remove(name)
                                # Add individual element names
                                for i in range(len(value)):
                                    feature_names.append(f"{name}_{i}")
                        else:
                            feature_vector.append(value)

                feature_vectors.append(feature_vector)

            # Create DataFrame
            # Adjust feature names to match feature vector length
            if feature_vectors:
                if len(feature_names) != len(feature_vectors[0]):
                    # Adjust feature names to match vector length
                    if len(feature_names) < len(feature_vectors[0]):
                        # Add generic names for missing columns
                        for i in range(len(feature_names), len(feature_vectors[0])):
                            feature_names.append(f"feature_{i}")
                    else:
                        # Truncate feature names if too many
                        feature_names = feature_names[:len(feature_vectors[0])]
                
                df = pd.DataFrame(feature_vectors, columns=feature_names)
            else:
                df = pd.DataFrame()

            # Add diagnosis column if labels are provided
            if labels is not None:
                df['diagnosis'] = ['melanoma' if label == 1 else 'benign' 
                                 for label in labels]

            return df

        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {str(e)}")
            raise RuntimeError(f"Error creating feature matrix: {str(e)}")