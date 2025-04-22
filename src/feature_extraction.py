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
            features['spectral_power'] = float(np.mean(np.square(gft_coeffs)))
            
            # Spectral entropy - information content in GFT coefficients
            spectral_entropy = 0.0
            if np.any(gft_coeffs):
                normalized_coeffs = np.abs(gft_coeffs) / np.sum(np.abs(gft_coeffs))
                entropy_terms = normalized_coeffs * np.log2(normalized_coeffs + 1e-10)
                spectral_entropy = -np.sum(entropy_terms)
            features['spectral_entropy'] = float(spectral_entropy)
            
            # Spectral amplitude - average magnitude of GFT coefficients
            features['spectral_amplitude'] = float(np.mean(np.abs(gft_coeffs)))

        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {str(e)}")
            features = self._get_default_spectral_features()

        return features

    def _compute_gft_features(self, G, eigenvecs):
        """Compute graph Fourier transform coefficients for node signals."""
        try:
            # Create a signal on the graph nodes
            # Use degree as the signal as suggested in paper
            degree_dict = dict(G.degree(weight='weight'))
            signal = np.array([degree_dict[i] for i in sorted(G.nodes())])
            
            # Normalize the signal
            if np.max(signal) > np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
            # Compute GFT coefficients
            # GFT is defined as the projection of the signal onto eigenvectors
            gft_coeffs = np.abs(np.dot(signal, eigenvecs))
            
            # Return a fixed-length vector by zero-padding or truncating
            target_length = 20  # As specified in the paper
            if len(gft_coeffs) < target_length:
                return np.pad(gft_coeffs, (0, target_length - len(gft_coeffs)))
            else:
                return gft_coeffs[:target_length]
                
        except Exception as e:
            self.logger.error(f"Error computing GFT features: {str(e)}")
            return np.zeros(20)  # Return zeros if computation fails

    def _get_default_local_features(self, G):
        """Return default values for local features when extraction fails."""
        n_nodes = len(G.nodes())
        return {
            'local_efficiency': 0.0,
            'clustering_coefficient': {node: 0.0 for node in G.nodes()},
            'nodal_strength': {node: 1.0 for node in G.nodes()},
            'betweenness_centrality': {node: 0.0 for node in G.nodes()},
            'closeness_centrality': {node: 0.0 for node in G.nodes()},
            'eccentricity': {node: float('inf') for node in G.nodes()}
        }

    def _get_default_global_features(self):
        """Return default values for global features when extraction fails."""
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
        """Return default values for spectral features when extraction fails."""
        return {
            'spectral_radius': 0.0,
            'energy': 0.0,
            'spectral_gap': 0.0,
            'normalized_laplacian_energy': 0.0,
            'gft_coefficients': np.zeros(20),
            'spectral_power': 0.0,
            'spectral_entropy': 0.0,
            'spectral_amplitude': 0.0
        }