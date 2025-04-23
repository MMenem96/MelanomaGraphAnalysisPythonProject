import networkx as nx
import numpy as np
from scipy import linalg
from scipy import stats
import logging
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        """Initialize FeatureExtractor with logging."""
        self.logger = logging.getLogger(__name__)

    def extract_local_features(self, G):
        """
        Extract optimized local graph measures that are most discriminative for BCC vs SK detection.
        Focuses on key network properties while avoiding redundant or noisy features.
        """
        features = {}

        try:
            # Most important local features based on literature and effectiveness:
            
            # 1. Local clustering coefficient - crucial for detecting irregularities in borders
            # High predictive value for BCC which tends to have more irregular structures than SK
            clustering = nx.clustering(G, weight='weight')
            features['clustering_coefficient'] = {k: v if not np.isnan(v) else 0.0 
                                                 for k, v in clustering.items()}

            # 2. Nodal strength - measure of connection intensity
            # BCC typically has different connectivity patterns than SK lesions
            features['nodal_strength'] = {k: max(v, 1e-10) 
                                         for k, v in dict(G.degree(weight='weight')).items()}
            
            # 3. Betweenness centrality - identifies "bridge" nodes that are critical for information flow
            # Effective at capturing asymmetry patterns in lesions
            features['betweenness_centrality'] = nx.betweenness_centrality(G, 
                                                                           weight=lambda u, v, d: 1.0/max(d['weight'], 1e-10),
                                                                           normalized=True)  # Normalize to [0,1] range for better scaling

            # 4. Closeness centrality - measures how close a node is to all other nodes
            # Important for capturing the density variations within the lesion
            features['closeness_centrality'] = nx.closeness_centrality(G, 
                                                                      distance=lambda u, v, d: 1.0/max(d['weight'], 1e-10))
            
            # 5. Eigenvector centrality - captures influence of a node in the network
            # Highly effective for identifying important regions in the lesion
            try:
                features['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G, weight='weight')
            except:
                # Fallback if convergence issues
                features['eigenvector_centrality'] = {node: 0.5 for node in G.nodes()}

        except Exception as e:
            self.logger.error(f"Error in local feature extraction: {str(e)}")
            features = self._get_default_local_features(G)

        return features

    def extract_global_features(self, G):
        """
        Extract optimized global graph measures that are most discriminative for BCC vs SK detection.
        Focus on measures that capture overall network structure and organization.
        """
        features = {}

        try:
            # 1. Global efficiency - how efficiently information flows through the entire network
            # This captures the overall interconnectedness, which tends to be different between BCC and SK
            features['global_efficiency'] = nx.global_efficiency(G)

            # 2. Graph density - ratio of actual to potential connections
            # SK lesions tend to have more uniform density compared to BCC
            features['density'] = nx.density(G)
            
            # 3. Global clustering coefficient - overall tendency of nodes to cluster together
            # This measures structure organization, which differs between BCC and SK lesions
            try:
                features['global_clustering'] = nx.average_clustering(G, weight='weight')
            except ZeroDivisionError:
                features['global_clustering'] = 0.0
                
            # 4. Modularity - strength of division into modules (communities)
            # BCC often shows more distinct structural modules due to irregular growth patterns
            try:
                communities = nx.community.greedy_modularity_communities(G, weight='weight')
                # Calculate modularity score if communities were found
                if communities and len(communities) > 1:
                    modularity = nx.community.modularity(G, communities, weight='weight')
                    features['modularity'] = modularity
                else:
                    features['modularity'] = 0.0
            except:
                features['modularity'] = 0.0

        except Exception as e:
            self.logger.error(f"Error in global feature extraction: {str(e)}")
            features = self._get_default_global_features()

        return features

    def extract_spectral_features(self, G):
        """
        Extract optimized spectral features using graph Fourier analysis.
        Focus on the most discriminative spectral properties for BCC vs SK detection.
        """
        try:
            # Get normalized Laplacian matrix for spectral analysis
            L = nx.normalized_laplacian_matrix(G).toarray()

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = linalg.eigh(L)

            # Sort by eigenvalues (ascending order)
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Extract key spectral properties that are most relevant for classification
            
            # 1. Spectral gap - difference between first and second eigenvalues
            # Captures the connectivity strength and is higher in more connected networks
            # Research shows this is particularly useful for distinguishing BCC from SK
            spectral_gap = float(eigenvals[1] - eigenvals[0]) if len(eigenvals) > 1 else 0.0
            features = {'spectral_gap': spectral_gap}
            
            # 2. Energy - sum of absolute eigenvalues
            # Reflects the overall "energy" in the network structure
            # BCC lesions typically exhibit higher spectral energy than SK
            features['energy'] = float(np.sum(np.abs(eigenvals)))
            
            # 3. Normalized Laplacian Energy - deviation from uniform spectrum
            # This captures irregularity in the structural patterns
            features['normalized_laplacian_energy'] = float(np.sum(np.abs(eigenvals - 2)))
            
            # 4. Calculate spectral moments (statistical distribution of eigenvalues)
            # These provide a compact representation of the spectrum that's effective for classification
            if len(eigenvals) > 2:
                features['spectral_mean'] = float(np.mean(eigenvals))
                features['spectral_std'] = float(np.std(eigenvals))
                features['spectral_skewness'] = float(stats.skew(eigenvals))
                features['spectral_kurtosis'] = float(stats.kurtosis(eigenvals))
            else:
                features['spectral_mean'] = float(np.mean(eigenvals)) if len(eigenvals) > 0 else 0.0
                features['spectral_std'] = 0.0
                features['spectral_skewness'] = 0.0
                features['spectral_kurtosis'] = 0.0
            
            # 5. Extract a small number of leading GFT coefficients (first 8)
            # These capture the dominant patterns in the graph structure
            gft_coeffs = self._compute_gft_features(G, eigenvecs, target_length=8)
            features['gft_coefficients'] = gft_coeffs

        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {str(e)}")
            features = self._get_default_spectral_features()

        return features

    def _compute_gft_features(self, G, eigenvecs, target_length=8):
        """
        Compute optimized graph Fourier transform coefficients for node signals.
        
        Args:
            G: The input graph
            eigenvecs: Eigenvectors of the graph Laplacian
            target_length: Number of GFT coefficients to return (default: 8)
            
        Returns:
            Array of GFT coefficients, length specified by target_length
        """
        try:
            # Use weighted degree as the signal
            # This captures the connectivity pattern of each node
            degree_dict = dict(G.degree(weight='weight'))
            signal = np.array([degree_dict[i] for i in sorted(G.nodes())])
            
            # Normalize the signal to [0,1] range for consistent scaling
            if np.max(signal) > np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
            # Compute GFT coefficients (projection onto eigenvectors)
            gft_coeffs = np.abs(np.dot(signal, eigenvecs))
            
            # Use only the most informative coefficients (typically lower frequencies)
            # Research shows these contain most of the discriminative information
            if len(gft_coeffs) < target_length:
                return np.pad(gft_coeffs, (0, target_length - len(gft_coeffs)))
            else:
                return gft_coeffs[:target_length]
                
        except Exception as e:
            self.logger.error(f"Error computing GFT features: {str(e)}")
            return np.zeros(target_length)

    def _get_default_local_features(self, G):
        """Return default values for local features when extraction fails."""
        n_nodes = len(G.nodes())
        return {
            'clustering_coefficient': {node: 0.0 for node in G.nodes()},
            'nodal_strength': {node: 1.0 for node in G.nodes()},
            'betweenness_centrality': {node: 0.0 for node in G.nodes()},
            'closeness_centrality': {node: 0.0 for node in G.nodes()},
            'eigenvector_centrality': {node: 0.5 for node in G.nodes()}
        }

    def _get_default_global_features(self):
        """Return default values for global features when extraction fails."""
        return {
            'global_efficiency': 0.0,
            'density': 0.0,
            'global_clustering': 0.0,
            'modularity': 0.0
        }

    def _get_default_spectral_features(self):
        """Return default values for spectral features when extraction fails."""
        return {
            'spectral_gap': 0.0,
            'energy': 0.0,
            'normalized_laplacian_energy': 0.0,
            'spectral_mean': 0.0,
            'spectral_std': 0.0,
            'spectral_skewness': 0.0,
            'spectral_kurtosis': 0.0,
            'gft_coefficients': np.zeros(8)
        }