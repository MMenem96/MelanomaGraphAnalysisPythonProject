import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import logging

class GraphConstructor:
    def __init__(self, connectivity_threshold=0.5):
        """Initialize GraphConstructor with parameters from the paper."""
        self.connectivity_threshold = connectivity_threshold
        self.logger = logging.getLogger(__name__)

    def build_graph(self, features, segments):
        """Construct graph from superpixel features as specified in the paper."""
        try:
            # Validate input dimensions
            if len(features) < 2:
                raise ValueError("Need at least 2 superpixels to construct a graph")

            if not isinstance(features, np.ndarray):
                features = np.array(features)

            # Calculate pairwise distances between superpixels using feature vectors
            distances = squareform(pdist(features))

            # Validate distances computation
            if np.any(np.isnan(distances)):
                raise ValueError("Distance computation resulted in NaN values")

            # Compute Gaussian weights as per equation (1) in the paper
            # Using adaptive sigma based on mean distance
            sigma = np.mean(distances[distances > 0])  # More robust sigma estimation
            if sigma == 0:
                sigma = 1.0  # Fallback value
            weights = np.exp(-distances**2 / (2 * sigma**2))

            # Normalize weights to [0,1] range
            if weights.max() > 0:
                weights = weights / weights.max()
            else:
                raise ValueError("All weights are zero, check feature vectors")

            # Create adjacency matrix based on threshold
            adjacency = weights > self.connectivity_threshold
            np.fill_diagonal(adjacency, 0)  # Remove self-loops

            # Verify graph will be constructible
            if not np.any(adjacency):
                self.logger.warning("No edges meet threshold criteria, lowering threshold")
                # Adaptively lower threshold until we get some edges
                while not np.any(adjacency) and self.connectivity_threshold > 0.1:
                    self.connectivity_threshold *= 0.9
                    adjacency = weights > self.connectivity_threshold
                    np.fill_diagonal(adjacency, 0)

            # Create NetworkX graph
            G = nx.from_numpy_array(adjacency)

            # Add node features
            for i in range(len(features)):
                G.nodes[i]['features'] = features[i]

            # Ensure graph is connected by adding minimum spanning tree edges
            if not nx.is_connected(G):
                self._ensure_connectivity(G, weights)

            # Add edge weights
            G = self.compute_edge_weights(G, features)

            # Validate final graph
            if not nx.is_connected(G):
                raise ValueError("Failed to create a connected graph")

            return G

        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
            raise RuntimeError(f"Error building graph: {str(e)}")

    def _ensure_connectivity(self, G, weights):
        """Ensure graph connectivity by adding necessary edges."""
        try:
            components = list(nx.connected_components(G))
            while len(components) > 1:
                # Find closest pair of nodes between components
                min_dist = float('inf')
                edge_to_add = None

                for comp1 in components:
                    for comp2 in components:
                        if comp1 != comp2:
                            for n1 in comp1:
                                for n2 in comp2:
                                    if weights[n1][n2] < min_dist:
                                        min_dist = weights[n1][n2]
                                        edge_to_add = (n1, n2)

                if edge_to_add:
                    G.add_edge(*edge_to_add)
                else:
                    self.logger.warning("Failed to find connecting edge, adding nearest neighbors")
                    # Fallback: connect nearest neighbors between components
                    n1 = list(components[0])[0]
                    n2 = list(components[1])[0]
                    G.add_edge(n1, n2)

                # Update components
                components = list(nx.connected_components(G))

        except Exception as e:
            self.logger.error(f"Error ensuring graph connectivity: {str(e)}")
            raise RuntimeError(f"Error ensuring graph connectivity: {str(e)}")

    def compute_edge_weights(self, G, features):
        """Compute edge weights based on feature similarity as per paper equations."""
        try:
            for (u, v) in G.edges():
                # Extract feature vectors for nodes u and v
                feat_u = features[u]
                feat_v = features[v]

                # Validate feature vectors
                if np.any(np.isnan(feat_u)) or np.any(np.isnan(feat_v)):
                    raise ValueError(f"NaN values found in features for nodes {u} and {v}")

                # Compute color difference (LAB values, first 3 features)
                color_diff = np.linalg.norm(feat_u[:3] - feat_v[:3])

                # Compute texture difference (std values, next 3 features)
                texture_diff = np.linalg.norm(feat_u[3:6] - feat_v[3:6])

                # Compute spatial difference (position features, next 2 features)
                spatial_diff = np.linalg.norm(feat_u[6:8] - feat_v[6:8])

                # Compute geometric difference (shape features, last 3 features)
                geometric_diff = np.linalg.norm(feat_u[8:] - feat_v[8:])

                # Combine differences using weighted sum as per paper
                total_diff = (0.4 * color_diff + 
                            0.3 * texture_diff + 
                            0.2 * spatial_diff + 
                            0.1 * geometric_diff)

                # Apply Gaussian weighting as per equation (1)
                weight = np.exp(-total_diff**2 / 2) + 1e-10  # Add small epsilon to avoid zero weights

                # Add weight to edge
                G[u][v]['weight'] = weight

            return G

        except Exception as e:
            self.logger.error(f"Error computing edge weights: {str(e)}")
            raise RuntimeError(f"Error computing edge weights: {str(e)}")