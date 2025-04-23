import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean
import logging

class GraphConstructor:
    def __init__(self, connectivity_threshold=0.5):
        """Initialize graph constructor with connectivity threshold."""
        self.connectivity_threshold = connectivity_threshold
        self.logger = logging.getLogger(__name__)

    def build_graph(self, features, segments):
        """Build a weighted graph from superpixel features."""
        try:
            # Create an empty graph
            G = nx.Graph()
            
            # Number of superpixels
            n_segments = features.shape[0]
            
            # Add nodes to the graph
            for i in range(n_segments):
                G.add_node(i, features=features[i])
            
            # Calculate adjacency between superpixels
            adjacency = self._compute_adjacency(segments)
            
            # Add edges based on feature similarity and adjacency
            for i in range(n_segments):
                for j in range(i+1, n_segments):
                    # Calculate similarity between superpixels
                    similarity = self._compute_similarity(features[i], features[j])
                    
                    # Add edge if similarity is above threshold or if superpixels are adjacent
                    if similarity > self.connectivity_threshold or adjacency[i, j]:
                        G.add_edge(i, j, weight=similarity)
            
            # Ensure the graph is connected
            self._ensure_connected(G)
            
            return G
            
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
            raise

    def _compute_adjacency(self, segments):
        """Compute adjacency matrix for superpixels."""
        try:
            # Number of segments
            n_segments = np.max(segments) + 1
            
            # Initialize adjacency matrix
            adjacency = np.zeros((n_segments, n_segments), dtype=bool)
            
            # Get image dimensions
            height, width = segments.shape
            
            # Check horizontal adjacency
            for y in range(height):
                for x in range(width - 1):
                    s1 = segments[y, x]
                    s2 = segments[y, x + 1]
                    if s1 != s2:
                        adjacency[s1, s2] = True
                        adjacency[s2, s1] = True
            
            # Check vertical adjacency
            for y in range(height - 1):
                for x in range(width):
                    s1 = segments[y, x]
                    s2 = segments[y + 1, x]
                    if s1 != s2:
                        adjacency[s1, s2] = True
                        adjacency[s2, s1] = True
            
            return adjacency
            
        except Exception as e:
            self.logger.error(f"Error computing adjacency: {str(e)}")
            raise

    def _compute_similarity(self, feature1, feature2):
        """Compute similarity between two feature vectors."""
        try:
            # Extract color and position features
            color1, pos1 = feature1[:3], feature1[3:]
            color2, pos2 = feature2[:3], feature2[3:]
            
            # Calculate color similarity (using LAB color space)
            color_dist = euclidean(color1, color2)
            
            # Calculate position similarity
            pos_dist = euclidean(pos1, pos2)
            
            # Normalize distances
            max_color_dist = 100.0  # Approximate max distance in LAB space
            color_similarity = 1.0 - min(color_dist / max_color_dist, 1.0)
            
            # Normalize position distance by image size
            # Assuming positions are normalized to [0, 1]
            max_pos_dist = np.sqrt(2)  # Maximum distance in a normalized 2D space
            pos_similarity = 1.0 - min(pos_dist / max_pos_dist, 1.0)
            
            # Combine similarities (weighted average)
            # Weight color similarity higher than position
            similarity = 0.7 * color_similarity + 0.3 * pos_similarity
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            raise

    def _ensure_connected(self, G):
        """Ensure that the graph is connected."""
        try:
            # Check if the graph is connected
            if not nx.is_connected(G):
                # Get all components
                components = list(nx.connected_components(G))
                
                # Connect each component to the largest one
                largest_component = max(components, key=len)
                other_components = [c for c in components if c != largest_component]
                
                # For each smaller component, add an edge to the largest one
                for component in other_components:
                    # Get a node from each component
                    node1 = next(iter(component))
                    node2 = next(iter(largest_component))
                    
                    # Add a weak edge between them
                    G.add_edge(node1, node2, weight=0.01)
                    
            return G
            
        except Exception as e:
            self.logger.error(f"Error ensuring graph connectivity: {str(e)}")
            raise