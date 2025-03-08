import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from skimage.segmentation import mark_boundaries
import networkx as nx
import os
import numpy as np
from scipy.stats import zscore

class Visualizer:
    def __init__(self):
        self.output_dir = 'output'
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_superpixels(self, image, segments):
        """Visualize superpixel segmentation."""
        plt.figure(figsize=(10, 10))
        plt.imshow(mark_boundaries(image, segments))
        plt.axis('off')
        plt.title('Superpixel Segmentation')
        plt.savefig(os.path.join(self.output_dir, 'superpixels.png'), bbox_inches='tight')
        plt.close()

    def plot_graph(self, G, pos=None):
        """Visualize the constructed graph with edge weights and node features."""
        plt.figure(figsize=(12, 12))
        if pos is None:
            pos = nx.spring_layout(G, k=1, iterations=50)

        # Get node features for coloring
        node_colors = []
        for node in G.nodes():
            if 'features' in G.nodes[node]:
                # Use the first feature (typically color) for visualization
                node_colors.append(G.nodes[node]['features'][0])
            else:
                node_colors.append(0)

        # Normalize colors
        if node_colors:
            node_colors = zscore(node_colors)

        # Get edge weights for width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)
        edge_weights = [2 * w/max_weight for w in edge_weights]

        # Draw the graph
        nx.draw(G, pos, 
                node_color=node_colors, 
                node_size=300,
                edge_color='gray',
                width=edge_weights,
                edge_cmap=plt.cm.viridis,
                cmap=plt.cm.viridis,
                with_labels=False)

        plt.title('Superpixel Graph Structure')
        plt.savefig(os.path.join(self.output_dir, 'graph.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_features(self, features, global_features):
        """Visualize extracted features with detailed analysis."""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)

        # Plot clustering coefficient distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'clustering_coefficient' in features:
            values = list(features['clustering_coefficient'].values())
            ax1.hist(values, bins=20, color='skyblue', edgecolor='black')
            ax1.set_title('Clustering Coefficient\nDistribution')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')

        # Plot centrality measures
        ax2 = fig.add_subplot(gs[0, 1])
        if 'betweenness_centrality' in features:
            values = list(features['betweenness_centrality'].values())
            ax2.hist(values, bins=20, color='lightgreen', edgecolor='black')
            ax2.set_title('Betweenness Centrality\nDistribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')

        # Plot nodal strength distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'nodal_strength' in features:
            values = list(features['nodal_strength'].values())
            ax3.hist(values, bins=20, color='salmon', edgecolor='black')
            ax3.set_title('Nodal Strength\nDistribution')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')

        # Plot global features
        ax4 = fig.add_subplot(gs[1, :])
        scalar_features = {k: v for k, v in global_features.items() 
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}

        if scalar_features:
            names = list(scalar_features.keys())
            values = list(scalar_features.values())
            positions = range(len(names))

            # Create bar plot
            bars = ax4.bar(positions, values)

            # Customize bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

            ax4.set_xticks(positions)
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.set_title('Global Graph Features')

            # Add grid for better readability
            ax4.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'features.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_feature_importance(self, classifier, feature_names):
        """Plot feature importance if available from the classifier."""
        plt.figure(figsize=(12, 6))

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            return

        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        # Plot feature importances
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance in Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()