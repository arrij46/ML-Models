# unsupervised_learning/from_scratch/hierarchical_clustering.py
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class HierarchicalClustering:
    """
    Hierarchical Clustering implementation from scratch
    
    Theory:
    Hierarchical clustering builds a tree of clusters by either:
    1. Agglomerative (bottom-up): Start with each point as a cluster and merge
    2. Divisive (top-down): Start with one cluster and split recursively
    
    This implementation focuses on agglomerative clustering with different linkage methods:
    - Single linkage: Minimum distance between clusters
    - Complete linkage: Maximum distance between clusters
    - Average linkage: Average distance between clusters
    - Ward's method: Minimizes variance when merging clusters
    """
    
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.distances_ = None
        
    def _calculate_distance(self, cluster1, cluster2, distance_matrix):
        """Calculate distance between two clusters based on linkage method"""
        if self.linkage == 'single':
            return np.min(distance_matrix[cluster1][:, cluster2])
        elif self.linkage == 'complete':
            return np.max(distance_matrix[cluster1][:, cluster2])
        elif self.linkage == 'average':
            return np.mean(distance_matrix[cluster1][:, cluster2])
        elif self.linkage == 'ward':
            # Ward's method implementation would be more complex
            # This is a simplified version
            return np.mean(distance_matrix[cluster1][:, cluster2])
        else:
            raise ValueError("Linkage method not supported")
    
    def fit(self, X):
        """
        Fit the hierarchical clustering model to the data
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        
        # Calculate pairwise distances
        self.distances_ = squareform(pdist(X))
        
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Store cluster history for dendrogram
        self.cluster_history = []
        
        # Continue until we have the desired number of clusters
        while len(clusters) > self.n_clusters:
            min_distance = float('inf')
            merge_indices = (0, 1)
            
            # Find the two closest clusters
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    distance = self._calculate_distance(clusters[i], clusters[j], self.distances_)
                    if distance < min_distance:
                        min_distance = distance
                        merge_indices = (i, j)
            
            # Merge the two closest clusters
            i, j = merge_indices
            new_cluster = clusters[i] + clusters[j]
            
            # Record the merge
            self.cluster_history.append({
                'merged_clusters': (i, j),
                'new_cluster': new_cluster,
                'distance': min_distance
            })
            
            # Remove the old clusters and add the new one
            clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)]
            clusters.append(new_cluster)
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_idx
                
        return self
    
    def plot_dendrogram(self):
        """Plot a dendrogram of the clustering process"""
        # This is a simplified dendrogram implementation
        # For a full implementation, consider using scipy's dendrogram function
        
        plt.figure(figsize=(10, 7))
        for i, merge in enumerate(self.cluster_history):
            plt.plot([i, i], [0, merge['distance']], 'b-')
        
        plt.title('Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()