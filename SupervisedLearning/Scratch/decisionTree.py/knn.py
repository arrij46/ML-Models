# unsupervised_learning/from_scratch/kmeans.py
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means clustering implementation from scratch
    
    Theory:
    K-Means is an unsupervised learning algorithm that partitions data into K clusters.
    It minimizes the within-cluster sum of squares (WCSS):
    WCSS = ΣΣ||xᵢ - μⱼ||²
    
    The algorithm works by:
    1. Randomly initializing K cluster centroids
    2. Assigning each data point to the nearest centroid
    3. Updating centroids as the mean of assigned points
    4. Repeating steps 2-3 until convergence
    """
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        """Initialize centroids using random selection"""
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids
    
    def _assign_clusters(self, X):
        """Assign each data point to the nearest centroid"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids as the mean of assigned points"""
        new_centroids = np.array([X[labels == i].mean(axis=0) 
                                 for i in range(self.n_clusters)])
        return new_centroids
    
    def fit(self, X):
        """
        Fit the K-Means model to the data
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterate until convergence or max iterations
        for _ in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        self.labels = self._assign_clusters(X)
        
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
        Cluster labels of shape (n_samples,)
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def inertia_(self, X):
        """
        Calculate within-cluster sum of squares
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
        WCSS value
        """
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[i])**2)
        return wcss