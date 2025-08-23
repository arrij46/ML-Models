# unsupervised_learning/with_libraries/hierarchical_clustering.py
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=50, centers=3, cluster_std=0.60, random_state=0)

# Create and fit hierarchical clustering model
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = model.fit_predict(X)

# Plot the clustering
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Create linkage matrix and plot dendrogram
Z = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()