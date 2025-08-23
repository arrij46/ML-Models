# unsupervised_learning/with_libraries/kmeans.py
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_

# Evaluate clustering performance
inertia = kmeans.inertia_
silhouette = silhouette_score(X, y_pred)

print(f"Inertia (WCSS): {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()