# docs/supervised_learning.md

## Linear Regression

### Theory

Linear regression is a supervised learning algorithm that models the relationship between a dependent variable (y) and one or more independent variables (X) using a linear function:

y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

Where:
- y is the dependent variable (target)
- X₁, X₂, ..., Xₙ are the independent variables (features)
- β₀ is the intercept term
- β₁, β₂, ..., βₙ are the coefficients
- ε is the error term

The algorithm estimates the parameters (β) by minimizing the sum of squared residuals (RSS):

RSS = Σ(yᵢ - ŷᵢ)²

This minimization can be solved analytically using the normal equation:

β = (XᵀX)⁻¹Xᵀy

Or numerically using optimization techniques like gradient descent.

### Assumptions

1. Linearity: The relationship between features and target is linear
2. Independence: Observations are independent of each other
3. Homoscedasticity: Constant variance of errors
4. Normality: Errors are normally distributed

### Implementation

We provide two implementations:
1. From scratch using NumPy
2. Using scikit-learn library

### Evaluation Metrics

Common evaluation metrics for regression:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)

### Applications

- Predicting house prices
- Sales forecasting
- Risk assessment in finance
- Temperature prediction

# docs/unsupervised_learning.md

## K-Means Clustering

### Theory

K-Means is an unsupervised learning algorithm that partitions data into K clusters based on feature similarity. The algorithm works as follows:

1. **Initialization**: Randomly select K points as initial cluster centroids
2. **Assignment**: Assign each data point to the nearest centroid
3. **Update**: Recalculate centroids as the mean of all points in the cluster
4. **Repeat**: Steps 2-3 until centroids stabilize or maximum iterations reached

The algorithm minimizes the within-cluster sum of squares (WCSS):

WCSS = ΣΣ||xᵢ - μⱼ||²

Where:
- xᵢ is a data point
- μⱼ is the centroid of cluster j

### Key Parameters

- `n_clusters`: The number of clusters to form
- `max_iter`: Maximum number of iterations
- `tol`: Relative tolerance to declare convergence

### Advantages and Limitations

**Advantages:**
- Simple to implement and understand
- Efficient for large datasets
- Works well with spherical clusters

**Limitations:**
- Requires specifying K in advance
- Sensitive to initial centroid placement
- Assumes spherical clusters of similar size
- Struggles with non-convex clusters

### Evaluation Metrics

- **Inertia (WCSS)**: Lower values indicate better clustering
- **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better)

### Applications

- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection

## DBSCAN

### Theory

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

Key concepts:
- **Core point**: A point that has at least `min_samples` points within distance `eps`
- **Border point**: A point that is reachable from a core point but doesn't have enough neighbors
- **Noise point**: A point that is not reachable from any core point

### Key Parameters

- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point

### Advantages and Limitations

**Advantages:**
- Doesn't require specifying the number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers

**Limitations:**
- Struggles with clusters of varying densities
- Sensitive to parameter settings
- Not suitable for high-dimensional data

### Applications

- Anomaly detection in network traffic
- Geographic data analysis
- Image segmentation
- Customer behavior analysis