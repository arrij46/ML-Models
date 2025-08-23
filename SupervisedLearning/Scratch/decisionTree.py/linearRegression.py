# supervised_learning/from_scratch/linear_regression.py
import numpy as np

class LinearRegression:
    """
    Linear Regression model implementation from scratch
    
    Theory:
    Linear regression models the relationship between a dependent variable (y) 
    and one or more independent variables (X) using a linear function:
    y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
    
    The parameters β are estimated by minimizing the sum of squared residuals:
    RSS = Σ(yᵢ - ŷᵢ)²
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        Fit the linear regression model using the normal equation
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        """
        # Add intercept term (column of ones)
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Calculate coefficients using the normal equation: β = (XᵀX)⁻¹Xᵀy
        X_transpose = X.T
        coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
        
    def predict(self, X):
        """
        Predict using the linear model
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
        Predictions of shape (n_samples,)
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model must be fitted before prediction")
            
        return self.intercept + np.dot(X, self.coefficients)
    
    def score(self, X, y):
        """
        Calculate the R² score of the model
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        y: True target values of shape (n_samples,)
        
        Returns:
        R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)