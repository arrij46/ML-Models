# supervised_learning/with_libraries/linear_regression.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv('../datasets/sample_regression.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, y_pred)
print(f"R² Score: {score:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")