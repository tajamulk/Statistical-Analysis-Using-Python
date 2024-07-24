import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
# Size (independent variable) and Price (dependent variable)
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000]
}
df = pd.DataFrame(data)

# Features and target variable
X = df[['Size']]  # Independent variable
y = df['Price']   # Dependent variable

# Split data into training and testing sets (optional)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluation metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
n = len(y)
p = X.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'R-squared: {r2}')
print(f'Adjusted R-squared: {adjusted_r2}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Size (square feet)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: House Price vs Size')
plt.legend()
plt.show()
