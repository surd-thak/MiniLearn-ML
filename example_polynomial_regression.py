
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from linear_model.polynomial_regression import PolynomialRegression

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate non-linear data
np.random.seed(0)
_X = 2 - 3 * np.random.normal(0, 1, 20)
y = _X - 2 * (_X ** 2) + 0.5 * (_X ** 3) + np.random.normal(-3, 3, 20)

# Reshape X to be a 2D array
X = _X[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

regressor = PolynomialRegression(degree=3, lr=0.001, n_iters=10000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

# Plot the results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
y_grid = regressor.predict(X_grid)

plt.scatter(X, y, color="red")
plt.plot(X_grid, y_grid, color="blue")
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

