
import numpy as np
from .linear_regression import LinearRegression

class PolynomialRegression:
    """
    Polynomial Regression model.

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial.
    lr : float, default=0.01
        Learning rate for gradient descent.
    n_iters : int, default=1000
        Number of iterations for gradient descent.
    """

    def __init__(self, degree=2, lr=0.01, n_iters=1000):
        self.degree = degree
        self.lr = lr
        self.n_iters = n_iters
        self.regressor = LinearRegression(lr=self.lr, n_iters=self.n_iters)

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        X_poly = self._transform(X)
        self.regressor.fit(X_poly, y)

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        array-like, shape (n_samples,)
            Predicted target values.
        """
        X_poly = self._transform(X)
        return self.regressor.predict(X_poly)

    def _transform(self, X):
        """
        Transform the data to include polynomial features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        array-like, shape (n_samples, n_features * degree)
            Transformed data.
        """
        X_poly = np.power(X, np.arange(1, self.degree + 1))
        return X_poly.reshape(X.shape[0], -1)
