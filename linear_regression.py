
import numpy as np

class LinearRegression:
    """
    Linear Regression model.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient descent.
    n_iters : int, default=1000
        Number of iterations for gradient descent.
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

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
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

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
        return np.dot(X, self.weights) + self.bias
