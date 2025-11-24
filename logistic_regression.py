
import numpy as np

class LogisticRegression:
    """
    Logistic Regression model.

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
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict class labels for new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        array-like, shape (n_samples,)
            Predicted class labels.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def _sigmoid(self, x):
        """
        Sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
