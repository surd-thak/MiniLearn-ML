
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int, default=3
        Number of neighbors to use.
    """

    def __init__(self, k=3):
        self.k = k

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
        self.X_train = X
        self.y_train = y

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
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            A single sample.

        Returns
        -------
        int
            The predicted class label.
        """
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
