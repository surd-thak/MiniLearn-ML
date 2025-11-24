
import numpy as np
from utils.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative, softmax

class TwoLayerNN:
    """
    Two-layer Neural Network classifier.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output classes.
    lr : float, default=0.01
        Learning rate for gradient descent.
    n_iters : int, default=1000
        Number of iterations for gradient descent.
    activation : str, default='relu'
        Activation function for the hidden layer. Can be 'relu', 'sigmoid', or 'tanh'.
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.01, n_iters=1000, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.n_iters = n_iters
        self.activation_str = activation

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        # Set activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Invalid activation function")

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
        n_samples = X.shape[0]

        for i in range(self.n_iters):
            # Forward pass
            self.Z1 = np.dot(X, self.W1) + self.b1
            self.A1 = self.activation(self.Z1)
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = softmax(self.Z2)

            # Backpropagation
            dZ2 = self.A2 - self._one_hot(y)
            dW2 = (1 / n_samples) * np.dot(self.A1.T, dZ2)
            db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.activation_derivative(self.Z1)
            dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
            db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)

            # Update weights and biases
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if i % 100 == 0:
                loss = self._cross_entropy_loss(y, self.A2)
                print(f"Iteration {i}, Loss: {loss}")

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
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = softmax(Z2)
        return np.argmax(A2, axis=1)

    def _one_hot(self, y):
        """
        One-hot encode the target values.
        """
        one_hot_y = np.zeros((y.size, self.output_size))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y

    def _cross_entropy_loss(self, y, y_pred):
        """
        Compute the cross-entropy loss.
        """
        n_samples = y.shape[0]
        log_likelihood = -np.log(y_pred[range(n_samples), y])
        loss = np.sum(log_likelihood) / n_samples
        return loss
