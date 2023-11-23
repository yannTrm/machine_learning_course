# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class SVM:
    """
    Support Vector Machine (SVM) classifier.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent. Default is 0.001.
    - lambda_param (float): Regularization parameter. Default is 0.01.
    - n_iters (int): Number of iterations for training. Default is 1000.

    Attributes:
    - lr (float): The learning rate for gradient descent.
    - lambda_param (float): Regularization parameter.
    - n_iters (int): Number of iterations for training.
    - w (numpy.ndarray): Coefficients of the SVM.
    - b (float): Bias term of the SVM.

    Methods:
    - fit(X, y): Train the SVM classifier on the given training data.
    - predict(X): Make predictions on new data.
    - score(X, y): Evaluate the accuracy of the model on a given dataset.
    - plot_svm(X, y): Plot the decision boundary and margins of the SVM.

    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        """
        Initialize the SVM classifier.

        Parameters:
        - learning_rate (float): The learning rate for gradient descent.
        - lambda_param (float): Regularization parameter.
        - n_iters (int): Number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM classifier on the given training data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.
        """
        n_samples, n_features = X.shape

        # Convert labels to -1 for negative class and 1 for positive class
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray):
        """
        Make predictions on new data.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Predicted labels.
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the accuracy of the model on a given dataset.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): True labels.

        Returns:
        - float: Accuracy of the model on the provided dataset.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot_svm(self, X: np.ndarray, y: np.ndarray):
        """
        Plot the decision boundary and margins of the SVM.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): True labels.
        """
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig, ax = plt.subplots()
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--", label="Decision Boundary")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k", label="Margin")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        ax.legend()
        plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------