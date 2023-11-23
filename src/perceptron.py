# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np
import matplotlib.pyplot as plt

from .utils.activation import Activation


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Perceptron:
    """
    A simple implementation of a Perceptron.

    Parameters
    ----------
    n_iter : int, optional
        The number of iterations to train the Perceptron for, by default 100.
    learning_rate : float, optional
        The learning rate for the Perceptron, by default 0.1.
    activation : str, optional
        The activation function to use for the Perceptron, by default 'sigmoid'.
        Can take 'sigmoid', 'tanh' or 'relu' values

    Attributes
    ----------
    W : np.ndarray
        The weights for the Perceptron.
    b : np.ndarray
        The bias for the Perceptron.

    Methods
    -------
    fit(X: np.ndarray, y: np.ndarray) -> None:
        Trains the Perceptron on the given data.
    predict(X: np.ndarray) -> np.ndarray:
        Predicts the output for the given input using the trained Perceptron.
    """
    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1, activation: str = 'sigmoid'):
        """
        Initializes a new instance of the Perceptron class.

        Parameters
        ----------
        n_iter : int, optional
            The number of iterations to train the Perceptron for, by default 100.
        learning_rate : float, optional
            The learning rate for the Perceptron, by default 0.1.
        activation : str, optional
            The activation function to use for the Perceptron, by default 'sigmoid'.
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        try :
            self.activation = getattr(Activation, activation)
        except :
            raise ValueError(f'Unknown activation function: {activation}')

    def __initialize_weights(self, input_size: int) -> None:
        """
        Initializes the weights for the Perceptron.

        Parameters
        ----------
        input_size : int
            The number of features in the input data.
        """
        self.W = np.random.randn(input_size, 1)
        self.b = np.random.randn(1)

    def __compute_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Computes the gradients for weights and bias.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target values.

        Returns
        -------
        tuple
            Tuple containing gradients for weights and bias.
        """
        Z = X.dot(self.W) + self.b
        A = self.activation(Z)
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dW, db

    def __update_weights(self, dW: np.ndarray, db: np.ndarray) -> None:
        """
        Updates the weights and bias using the computed gradients.

        Parameters
        ----------
        dW : np.ndarray
            Gradients for weights.
        db : np.ndarray
            Gradient for bias.
        """
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray, record_loss: bool = False) -> None:
        """
        Trains the Perceptron on the given data.

        Parameters
        ----------
        X : np.ndarray
            The input data to train the Perceptron on.
        y : np.ndarray
            The target values for the input data.
        """
        self.__initialize_weights(X.shape[1])
        if record_loss:
            self.losses = []

        for _ in range(self.n_iter):
            dW, db = self.__compute_gradients(X, y)
            self.__update_weights(dW, db)

            if record_loss:
                
                A = self.activation(X.dot(self.W) + self.b)
                loss = np.mean((A - y) ** 2)
                self.losses.append(loss)
                

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input using the trained Perceptron.

        Parameters
        ----------
        X : np.ndarray
            The input data to predict the output for.

        Returns
        -------
        np.ndarray
            The predicted output for the input data.
        """
        Z = X.dot(self.W) + self.b
        if self.activation == Activation.sigmoid:
            A = self.activation(Z)
            return A >= 0.5
        elif self.activation == Activation.tanh:
            A = self.activation(Z)
            return A >= 0
        elif self.activation == Activation.relu:
            A = self.activation(Z)
            return A >= 0
        else:
            raise ValueError(f'Unknown activation function: {self.activation}')
            
    def plot_decision_boundary(self, X, y, title="Decision Boundaries"):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", label="Data Points")
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, alpha=0.3, levels=[0], linestyles='dashed', colors='blue', label="Decision Boundary")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.show()

    def get_loss_history(self) -> list:
        """
        Returns the recorded loss values during training.

        Returns
        -------
        list
            List of loss values recorded during training.
        """
        if hasattr(self, 'losses'):
            return self.losses
        else:
            print("Loss history not available. Train the model with record_loss=True to capture loss values.")
            return []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    