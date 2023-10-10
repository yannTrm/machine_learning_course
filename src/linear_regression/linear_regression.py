# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np

from typing import Union, Callable
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LinearRegression:
    def __init__(self):
        self.theta = None
        
    def initialize_theta(self, X: np.ndarray, biais : bool = True):
        if biais:
            self.theta = np.random.randn(X.shape[1] + 1, 1)
        else :
            self.theta = np.random.randn(X.shape[1], 1)

    def add_bias_column(self, X: np.ndarray):
        return np.concatenate((X,np.ones((X.shape[0], 1))), axis = 1)

    def fit(self, X: np.ndarray, y: np.ndarray, algorithm: Callable, *args, **kwargs) -> None:
        """
        Train the linear regression model on the training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target values.
            algorithm (Callable): The regression algorithm function.
            *args: Additional positional arguments for the algorithm function.
            **kwargs: Additional keyword arguments for the algorithm function.

        Returns:
            None
        """
        if self.theta is None:
            self.initialize_theta(X)
        X = self.add_bias_column(X)
        self.theta = algorithm(X, y, self.theta,  *args, **kwargs)

    def predict(self, X: np.ndarray) -> Union[np.ndarray, None]:
        """
        Predict target values for new data.

        Args:
            X (np.ndarray): Feature matrix for predictions.

        Returns:
            Union[np.ndarray, None]: Predicted target values or None if the model is not trained.
        """
        if self.theta is not None:
            y_pred = np.dot(self.add_bias_column(X), self.theta)
            return y_pred
        else:
            raise ValueError("The model must be trained before making predictions.")
