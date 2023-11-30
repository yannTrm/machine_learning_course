# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Classes
class CostFunctions:
    """
    A class containing common cost functions used in machine learning.

    This class provides implementations of various cost functions commonly used in machine learning tasks,
    including regression and other supervised learning problems.

    Methods:
        - mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Mean Squared Error (MSE) between true and predicted values.

        - mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Mean Absolute Error (MAE) between true and predicted values.

        - rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        - mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.
            
        - log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Log Loss between true and predicted values.
    """

    @staticmethod
    def check_shape(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays y_true and y_pred must have the same shape.")


    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true (np.ndarray): NumPy array of true target values.
            y_pred (np.ndarray): NumPy array of predicted target values.

        Returns:
            float: The computed MSE.
        """
        CostFunctions.check_shape(y_true, y_pred)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between true and predicted values.

        Args:
            y_true (np.ndarray): NumPy array of true target values.
            y_pred (np.ndarray): NumPy array of predicted target values.

        Returns:
            float: The computed MAE.
        """
        CostFunctions.check_shape(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        Args:
            y_true (np.ndarray): NumPy array of true target values.
            y_pred (np.ndarray): NumPy array of predicted target values.

        Returns:
            float: The computed RMSE.
        """
        CostFunctions.check_shape(y_true, y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

        Args:
            y_true (np.ndarray): NumPy array of true target values.
            y_pred (np.ndarray): NumPy array of predicted target values.

        Returns:
            float: The computed MAPE.
        """
        CostFunctions.check_shape(y_true, y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def log_loss(y_true: np.ndarray, A: np.ndarray, epsilon : float = 1e-15) -> float:
        """
        Calculate the Log Loss between true and predicted values for a perceptron.
        
        Args:
            y_true (np.ndarray): NumPy array of true target values.
            A (np.ndarray): NumPy array of predicted probabilities (activations) from the perceptron.
        
        Returns:
            float: The computed Log Loss.
        
        Note:
            The input arrays `y_true` and `A` must have the same shape.
        """
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        return 1 / len(y_true) * np.sum(-y_true * np.log(A + epsilon) - (1 - y_true) * np.log(1 - A + epsilon))



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------