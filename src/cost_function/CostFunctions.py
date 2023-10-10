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
    """

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
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays y_true and y_pred must have the same shape.")

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
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays y_true and y_pred must have the same shape.")

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
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays y_true and y_pred must have the same shape.")

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
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays y_true and y_pred must have the same shape.")

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------