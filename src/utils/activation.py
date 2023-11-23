# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class Activation:
    """
    A collection of activation functions for a Perceptron.

    Methods
    -------
    sigmoid(x: np.ndarray) -> np.ndarray:
        Returns the sigmoid activation function applied to the input.

    tanh(x: np.ndarray) -> np.ndarray:
        Returns the hyperbolic tangent activation function applied to the input.

    relu(x: np.ndarray) -> np.ndarray:
        Returns the rectified linear unit activation function applied to the input.

    sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        Returns the derivative of the sigmoid activation function.

    tanh_derivative(x: np.ndarray) -> np.ndarray:
        Returns the derivative of the hyperbolic tangent activation function.

    relu_derivative(x: np.ndarray) -> np.ndarray:
        Returns the derivative of the rectified linear unit activation function.
    """
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Returns the sigmoid activation function applied to the input.

        Parameters
        ----------
        x : np.ndarray
            The input to the sigmoid function.

        Returns
        -------
        np.ndarray
            The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Returns the hyperbolic tangent activation function applied to the input.

        Parameters
        ----------
        x : np.ndarray
            The input to the hyperbolic tangent function.

        Returns
        -------
        np.ndarray
            The output of the hyperbolic tangent function.
        """
        return np.tanh(x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Returns the rectified linear unit activation function applied to the input.

        Parameters
        ----------
        x : np.ndarray
            The input to the rectified linear unit function.

        Returns
        -------
        np.ndarray
            The output of the rectified linear unit function.
        """
        return np.maximum(0, x)

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the sigmoid activation function.

        Parameters
        ----------
        x : np.ndarray
            The input to the sigmoid function.

        Returns
        -------
        np.ndarray
            The derivative of the sigmoid function.
        """
        sigmoid_x = Activation.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the hyperbolic tangent activation function.

        Parameters
        ----------
        x : np.ndarray
            The input to the hyperbolic tangent function.

        Returns
        -------
        np.ndarray
            The derivative of the hyperbolic tangent function.
        """
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the rectified linear unit activation function.

        Parameters
        ----------
        x : np.ndarray
            The input to the rectified linear unit function.

        Returns
        -------
        np.ndarray
            The derivative of the rectified linear unit function.
        """
        return np.where(x > 0, 1, 0)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------