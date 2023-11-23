# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Classes

class RegressionAlgorithms:
    """
    A class containing common regression algorithms.
    
    This class provides implementations of various regression algorithms, including:
    - Gradient Descent
    - Stochastic Gradient Descent (SGD)
    - Momentum
    - RMSprop
    - Adam
    """

    @staticmethod
    def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, num_iterations: int = 100) -> np.ndarray:
        """
        Perform linear regression using Gradient Descent.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for gradient descent.
            num_iterations (int): Number of iterations for gradient descent.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features

        for _ in range(num_iterations):
            # Compute predictions
            predictions = np.dot(X, theta)

            # Calculate error
            error = predictions - y

            # Compute gradients
            gradients = (1 / m) * np.dot(X.T, error)

            # Update parameters using gradients and learning rate
            theta -= learning_rate * gradients

        return theta

    @staticmethod
    def stochastic_gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, num_iterations: int = 100) -> np.ndarray:
        """
        Perform linear regression using Stochastic Gradient Descent (SGD).

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for SGD.
            num_iterations (int): Number of iterations for SGD.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features

        for _ in range(num_iterations):
            for i in range(m):
                # Randomly select a training example
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                # Compute prediction
                prediction = np.dot(xi, theta)

                # Calculate error
                error = prediction - yi

                # Compute gradient for the selected example
                gradient = xi.T.dot(error)

                # Update parameters using gradients and learning rate
                theta -= learning_rate * gradient

        return theta
    
    @staticmethod
    def mini_batch_gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, batch_size: int = 32, num_iterations: int = 100, verbose: bool = False) -> np.ndarray:
        """
        Perform linear regression using Mini-Batch Gradient Descent.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for Mini-Batch GD.
            batch_size (int): Size of the mini-batch.
            num_iterations (int): Number of iterations for Mini-Batch GD.
            verbose (bool): Whether to display information for each epoch.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features

        for epoch in range(num_iterations):
            # Shuffle the data for randomness in mini-batch selection
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                # Select a mini-batch
                mini_batch_X = X_shuffled[i:i + batch_size]
                mini_batch_y = y_shuffled[i:i + batch_size]

                # Compute predictions for the mini-batch
                predictions = np.dot(mini_batch_X, theta)

                # Calculate errors for the mini-batch
                errors = predictions - mini_batch_y

                # Compute gradients for the mini-batch
                gradients = mini_batch_X.T.dot(errors)

                # Update parameters using gradients and learning rate
                theta -= learning_rate * gradients

            if verbose:
                cost = np.mean((np.dot(X, theta) - y) ** 2) / 2
                print(f"Epoch {epoch + 1}/{num_iterations}, Cost: {cost}")

        return theta


    @staticmethod
    def momentum(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, num_iterations: int = 100, momentum_factor: float = 0.9) -> np.ndarray:
        """
        Perform linear regression using Momentum.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for momentum.
            num_iterations (int): Number of iterations for momentum.
            momentum_factor (float, optional): Momentum factor (usually between 0 and 1). Default is 0.9.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features
        velocity = np.zeros_like(theta)  # Initialize velocity

        for _ in range(num_iterations):
            # Compute predictions
            predictions = np.dot(X, theta)

            # Calculate error
            error = predictions - y

            # Compute gradients
            gradients = (1 / m) * np.dot(X.T, error)

            # Update velocity using gradients and momentum factor
            velocity = momentum_factor * velocity + learning_rate * gradients

            # Update parameters using velocity
            theta -= velocity

        return theta

    @staticmethod
    def rmsprop(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, num_iterations: int = 100, decay_factor: float = 0.9) -> np.ndarray:
        """
        Perform linear regression using RMSprop.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for RMSprop.
            num_iterations (int): Number of iterations for RMSprop.
            decay_factor (float, optional): Decay factor for RMSprop. Default is 0.9.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features
        epsilon = 1e-8  # Small constant to prevent division by zero
        cache = np.zeros_like(theta)  # Initialize cache

        for _ in range(num_iterations):
            # Compute predictions
            predictions = np.dot(X, theta)

            # Calculate error
            error = predictions - y

            # Compute gradients
            gradients = (1 / m) * np.dot(X.T, error)

            # Update cache with squared gradients
            cache = decay_factor * cache + (1 - decay_factor) * gradients**2

            # Update parameters using RMSprop update rule
            theta -= learning_rate * gradients / (np.sqrt(cache) + epsilon)

        return theta

    @staticmethod
    def adam(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float = 0.01, num_iterations: int = 100, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> np.ndarray:
        """
        Perform linear regression using the Adam optimizer.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            theta (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate for Adam.
            num_iterations (int): Number of iterations for Adam.
            beta1 (float, optional): Exponential decay rate for the first moment estimate. Default is 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimate. Default is 0.999.
            epsilon (float, optional): Small constant to prevent division by zero. Default is 1e-8.

        Returns:
            np.ndarray: Learned parameters (theta).
        """
        m, _ = X.shape  # Number of training examples and features
        moment1 = np.zeros_like(theta)  # Initialize first moment estimate
        moment2 = np.zeros_like(theta)  # Initialize second moment estimate
        t = 0  # Initialize time step

        for _ in range(num_iterations):
            t += 1  # Update time step
            # Compute predictions
            predictions = np.dot(X, theta)

            # Calculate error
            error = predictions - y

            # Compute gradients
            gradients = (1 / m) * np.dot(X.T, error)

            # Update first moment estimate
            moment1 = beta1 * moment1 + (1 - beta1) * gradients

            # Update second moment estimate
            moment2 = beta2 * moment2 + (1 - beta2) * gradients**2

            # Correct for bias in first moment estimate
            moment1_corrected = moment1 / (1 - beta1**t)

            # Correct for bias in second moment estimate
            moment2_corrected = moment2 / (1 - beta2**t)

            # Update parameters using Adam update rule
            theta -= learning_rate * moment1_corrected / (np.sqrt(moment2_corrected) + epsilon)

        return theta

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------