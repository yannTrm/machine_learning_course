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
    def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:
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
        m, n = X.shape  # Number of training examples and features

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
    def stochastic_gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:
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
        m, n = X.shape  # Number of training examples and features

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
    def momentum(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float, num_iterations: int, momentum_factor: float = 0.9) -> np.ndarray:
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
        m, n = X.shape  # Number of training examples and features
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
    def rmsprop(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float, num_iterations: int, decay_factor: float = 0.9) -> np.ndarray:
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
        m, n = X.shape  # Number of training examples and features
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
    def adam(X: np.ndarray, y: np.ndarray, theta: np.ndarray, learning_rate: float, num_iterations: int, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> np.ndarray:
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
        m, n = X.shape  # Number of training examples and features
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






from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Générer des données aléatoires
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

X = np.concatenate((x,np.ones((x.shape[0], 1))), axis = 1)

# Initialiser les paramètres
theta_initial = np.random.randn(2, 1)  # theta0 et theta1

# Utiliser l'algorithme de Gradient Descent personnalisé
learning_rate = 0.1
num_iterations = 1000
theta_gd = RegressionAlgorithms.gradient_descent(X, y, theta_initial, learning_rate, num_iterations)

# Utiliser scikit-learn pour comparer
model = LinearRegression()
model.fit(X, y)
theta_sklearn = np.concatenate((model.intercept_, model.coef_.flatten()))


# Calculer les prédictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Ajouter x0 = 1 à chaque instance
y_predict_gd = X_new_b.dot(theta_gd)
y_predict_sklearn = X_new_b.dot(theta_sklearn)

# Comparer les prédictions
print("Prédictions avec Gradient Descent personnalisé:")
print(y_predict_gd)
print("Prédictions avec scikit-learn:")
print(y_predict_sklearn)

# Calculer l'erreur MSE
mse_gd = mean_squared_error(y, X.dot(theta_gd))
mse_sklearn = mean_squared_error(y, X.dot(theta_sklearn))
print("MSE avec Gradient Descent personnalisé:", mse_gd)
print("MSE avec scikit-learn:", mse_sklearn)
