# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 

import numpy as np

from typing import List

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ActivationFunction:
    @staticmethod
    def step(x):
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


class Perceptron:
    def __init__(self, input_size, activation_function=ActivationFunction.step):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for input_data, target in zip(inputs, targets):
                prediction = self.predict(input_data)
                error = target - prediction

                # Mise à jour des poids et du biais
                self.weights += learning_rate * error * input_data
                self.bias += learning_rate * error

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'entraînement
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 0, 0, 1])

    # Création d'un perceptron avec la fonction d'activation sigmoïde
    perceptron = Perceptron(input_size=2, activation_function=ActivationFunction.sigmoid)

    # Entraînement du perceptron
    perceptron.train(inputs, targets)

    # Test du perceptron
    for input_data in inputs:
        prediction = perceptron.predict(input_data)
        print(f"Input: {input_data}, Prediction: {prediction}")



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------