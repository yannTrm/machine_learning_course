# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np


from sklearn.linear_model import LinearRegression

from src.utils.regression_algorithms import RegressionAlgorithms
from src.utils.cost_functions import CostFunctions


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__=="__main__":
    
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
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    theta_sklearn = model.coef_.T
    
    
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
    mse_gd = CostFunctions.mse(y, X.dot(theta_gd))
    mse_sklearn = CostFunctions.mse(y, X.dot(theta_sklearn))
    print("MSE avec Gradient Descent personnalisé:", mse_gd)
    print("MSE avec scikit-learn:", mse_sklearn)

