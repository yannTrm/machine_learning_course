# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import mean_squared_error

from src.linear_regression import LinearRegression
from src.utils.regression_algorithms import RegressionAlgorithms

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Générer des données aléatoires
# np.random.seed(0)
a = np.random.uniform(-10, 10)
b = np.random.uniform(-10, 10)

X = 2 * np.random.rand(100, 1)
y = b + a * X + np.random.randn(100, 1)

# Utiliser l'algorithme de Gradient Descent personnalisé
lr_custom = LinearRegression()
lr_custom.fit(X, y, RegressionAlgorithms.adam, learning_rate=0.1, num_iterations=500)

# Utiliser scikit-learn pour comparer
lr_sklearn = skLinearRegression()
lr_sklearn.fit(X, y)

# Calculer les prédictions
X_new = np.array([[0], [2]])
y_predict_custom = lr_custom.predict(X_new)
y_predict_sklearn = lr_sklearn.predict(X_new)

# Comparer les prédictions
print("Prédictions avec Gradient Descent personnalisé:")
print(y_predict_custom)
print("Prédictions avec scikit-learn:")
print(y_predict_sklearn)

# Calculer l'erreur MSE
mse_custom = mean_squared_error(y, lr_custom.predict(X))
mse_sklearn = mean_squared_error(y, lr_sklearn.predict(X))
print("MSE avec Gradient Descent personnalisé:", mse_custom)
print("MSE avec scikit-learn:", mse_sklearn)

# Afficher les données et les régressions
plt.scatter(X, y, label='Données')
plt.plot(X_new, y_predict_custom, 'r-', label='Gradient Descent personnalisé', linewidth=2)
plt.plot(X_new, y_predict_sklearn, 'g--', label='scikit-learn', linewidth=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------