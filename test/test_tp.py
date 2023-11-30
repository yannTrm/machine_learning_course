# Import des bibliothèques nécessaires
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor

# 1. Génération de 2000 points de données avec une caractéristique à une dimension
n_samples = 2000
X, y = make_regression(n_samples=n_samples, n_features=1, noise=10, random_state=42)

# 2. Ajustement d'une ligne avec deux méthodes différentes
# Méthode 1 : Utilisation de la régression linéaire de sklearn
model1 = LinearRegression()
model1.fit(X, y)

# Méthode 2 : Utilisation de la descente de gradient (exemple, vous pouvez choisir une autre méthode)
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, b = 0, 0  # initialisation des coefficients
    for _ in range(epochs):
        y_pred = m * X + b
        m -= learning_rate * ((y_pred - y) * X).mean()
        b -= learning_rate * (y_pred - y).mean()
    return m, b

m, b = gradient_descent(X.flatten(), y)

# 3. Ajout de 20 points aberrants
n_outliers = 20
X_outliers = np.random.uniform(low=-3, high=3, size=(n_outliers, 1))
y_outliers = np.random.uniform(low=-200, high=200, size=n_outliers)
X = np.concatenate([X, X_outliers])
y = np.concatenate([y, y_outliers])

# 4. Ajustement d'une ligne avec tous les points de données
model2 = LinearRegression()
model2.fit(X, y)

# 5. Ajout d'une régularisation (ridge) à la méthode 2
model3 = Ridge(alpha=1.0)
model3.fit(X, y)

# 6. Ajustement du modèle linéaire avec l'algorithme RANSAC
ransac = RANSACRegressor()
ransac.fit(X, y)

# Visualisation des résultats
plt.scatter(X, y, color='black', label='Data points')
plt.plot(X, model1.predict(X), color='blue', linewidth=2, label='Linear Regression (sklearn)')
plt.plot(X, m * X + b, color='green', linewidth=2, label='Linear Regression (Gradient Descent)')
plt.plot(X, model2.predict(X), color='red', linewidth=2, label='Linear Regression with Outliers')
plt.plot(X, model3.predict(X), color='orange', linewidth=2, label='Linear Regression with Ridge')
plt.plot(X, ransac.predict(X), color='purple', linewidth=2, label='Linear Regression with RANSAC')
plt.legend()
plt.show()

# 7. Création d'une instance du classificateur de régression logistique et ajustement des données
log_reg = LogisticRegression()
log_reg.fit(X, y)

# 8. Tracé de la frontière de décision
def plot_decision_boundary(X, y, model, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.show()

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustement du modèle de régression logistique
log_reg.fit(X_train, y_train)

# Tracé de la frontière de décision
plot_decision_boundary(X_test, y_test, log_reg, "Logistic Regression Decision Boundary")
