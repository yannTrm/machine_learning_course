# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np 
import matplotlib.pyplot as plt

import warnings


from src.perceptron import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Exemple d'utilisation

if __name__=="__main__":   
    

    # test with real data
    iris = load_iris()
    
    X = iris.data[:, (0, 1)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int32)
    y = y.reshape((y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    perceptron = Perceptron(n_iter=500)
    perceptron.fit(X_train, y_train, record_loss = True)
    pred = perceptron.predict(X_test)
    # Accuracy scores
    print(f"Accuracy of our perceptron: {accuracy_score(pred, y_test)}")

    plt.figure()
    perceptron.plot_decision_boundary(X_test[:, :2], y_test, title="our decision boundaries")
    
    losses = perceptron.get_loss_history()
    
    plt.figure()
    plt.plot(losses)
    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------