# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import
import numpy as np
import matplotlib.pyplot as plt

from src.support_vector_machine import PegasosSVM, SVM

from sklearn.model_selection import train_test_split
from sklearn import datasets

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    np.random.seed(42)

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create an instance of PegasosSVM
    pegasos_svm = PegasosSVM(learning_rate=0.01, lambda_param=0.1, n_iters=100)
    loss = pegasos_svm.fit(X_train, y_train, loss=True)

    # Test the trained model
    accuracy = pegasos_svm.score(X_test, y_test)
    print("Accuracy:", accuracy)

    pegasos_svm.plot_svm(X_train, y_train)
        
    plt.figure()
    # Plot the learning curve
    plt.plot(loss)
    plt.title('Pegasos SVM Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    
    svm = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=100)
    svm.fit(X_train, y_train)
    
    accuracy = svm.score(X_test, y_test)
    print("Accuracy:", accuracy)
    
    svm.plot_svm(X_train, y_train)
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------