# -*- coding: utf-8 -*-

from src.perceptron import Perceptron
from utilities import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_test = X_test.reshape(X_test.shape[0], -1) / X_train.max()



clf = Perceptron(learning_rate = 0.01, n_iter = 1000)
clf.fit(X_train, y_train, record_loss = True)
plt.plot(clf.get_loss_history())

pred = clf.predict(X_test)
# Accuracy scores
print(f"Accuracy of our perceptron: {accuracy_score(pred, y_test)}")