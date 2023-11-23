# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import DecisionTree
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



def accuracy(y_true, y_pred):   
    accuracy = np.sum(y_true == y_pred)/len(y_true)   
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=123)
clf = DecisionTree(max_depth = 10)
clf.fit(X_train, y_train)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)


y_pred1 = clf.predict(X_train)
acc1 = accuracy(y_train, y_pred1)
print("Training Accuracy: ", acc1)

y_pred2 = clf.predict(X_test) 
acc2 = accuracy(y_test, y_pred2)
print("Testing Accuracy: ", acc2)


y_pred11 = clf1.predict(X_train)
acc1 = accuracy(y_train, y_pred11)
print("Training Accuracy sklearn: ", acc1)

y_pred22 = clf1.predict(X_test) 
acc2 = accuracy(y_test, y_pred22)
print("Testing Accuracy sklearn: ", acc2)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------