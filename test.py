from sklearn import datasets
from adaboost_bindings import AdaBoost # type: ignore
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

data = np.loadtxt('vampire_dataset.txt', delimiter=',')
data = data.astype(int)
X = data[:, 2:]
y = data[:, 1]

n_estimators = 3
clf = AdaBoost(n_estimators)
clf.fit(X, y)
print(clf.predict(X))
