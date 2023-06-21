from adaboost_bindings import AdaBoost # type: ignore
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import pandas as pd

data = None
with open('tic-tac-toe.txt') as f:
    data = f.read()
    
data = data.split('\n')
data = [row.split(',') for row in data]
data = pd.DataFrame(data)

data = data.replace('b', 0)
data = data.replace('x', 1)
data = data.replace('o', 2)
data = data.replace('positive', 1)
data = data.replace('negative', 0)
data = data.astype(int)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

n_estimators_range = np.arange(1, 40)
accuracies = dict()

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

    for n_estimators in n_estimators_range:
        clf = AdaBoost(n_estimators)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[n_estimators] = accuracies.get(n_estimators, []) + [accuracy]

accuracies = {k: np.mean(v) for k, v in accuracies.items()}

os.makedirs("plots", exist_ok=True)

x = list(accuracies.keys())
y = list(accuracies.values())
plt.plot(x, y)
plt.xlabel("Número de estimadores")
plt.ylabel("Acurácia")
plt.legend()
plt.grid()
# plt.ylim(0.5, 1)
plt.savefig("plots/acuracia.png")
plt.clf()
