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
data = data.replace('negative', -1)
data = data.astype(int)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

os.makedirs("plots", exist_ok=True)

n_estimators_range = np.arange(1, 500)
colors=['blue', 'orange', 'green', 'red', 'purple']
color_index = -1
for learning_rate in np.arange(0.5, 2.5, 0.5):
    print(f"=== Learning rate: {learning_rate} ===")
    color_index += 1
    index = 0
    accuracies = dict()

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kfold.split(X):
        index += 1
        print(f"====== KFold: {index} ======")
        X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
        
        for n_estimators in n_estimators_range:
            percentage = (n_estimators / n_estimators_range[-1])
            print(f"    {percentage:.2%}", end="\r")
            clf = AdaBoost(n_estimators, learning_rate)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[n_estimators] = accuracies.get(n_estimators, []) + [accuracy]

    accuracies = {k: np.mean(v) for k, v in accuracies.items()}
    print(accuracies)

    x_s = list(accuracies.keys())
    y_s = list(accuracies.values())
    plt.plot(x_s, y_s, color=colors[color_index], label=f"Taxa de aprendizado: {learning_rate:.1f}")

plt.xlabel("Número de estimadores")
plt.ylabel("Acurácia")
plt.legend()
plt.grid()
plt.ylim(0.5, 1)
plt.savefig(f"plots/acuracias.png")
plt.clf()