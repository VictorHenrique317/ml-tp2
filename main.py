from adaboost_bindings import AdaBoost # type: ignore
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

data = np.loadtxt('mnist.txt', delimiter=',')
X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

n_estimators_range = np.arange(1, 11)
sklearn_runtimes = []
rust_runtimes = []
sklearn_accuracies = []
rust_accuracies = []

for n_estimators in n_estimators_range:
    # Create and fit an AdaBoost classifier from scikit-learn
    sklearn_clf = AdaBoostClassifier(n_estimators=n_estimators)
    start_time = time.time()
    sklearn_clf.fit(X_train, y_train)
    sklearn_runtimes.append(time.time() - start_time)

    # Create and fit an AdaBoost classifier from our Rust implementation
    rust_clf = AdaBoost(n_estimators)
    start_time = time.time()
    rust_clf.fit(X_train, y_train)
    rust_runtimes.append(time.time() - start_time)

    # Make predictions on the test set with both classifiers
    sklearn_y_pred = sklearn_clf.predict(X_test)
    rust_y_pred = rust_clf.predict(X_test)

    # Compute the accuracy of both classifiers
    sklearn_accuracies.append(accuracy_score(y_test, sklearn_y_pred))
    rust_accuracies.append(accuracy_score(y_test, rust_y_pred))

# Plot the runtime comparison
plt.plot(n_estimators_range, sklearn_runtimes, label="scikit-learn")
plt.plot(n_estimators_range, rust_runtimes, label="Rust")
plt.xlabel("Number of estimators")
plt.ylabel("Runtime (s)")
plt.legend()
plt.show()

# Plot the accuracy comparison
plt.plot(n_estimators_range, sklearn_accuracies, label="scikit-learn")
plt.plot(n_estimators_range, rust_accuracies, label="Rust")
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
