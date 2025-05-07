import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

a = np.loadtxt('Dane/dane3.txt')
X = a[:, [1]]
y = a[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

F = np.hstack([X_train, np.ones(X_train.shape)])
V = np.linalg.inv(F.T @ F) @ F.T @ y_train
E = y_test - (V[0] * X_test + V[1])

F1 = np.hstack([X_train * X_train, X_train, np.ones(X_train.shape)])
V1 = np.linalg.pinv(F1) @ y_train
E1 = y_test - (V1[0] * X_test ** 2 + V1[1] * X_test + V1[2])


print("Współczynniki modelu liniowego: ", V.ravel())
print("Współczynniki modelu kwadratowego: ", V1.ravel())
plt.plot(X, y, 'ro')
plt.plot(X, V[0] * X + V[1])
plt.plot(X, V1[0] * X ** 2 + V1[1] * X + V1[2])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
