import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    """Ładowanie danych z ArgumentParser"""
    parser = argparse.ArgumentParser(description='Wczytuje dane.')
    parser.add_argument("-s", type=argparse.FileType('r'), help="np. -s Dane/daneXX.txt", required=True)
    args = parser.parse_args()
    data = [line.strip().split() for line in args.s if line.strip()]
    X = np.array([[float(x)] for x, _ in data])
    y = np.array([[float(y)] for _, y in data])
    return X, y, train_test_split(X, y, test_size=0.2, random_state=42)

def train_model_iterative(X_train, y_train, lr=0.01, epochs=1000):
    """Iteracyjne trenowanie regresji liniowej (gradient descent)."""
    m = X_train.shape[0]
    F = np.hstack([X_train**2])  # dodanie biasu
    w = np.zeros((F.shape[1], 1))

    for _ in range(epochs):
        y_pred = F @ w
        error = y_pred - y_train
        grad = (2 / m) * F.T @ error
        w -= lr * grad

    return w

if __name__ == "__main__":
    X, y, (X_train, X_test, y_train, y_test) = load_data()

    # Model liniowy – metoda analityczna
    F = np.hstack([X_train, np.ones_like(X_train)])
    V = np.linalg.inv(F.T @ F) @ F.T @ y_train
    E = y_test - (V[0] * X_test + V[1])

    # Model liniowy – metoda iteracyjna
    V_iter = train_model_iterative(X_train, y_train, lr=0.01, epochs=1000)

    # Model kwadratowy
    F1 = np.hstack([X_train ** 2, X_train, np.ones_like(X_train)])
    V1 = np.linalg.pinv(F1) @ y_train
    E1 = y_test - (V1[0] * X_test ** 2 + V1[1] * X_test + V1[2])

    print("Współczynniki modelu liniowego (analitycznie): ", V.ravel())
    print("Współczynniki modelu liniowego (iteracyjnie): ", V_iter.ravel())
    print("Współczynniki modelu kwadratowego: ", V1.ravel())

    # Sortowanie danych dla płynnych wykresów
    X_sorted = np.sort(X, axis=0)
    y_lin_pred = V[0] * X_sorted + V[1]
    y_quad_pred = V1[0] * X_sorted ** 2 + V1[1] * X_sorted + V1[2]

    x_vals = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    X_poly = np.hstack([x_vals**2])
    y_vals = X_poly @ V_iter
    plt.plot(X, y, 'ro', label="Dane")
    plt.plot(X_sorted, y_lin_pred, 'b--', label="Regresja liniowa (analityczna)")
    plt.plot(x_vals, y_vals, 'g--', label="Regresja liniowa (iteracyjna)")
    plt.plot(X_sorted, y_quad_pred, 'm-', label="Regresja kwadratowa")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
