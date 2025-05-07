import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data():
    """Ładowanie danych z ArgumentParser"""
    parser = argparse.ArgumentParser(description='Wczytuje dane.')
    parser.add_argument("-s", type=argparse.FileType('r'), help="np. -s Dane/daneXX.txt", required=True)
    args = parser.parse_args()
    data = [line.strip().split() for line in args.s if line.strip()]
    X = np.array([[float(x)] for x, _ in data])
    y = np.array([[float(y)] for _, y in data])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X, y):
    """Trenowanie modelu."""
    F = np.hstack([X, np.ones(X.shape)])
    V = np.linalg.inv(F.T @ F) @ F.T @ y
    E = y - (V[0] * X + V[1])
    MSE = (E.T @ E) / len(E)
    print(MSE)
    return V


def train_model2(X, y):
    F = np.hstack([X ** 2, X, np.ones(X.shape)])
    V = np.linalg.pinv(F) @ y
    E = y - (V[0]*X*X + V[1]*X + V[2])
    print((E.T @ E) / len(E))
    return V


def print_model_equation(w):
    """Dopasowana funkcja liniowa do danych."""
    terms = [f"{coef:.4f} * x^{i}" if i > 0 else f"{coef:.4f}"
             for i, coef in zip(range(len(w) - 1, -1, -1), w.flatten())]
    print("Dopasowana funkcja: y =", " + ".join(terms))


def visualize(X, y, V):
    """Wizualizacja matplotlib"""
    plt.scatter(X, y, label='Dane treningowe', color='blue')
    plt.plot(X, y, 'ro')
    plt.plot(X, V[0] * X + V[1])
    plt.xlabel('x_wejscie')
    plt.ylabel('y_wejscie')
    plt.show()


def visualize2(X, y, V):
    plt.scatter(X, y, label='Dane treningowe', color='pink')
    plt.plot(X, y, 'ro')
    plt.plot(X, V[0] * X * X + V[1] * X + V[2])
    plt.xlabel('x_wejscie')
    plt.ylabel('y_wejscie')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_data()
    visualize(X_test, y_test, train_model(X_train, y_train))
    visualize2(X_test, y_test, train_model2(X_train, y_train))


if __name__ == "__main__":
    main()
