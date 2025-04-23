import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data():
    parser = argparse.ArgumentParser(description='Wczytuje dane.')
    parser.add_argument("-s", type=argparse.FileType('r'), help="np. -s Dane/daneXX.txt", required=True)
    args = parser.parse_args()
    data = [line.strip().split() for line in args.s if line.strip()]
    X = np.array([[float(x)] for x, _ in data])
    y = np.array([[float(y)] for _, y in data])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_design_matrix(X, degree):
    """Tworzy macierz cech dla wielomianu stopnia `degree`."""
    return np.hstack([X ** i for i in reversed(range(degree + 1))])


def train_model(X, y, degree=1, lr=0.01, epochs=1000):
    m = X.shape[0]
    F = create_design_matrix(X, degree)
    w = np.zeros((F.shape[1], 1))

    for _ in range(epochs):
        y_pred = F @ w
        error = y_pred - y
        grad = (2 / m) * F.T @ error
        w -= lr * grad

    return w


def evaluate_model(X_test, y_test, w, degree):
    F_test = create_design_matrix(X_test, degree)
    y_pred = F_test @ w
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred


def print_model_equation(w):
    terms = [f"{coef:.4f} * x^{i}" if i > 0 else f"{coef:.4f}"
             for i, coef in zip(range(len(w)-1, -1, -1), w.flatten())]
    print("Dopasowana funkcja: y =", " + ".join(terms))


def visualize(X, y, w, degree):
    plt.scatter(X, y, label='Dane treningowe', color='blue')
    x_vals = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    X_poly = create_design_matrix(x_vals, degree)
    y_vals = X_poly @ w
    plt.plot(x_vals, y_vals, label=f'Dopasowany model (stopień {degree})', color='green')
    plt.title(f"Regresja wielomianowa (stopień {degree})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_data()

    for degree in [1, 2]:
        print(f"\n--- Model stopnia {degree} ---")
        w = train_model(X_train, y_train, degree)
        print_model_equation(w)
        mse, _ = evaluate_model(X_test, y_test, w, degree)
        print(f"MSE dla modelu stopnia {degree}: {mse:.5f}")
        visualize(X_train, y_train, w, degree)


if __name__ == "__main__":
    main()
