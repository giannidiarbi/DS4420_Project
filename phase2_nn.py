from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_boston_housing():
    dataset = fetch_openml(name="boston", version=1, as_frame=True)
    X = dataset.data.astype("float64").to_numpy()
    y = dataset.target.to_numpy(dtype=np.float64)
    return X, y

# Manual Neural Network
class SimpleMLP:
    def __init__(self, input_dim, hidden_dim=10, lr=0.01):
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz2 = (y_pred - y.reshape(-1,1)) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = dz2 @ self.W2.T * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=500):
        history = []
        for i in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            loss = float(np.mean((y - y_pred.flatten()) ** 2))
            history.append(loss)
            if i % 100 == 0:
                print(f"Epoch {i}, MSE: {loss:.3f}")
        return history

def main():
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_boston_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    mlp = SimpleMLP(input_dim=X.shape[1], hidden_dim=20, lr=0.01)
    history = mlp.train(X_train, y_train, epochs=500)

    y_pred = mlp.forward(X_test).flatten()
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    print(f"MLP Test RMSE: {rmse:.3f}")

    mae = float(np.mean(np.abs(y_test - y_pred)))
    print(f"MLP Test MAE: {mae:.3f}")

    # Training loss
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(range(1, len(history) + 1), history, color="steelblue", linewidth=1.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE")
    ax1.set_title("MLP training loss (Boston Housing)")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_dir / "mlp_train_mse.png", dpi=300)
    plt.close(fig1)

    # Predicted vs actual (test set)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(y_test, y_pred, alpha=0.65, edgecolors="none", s=28, c="steelblue")
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect prediction")
    ax2.set_xlabel("Actual medv ($1000s)")
    ax2.set_ylabel("Predicted medv ($1000s)")
    ax2.set_title(f"MLP test set (RMSE = {rmse:.3f})")
    ax2.legend(loc="upper left")
    ax2.set_aspect("equal", adjustable="box")
    fig2.tight_layout()
    fig2.savefig(out_dir / "mlp_pred_vs_actual.png", dpi=300)
    plt.close(fig2)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
