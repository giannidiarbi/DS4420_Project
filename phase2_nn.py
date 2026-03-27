import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_boston_housing():
    dataset = fetch_openml(name="boston", version=1, as_frame=True)
    X = dataset.data.to_numpy()
    y = dataset.target.to_numpy(dtype=float)
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
        for i in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            if i % 100 == 0:
                loss = np.mean((y - y_pred.flatten())**2)
                print(f"Epoch {i}, MSE: {loss:.3f}")

# run model
X, y = load_boston_housing()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = SimpleMLP(input_dim=X.shape[1], hidden_dim=20, lr=0.01)
mlp.train(X_train, y_train, epochs=500)

# Test predictions
y_pred = mlp.forward(X_test).flatten()
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print("MLP Test RMSE:", rmse)
