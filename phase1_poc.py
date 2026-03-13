"""
Phase I proof-of-concept model for DS 4420 project.

This script trains a simple non-Bayesian regression model on the
Boston Housing dataset and reports a basic test error. It is
intended only as a minimal working example to satisfy the Phase I
requirement of having a first model in the GitHub repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 200
    max_depth: int | None = None


def load_boston_housing():
    """
    Load the Boston Housing dataset from OpenML.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector (median home value).
    """
    dataset = fetch_openml(name="boston", version=1, as_frame=True)
    X = dataset.data.to_numpy()
    y = dataset.target.to_numpy(dtype=float)
    return X, y


def train_poc_model(config: ModelConfig) -> None:
    """
    Train a simple Random Forest regression model and print test RMSE.
    """
    X, y = load_boston_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Phase I Proof-of-Concept Model")
    print("--------------------------------")
    print(f"Model: RandomForestRegressor (n_estimators={config.n_estimators})")
    print(f"Test RMSE: {rmse:.3f}")


if __name__ == "__main__":
    cfg = ModelConfig()
    train_poc_model(cfg)

