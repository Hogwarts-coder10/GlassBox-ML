import numpy as np
from glassboxml.models import RidgeRegression, LassoRegression
from glassboxml.core.optimizer import Momentum


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 0.0, 1.0]) + 0.1
    return X, y


def test_ridge():
    X, y = generate_data()

    model = RidgeRegression(
        optimizer=Momentum(learning_rate=0.1),
        epochs=100
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (100,)

    mse = np.mean((y - preds) ** 2)
    assert mse < 3.0


def test_lasso():
    X, y = generate_data()

    model = LassoRegression(
        optimizer=Momentum(learning_rate=0.1),
        epochs=100
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (100,)

    mse = np.mean((y - preds) ** 2)
    assert mse < 4.0