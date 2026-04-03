import numpy as np
from glassboxml.models import LogisticRegression
from glassboxml.core.optimizer import Momentum


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_logistic_regression():
    X, y = generate_data()

    model = LogisticRegression(
        optimizer=Momentum(learning_rate=0.1),
        epochs=200
    )

    model.fit(X, y)
    preds = model.predict(X)

    # Shape check
    assert preds.shape == (100,)

    # Valid classes
    assert set(preds).issubset(set(y))

    # Accuracy check (keep safe threshold)
    acc = np.mean(preds == y)
    assert acc > 0.6

    # Explain & diagnose
    assert isinstance(model.explain(), str)
    assert isinstance(model.diagnose(), dict)


def test_deterministic():
    X, y = generate_data()
    model2 = LogisticRegression(optimizer=Momentum(0.1), epochs=100)
    model2.fit(X, y)

    preds = model2.predict(X)

    assert np.allclose(preds, model2.predict(X))