import numpy as np
from glassboxml.models import Perceptron


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_perceptron():
    X, y = generate_data()

    model = Perceptron(n_iters=100, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict(X)

    # Shape check
    assert preds.shape == (100,)

    # Valid classes
    assert set(preds).issubset(set(y))

    # Accuracy check
    acc = np.mean(preds == y)
    assert acc > 0.6

    # Explain & diagnose
    assert isinstance(model.explain(), str)
    assert isinstance(model.diagnose(), dict)