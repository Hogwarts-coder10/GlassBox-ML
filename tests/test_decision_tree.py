import numpy as np
from glassboxml.models import DecisionTreeClassifier


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_decision_tree():
    X, y = generate_data()

    model = DecisionTreeClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (100,)
    assert set(preds).issubset(set(y))

    acc = np.mean(preds == y)
    assert acc > 0.6