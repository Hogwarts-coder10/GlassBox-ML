import numpy as np

from glassboxml.models import KNNClassifier

def generate_classification():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_knn():
    X, y = generate_classification()

    model = KNNClassifier(k=3)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (100,)
    assert set(preds).issubset(set(y))

    acc = np.mean(preds == y)
    assert acc > 0.6

    assert isinstance(model.explain(), str)
    assert isinstance(model.diagnose(), dict)
