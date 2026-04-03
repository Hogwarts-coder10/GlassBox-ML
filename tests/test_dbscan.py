from glassboxml.models import DBSCAN
import numpy as np

def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X,y

def test_dbscan():
    X,_ = generate_data()

    model = DBSCAN()
    model.fit(X)

    labels = model.labels_

    assert len(labels) == len(X)

    # DBSCAN should find clusters or noise
    assert len(set(labels)) >= 1

    assert isinstance(model.explain(), str)
