import numpy as np
from glassboxml.models import KMeansClustering


def generate_data():
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([5, 5])
    X2 = np.random.randn(50, 2) + np.array([-5, -5])
    return np.vstack([X1, X2])


def test_kmeans():
    X = generate_data()

    model = KMeansClustering(k=2)   # ⚠️ not n_clusters
    model.fit(X)

    labels = model.predict(X)

    assert len(labels) == len(X)

    # should detect 2 clusters
    assert len(set(labels)) == 2

    assert isinstance(model.explain(), str)
    assert isinstance(model.diagnose(), dict)