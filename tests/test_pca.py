import numpy as np
from glassboxml.models import PCA


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X


def test_pca():
    X = generate_data()

    model = PCA(n_components=2)
    X_trans = model.fit_transform(X)

    assert X_trans.shape == (100, 2)

    assert isinstance(model.explain(), str)