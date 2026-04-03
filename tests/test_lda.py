import numpy as np
from glassboxml.models import LDA

def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_lda():
    X, y = generate_data()

    model = LDA(n_components=1)
    X_trans = model.fit_transform(X, y)

    assert X_trans.shape == (100, 1)

    assert isinstance(model.explain(), str)