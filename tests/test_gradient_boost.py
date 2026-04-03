import numpy as np
from glassboxml.models import GradientBoostingClassifier

def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)
    return X, y

def test_gradient_boost():
    X, y = generate_data()

    model = GradientBoostingClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (100,)