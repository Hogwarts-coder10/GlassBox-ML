import numpy as np
from glassboxml.models import RandomForestClassifier

def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y

def test_random_forest():
    X, y = generate_data()

    model = RandomForestClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (100,)