import numpy as np


def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y

def test_fit_does_not_crash(model, X, y):
    model.fit(X, y)

def test_predict_shape(model, X, y):
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]

def test_output_validity(model, X, y):
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

def test_single_sample(model, X, y):
    model.fit(X, y)
    pred = model.predict(X[:1])
    assert pred.shape[0] == 1

def test_explain(model, X, y):
    model.fit(X, y)
    assert isinstance(model.explain(), str)

def test_diagnose(model, X, y):
    model.fit(X, y)
    report = model.diagnose()
    assert isinstance(report, dict)