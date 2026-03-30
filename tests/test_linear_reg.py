import numpy as np
import pytest

from glassboxml.core import Momentum
from glassboxml.models import LinearRegression
from glassboxml.datasets import make_regression


# -------------------------------
# Fixtures
# -------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Deterministic regression dataset."""
    X, y, true_w, true_b = make_regression(
        n_samples=100,
        n_features=1,
        noise=1.0,
        random_state=42
    )
    return X, y.flatten(), true_w, true_b


@pytest.fixture(scope="module")
def trained_model(dataset):
    """Train model once for reuse."""
    X, y, _, _ = dataset
    optimizer = Momentum(learning_rate=0.1, beta=0.09)
    model = LinearRegression(optimizer=optimizer, epochs=100)
    model.fit(X, y)
    return model


# -------------------------------
# Core Functionality
# -------------------------------

def test_fit_does_not_crash(dataset):
    X, y, _, _ = dataset
    optimizer = Momentum(learning_rate=0.1, beta=0.09)
    model = LinearRegression(optimizer=optimizer, epochs=100)
    model.fit(X, y)


def test_predict_shape(trained_model, dataset):
    X, _, _, _ = dataset
    preds = trained_model.predict(X)

    assert preds.shape[0] == X.shape[0]


# -------------------------------
# Learning Quality
# -------------------------------

def test_model_learns_linear_relationship(trained_model, dataset):
    """Model should approximate true parameters reasonably."""
    _, _, true_w, true_b = dataset

    learned_w = trained_model.coef_
    learned_b = trained_model.intercept_

    # Loose tolerance because of noise
    assert np.allclose(learned_w, true_w, atol=1.0)
    assert np.isclose(learned_b, true_b, atol=1.0)


def test_loss_decreases(trained_model):
    """Loss should decrease over epochs."""
    loss_history = trained_model.loss_history

    assert len(loss_history) > 1
    assert loss_history[-1] < loss_history[0]


def test_prediction_accuracy(dataset, trained_model):
    """Check MSE is reasonably low."""
    X, y, _, _ = dataset
    preds = trained_model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert mse < 5.0   # depends on noise level


# -------------------------------
# Stability & Determinism
# -------------------------------

def test_deterministic_training(dataset):
    """Same data + params → same model."""
    X, y, _, _ = dataset

    optimizer1 = Momentum(learning_rate=0.1, beta=0.09)
    model1 = LinearRegression(optimizer=optimizer1, epochs=100)
    model1.fit(X, y)

    optimizer2 = Momentum(learning_rate=0.1, beta=0.09)
    model2 = LinearRegression(optimizer=optimizer2, epochs=100)
    model2.fit(X, y)

    assert np.allclose(model1.coef_, model2.coef_)
    assert np.isclose(model1.intercept_, model2.intercept_)


# -------------------------------
# Edge Cases
# -------------------------------

def test_single_sample_prediction(trained_model):
    sample = np.array([[0.5]])
    pred = trained_model.predict(sample)

    assert pred.shape == (1,)


def test_empty_input_raises(trained_model):
    with pytest.raises(Exception):
        trained_model.predict(np.array([]))


# -------------------------------
# Diagnostics & Explainability
# -------------------------------

def test_diagnose_returns_dict(trained_model):
    report = trained_model.diagnose()

    assert isinstance(report, dict)


def test_explain_returns_string(trained_model):
    explanation = trained_model.explain()

    assert isinstance(explanation, str)
    assert len(explanation) > 0


# -------------------------------
# Generalization Check
# -------------------------------

def test_generalization():
    X, y, _, _ = make_regression(
        n_samples=100,
        n_features=1,
        noise=1.0,
        random_state=42
    )
    y = y.flatten()

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    optimizer = Momentum(learning_rate=0.1, beta=0.09)
    model = LinearRegression(optimizer=optimizer, epochs=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)

    assert mse < 2.0
