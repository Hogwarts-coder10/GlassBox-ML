import numpy as np
import pytest

from glassboxml.datasets import make_blobs
from glassboxml.models import GaussianNaiveBayes


# -------------------------------
# Fixtures
# -------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Deterministic dataset for reproducible tests."""
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.5,
        random_state=42
    )
    return X, y


@pytest.fixture(scope="module")
def trained_model(dataset):
    """Train model once per test module."""
    X, y = dataset
    model = GaussianNaiveBayes()
    model.fit(X, y)
    return model


# -------------------------------
# Core Functional Tests
# -------------------------------

def test_fit_does_not_crash(dataset):
    """Sanity: model should fit without errors."""
    X, y = dataset
    model = GaussianNaiveBayes()
    model.fit(X, y)


def test_predict_shape_consistency(dataset, trained_model):
    """Predictions must match input size."""
    X, _ = dataset
    preds = trained_model.predict(X)

    assert preds.shape[0] == X.shape[0]


def test_prediction_values_are_valid_classes(dataset, trained_model):
    """Predictions must belong to known class labels."""
    X, y = dataset
    preds = trained_model.predict(X)

    unique_labels = set(y)
    assert set(preds).issubset(unique_labels)


def test_training_accuracy_reasonable(dataset, trained_model):
    """
    Accuracy should be reasonably high on training data.
    Not 100% strict to avoid overfitting assumptions.
    """
    X, y = dataset
    preds = trained_model.predict(X)

    accuracy = np.mean(preds == y)

    # Gaussian NB on blobs should be strong
    assert accuracy > 0.8


# -------------------------------
# Stability & Determinism
# -------------------------------

def test_prediction_determinism(dataset):
    """Same input → same output (no randomness after fit)."""
    X, y = dataset

    model = GaussianNaiveBayes()
    model.fit(X, y)

    preds1 = model.predict(X)
    preds2 = model.predict(X)

    assert np.array_equal(preds1, preds2)


# -------------------------------
# Edge Cases
# -------------------------------

def test_empty_input_raises(trained_model):
    """Model should fail gracefully on empty input."""
    with pytest.raises(Exception):
        trained_model.predict(np.array([]))


def test_single_sample_prediction(trained_model):
    """Model should handle single sample correctly."""
    sample = np.array([[0.0, 0.0]])
    pred = trained_model.predict(sample)

    assert pred.shape == (1,)


# -------------------------------
# Generalization Check
# -------------------------------

def test_model_generalization():
    """
    Train/test split to ensure model isn't just memorizing.
    """
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.5,
        random_state=42
    )

    # Manual split (no sklearn dependency)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)

    assert accuracy > 0.7


# -------------------------------
# Explainability (GlassBoxML specific)
# -------------------------------

def test_explain_does_not_crash(trained_model):
    """Explain function should run without errors."""
    trained_model.explain()
