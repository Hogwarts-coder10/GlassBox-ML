import numpy as np
import pytest

from glassboxml.models.sparse_random_projection import SparseRandomProjection


# -------------------------------
# Fixtures
# -------------------------------

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 50))


# -------------------------------
# Core Functionality
# -------------------------------

def test_fit_does_not_crash(sample_data):
    model = SparseRandomProjection(n_components=10, random_state=42)
    model.fit(sample_data)


def test_transform_shape(sample_data):
    model = SparseRandomProjection(n_components=10, random_state=42)
    X_proj = model.fit_transform(sample_data)

    assert X_proj.shape == (100, 10)


# -------------------------------
# Determinism
# -------------------------------

def test_deterministic_projection(sample_data):
    model1 = SparseRandomProjection(n_components=10, random_state=42)
    model2 = SparseRandomProjection(n_components=10, random_state=42)

    X1 = model1.fit_transform(sample_data)
    X2 = model2.fit_transform(sample_data)

    assert np.allclose(X1, X2)


# -------------------------------
# Distance Preservation
# -------------------------------

def test_distance_preservation(sample_data):
    model = SparseRandomProjection(n_components=20, random_state=42)
    X_proj = model.fit_transform(sample_data)

    # Pick two random points
    i, j = 0, 1

    dist_original = np.linalg.norm(sample_data[i] - sample_data[j])
    dist_projected = np.linalg.norm(X_proj[i] - X_proj[j])

    # Allow approximation tolerance
    assert np.isclose(dist_projected, dist_original, atol=0.5)


# -------------------------------
# Edge Cases
# -------------------------------

def test_transform_without_fit_raises(sample_data):
    model = SparseRandomProjection(n_components=10)

    with pytest.raises(ValueError):
        model.transform(sample_data)


def test_invalid_input_dimension():
    model = SparseRandomProjection(n_components=10)

    X_invalid = np.array([1, 2, 3])  # 1D input

    with pytest.raises(Exception):
        model.fit(X_invalid)


def test_feature_mismatch(sample_data):
    model = SparseRandomProjection(n_components=10, random_state=42)
    model.fit(sample_data)

    # Wrong feature size
    X_wrong = np.random.randn(100, 30)

    with pytest.raises(ValueError):
        model.transform(X_wrong)


# -------------------------------
# Explainability & Diagnostics
# -------------------------------

def test_explain_returns_string(sample_data):
    model = SparseRandomProjection(n_components=10, random_state=42)
    model.fit(sample_data)

    explanation = model.explain()

    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_diagnose_returns_dict(sample_data):
    model = SparseRandomProjection(n_components=10, random_state=42)
    model.fit(sample_data)

    report = model.diagnose()

    assert isinstance(report, dict)
    assert report["fitted"] is True