import pytest
import numpy as np
import scipy.sparse as sp

# Route to your module
from glassboxml.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------
@pytest.fixture
def sample_corpus():
    """A minimal, deterministic medical corpus for testing."""
    return [
        "The patient requires an immediate dose of epinephrine.",
        "Inventory alert: The hospital is running low on o-negative blood.",
        "The patient is stable. No further epinephrine is needed at this time."
    ]

# ---------------------------------------------------------
# Exception & Input Validation Tests
# ---------------------------------------------------------
def test_invalid_engine_mode():
    """Ensure the system catches typo'd architectural modes instantly."""
    with pytest.raises(ValueError, match="mode must be either 'dense' or 'sparse'"):
        TfidfVectorizer(mode="quantum")

def test_invalid_input_types():
    """Prevent the math engines from choking on bad data types."""
    vectorizer = TfidfVectorizer()
    
    with pytest.raises(TypeError, match="Input must be a list of strings"):
        vectorizer.fit("This is a single string, not a list.")
        
    with pytest.raises(TypeError, match="Input must be a list of strings"):
        vectorizer.fit([100, 200, 300])  # List of ints

def test_unfitted_transform_protection(sample_corpus):
    """Ensure Transformers enforce the fit -> transform sequence."""
    vectorizer = TfidfVectorizer()
    with pytest.raises(ValueError, match="Call fit\\(\\) before transform\\(\\)"):
        vectorizer.transform(sample_corpus)

# ---------------------------------------------------------
# Architectural Integrity Tests
# ---------------------------------------------------------
def test_engine_output_types(sample_corpus):
    """Verify each engine returns the exact data structure it promises."""
    # Production Engine
    sparse_vec = TfidfVectorizer(mode="sparse")
    sparse_matrix = sparse_vec.fit_transform(sample_corpus)
    assert isinstance(sparse_matrix, sp.csr_matrix), "Sparse mode failed to return a SciPy CSR matrix."

    # Educational Engine
    dense_vec = TfidfVectorizer(mode="dense")
    dense_matrix = dense_vec.fit_transform(sample_corpus)
    assert isinstance(dense_matrix, np.ndarray), "Dense mode failed to return a pure NumPy array."

def test_vocabulary_consistency(sample_corpus):
    """Verify both engines extract the exact same dictionary of words."""
    sparse_vec = TfidfVectorizer(mode="sparse").fit(sample_corpus)
    dense_vec = TfidfVectorizer(mode="dense").fit(sample_corpus)
    
    assert sparse_vec.vocabulary_ == dense_vec.vocabulary_, "Engines built different dictionaries!"
    assert sparse_vec.n_features_ == dense_vec.n_features_, "Feature dimensions do not match."

# ---------------------------------------------------------
# The Mathematical Parity Check (The Big One)
# ---------------------------------------------------------
def test_dual_engine_mathematical_parity(sample_corpus):
    """
    The Ultimate CI/CD Test: 
    Proves that the optimized pointer-math in the SciPy engine 
    produces the exact same geometry as the standard NumPy loop.
    """
    sparse_matrix = TfidfVectorizer(mode="sparse").fit_transform(sample_corpus).toarray()
    dense_matrix = TfidfVectorizer(mode="dense").fit_transform(sample_corpus)
    
    # np.allclose is crucial here because floating-point math can have micro-variations
    # between CPU architectures and C-backends.
    assert np.allclose(sparse_matrix, dense_matrix), "CRITICAL: Mathematical parity broken between engines!"

def test_explain_method_telemetry():
    """Ensure the explain method reflects the correct active engine."""
    sparse_vec = TfidfVectorizer(mode="sparse")
    dense_vec = TfidfVectorizer(mode="dense")
    
    # Must be fitted to call explain
    sparse_vec.fit(["test string"])
    dense_vec.fit(["test string"])
    
    assert "SciPy CSR" in sparse_vec.explain()
    assert "NumPy Dense" in dense_vec.explain() 