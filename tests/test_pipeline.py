import pytest
import numpy as np
import sys, os

# Route Python to your GlassBoxML root directory

from glassboxml.core import Pipeline
from glassboxml.core._base_model import GlassBoxModel

# ---------------------------------------------------------
# Isolated Mock Components for Unit Testing
# ---------------------------------------------------------
class MockTransformer(GlassBoxModel):
    """Multiplies incoming data by a set factor."""
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor
        self.is_fitted = False

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Not fitted")
        return X * self.factor

    # ---> ADDED TO SUPPORT PIPELINE FIT_TRANSFORM CHAIN <---
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        raise NotImplementedError("MockTransformer is a Transformer, not a Predictor.")

    def explain(self):
        return f"MockTransformer (x{self.factor})"

class MockPredictor(GlassBoxModel):
    """Sums the incoming matrix rows to simulate a prediction."""
    def __init__(self):
        super().__init__()
        self.is_fitted = False

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Not fitted")
        return np.sum(X, axis=1)

    def explain(self):
        return "MockPredictor"

# ---------------------------------------------------------
# Pipeline Architecture Tests
# ---------------------------------------------------------
def test_pipeline_duck_typing_validation():
    """Ensure the Pipeline crashes if a Predictor is put in the middle."""
    with pytest.raises(TypeError, match="must implement .transform()"):
        Pipeline([
            ('bad_step', MockPredictor()), # Predictors don't have transform()
            ('final', MockPredictor())
        ])

def test_pipeline_execution_chain():
    """Verify data flows sequentially through multiple transformers."""
    pipeline = Pipeline([
        ('doubler', MockTransformer(factor=2)),
        ('tripler', MockTransformer(factor=3)),
        ('classifier', MockPredictor())
    ])

    # Raw input: a 2x2 matrix of ones
    X_raw = np.ones((2, 2))

    # Fit and predict in one seamless chain
    pipeline.fit(X_raw)
    predictions = pipeline.predict(X_raw)

    # The mathematical flow:
    # 1. Doubler turns 1s into 2s.
    # 2. Tripler turns 2s into 6s.
    # 3. Predictor sums the rows: [6+6, 6+6] = [12, 12]
    expected_output = np.array([12, 12])

    assert np.array_equal(predictions, expected_output), "Data did not flow through transformers correctly!"

def test_unfitted_pipeline_protection():
    """Ensure the pipeline blocks predictions before the chain is trained."""
    pipeline = Pipeline([
        ('step1', MockTransformer()),
        ('step2', MockPredictor())
    ])

    X_raw = np.ones((2, 2))
    with pytest.raises(ValueError, match="Call fit\\(\\) before predict\\(\\)"):
        pipeline.predict(X_raw)

def test_pipeline_fit_transform():
    """Verify fit_transform works when the pipeline consists entirely of Transformers."""
    pipeline = Pipeline([
        ('doubler', MockTransformer(factor=2)),
        ('tripler', MockTransformer(factor=3))
    ])

    X_raw = np.ones((2, 2))

    # The mathematical flow: 1s -> 2s -> 6s
    expected_output = np.ones((2, 2)) * 6

    final_matrix = pipeline.fit_transform(X_raw)

    assert pipeline.is_fitted, "Pipeline state was not updated to fitted."
    assert np.array_equal(final_matrix, expected_output), "fit_transform failed to route data correctly!"
