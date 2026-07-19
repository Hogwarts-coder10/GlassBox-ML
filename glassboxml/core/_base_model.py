from abc import ABC, abstractmethod

import numpy as np


class GlassBoxModel(ABC):
    """
    The base class for all GlassBox-ML models.
    Enforces a Scikit-Learn style API with educational diagnostics.
    """

    def __init__(self):
        # Standard library naming conventions
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False

        # Educational & Diagnostic trackers
        self.failure_modes = []
        self.loss_history = []
        self.gradient_history = []
        self.training_error = None
        self.dataset_stats_ = {}

    def _store_dataset_stats(self, X):
        """Records the statistical footprint of the training data."""
        self.dataset_stats_ = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "mean": np.mean(X, axis=0),
            "variance": np.var(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
        }

    @abstractmethod
    def fit(self, X, y):
        """Trains the model and updates self.coef_ and self.intercept_."""
        pass

    @abstractmethod
    def predict(self, X):
        """Returns the final prediction."""
        pass

    def decision_function(self, X):
        """
        Returns the raw mathematical output before any thresholding or activation.
        For linear models, this is: X * coef_ + intercept_
        """
        if not self.is_fitted:
            raise ValueError(
                "GlassBox Error: Model is not fitted yet. Call .fit() first."
            )
        return np.dot(X, self.coef_) + self.intercept_

    @abstractmethod
    def explain(self):
        """Returns a human-readable interpretation of the learned weights."""
        pass

    def check_assumptions(self, X, y):
        """Analyzes data for potential mathematical failures."""
        return []

    def _record_step(self, epoch_loss, epoch_gradients):
        """Logs the learning journey for visualization."""
        self.loss_history.append(epoch_loss)
        self.gradient_history.append(epoch_gradients)

    def _get_effective_params(self):
        """
        Estimates the number of effective trainable parameters for this model,
        used as a lightweight proxy for model capacity.

        Default assumes a linear-style model: one parameter per feature plus
        an intercept (based on self.coef_). Subclasses whose capacity isn't
        well described by coef_ (e.g. trees, k-NN, SVM) can override this
        for a more accurate, model-specific estimate.
        """
        if self.coef_ is not None:
            return np.size(self.coef_) + 1
        return self.dataset_stats_.get("n_features", 1)

    @property
    def generalization_estimate(self):
        """
        A lightweight, heuristic estimate of the gap between training error
        and expected out-of-sample error, based on model capacity relative
        to dataset size (a structural-risk-minimization-style intuition:
        more effective parameters per sample widens the expected gap).

        This is NOT a formal generalization bound and does not require a
        held-out validation set -- it is meant to build intuition about
        overfitting risk directly from what fit() already knows, in the
        same spirit as the library's other theory tools.
        """
        if not self.is_fitted or self.training_error is None:
            return {"status": "Model is not fitted yet. Call .fit() first."}

        if not isinstance(self.training_error, (int, float, np.floating, np.integer)):
            return {
                "status": (
                    "Generalization estimate not applicable: this model does not "
                    f"report a numeric training_error (got {self.training_error!r})."
                )
            }

        n_samples = self.dataset_stats_.get("n_samples", 0)
        if n_samples <= 1:
            return {"status": "Insufficient data to estimate generalization."}

        n_params = self._get_effective_params()
        complexity_ratio = n_params / n_samples

        if complexity_ratio < 1:
            penalty = np.sqrt(complexity_ratio / max(1 - complexity_ratio, 1e-6))
            estimated_gap = self.training_error * penalty
            estimated_test_error = self.training_error + estimated_gap
        else:
            estimated_gap = float("inf")
            estimated_test_error = float("inf")

        if complexity_ratio < 0.1:
            risk = "Low"
        elif complexity_ratio < 0.5:
            risk = "Moderate"
        else:
            risk = "High"

        return {
            "training_error": round(float(self.training_error), 4),
            "estimated_test_error": (
                round(float(estimated_test_error), 4)
                if np.isfinite(estimated_test_error)
                else float("inf")
            ),
            "effective_parameters": int(n_params),
            "n_samples": int(n_samples),
            "complexity_ratio": round(float(complexity_ratio), 4),
            "overfitting_risk": risk,
            "note": (
                "Heuristic estimate based on model capacity vs. sample size, "
                "not a validation-set measurement."
            ),
        }

    def diagnose(self):
        """Returns a comprehensive diagnostic report of the model's health."""
        features = self.dataset_stats_.get("n_features", 0)
        samples = self.dataset_stats_.get("n_samples", 0)

        return {
            "Dataset Profile": f"{samples} samples, {features} features",
            "Final Training Error": self.training_error,
            "Optimization Steps": len(self.loss_history),
            "Logged Failure Modes": self.failure_modes
            if self.failure_modes
            else "None detected",
            "Generalization Estimate": self.generalization_estimate,
        }
