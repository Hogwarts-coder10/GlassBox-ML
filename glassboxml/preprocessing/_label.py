import numpy as np
from typing import Union, List, Optional

# Adjust path if your base class is elsewhere
from glassboxml.core._base_model import GlassBoxModel

class LabelEncoder(GlassBoxModel):
    """
    Encode target labels with values between 0 and n_classes-1.
    
    Converts categorical text labels into machine-readable integers. 
    Strictly designed for target variables (y), not 2D feature matrices (X).
    """
    def __init__(self):
        super().__init__()
        # State attributes
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, y: Union[List, np.ndarray]) -> "LabelEncoder":
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("LabelEncoder should be fed a 1D array of targets (y).")

        # np.unique automatically finds unique elements AND sorts them alphabetically/numerically
        self.classes_ = np.unique(y_array)
        self.is_fitted = True
        return self

    def transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")

        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("LabelEncoder should be fed a 1D array of targets (y).")

        # Strict Validation: Check for unseen labels
        unseen = np.setdiff1d(y_array, self.classes_)
        if len(unseen) > 0:
            raise ValueError(f"y contains previously unseen labels: {unseen}")

        # Vectorized Mapping (The Optimization)
        # Because np.unique() sorted self.classes_, we can use binary search mapping 
        # via np.searchsorted. This is drastically faster than a Python dictionary loop.
        return np.searchsorted(self.classes_, y_array)

    def fit_transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Union[List[int], np.ndarray]) -> np.ndarray:
        """Converts integer predictions back into their original text labels."""
        if not self.is_fitted:
            raise ValueError("Call fit() before inverse_transform().")

        y_array = np.asarray(y)
        
        # Prevent index out-of-bounds errors
        if np.any((y_array < 0) | (y_array >= len(self.classes_))):
            raise ValueError("y contains out-of-bounds integer indices.")

        # Vectorized array indexing to instantly map ints back to strings
        return self.classes_[y_array]

    def predict(self, X):
        """Satisfies the GlassBoxModel abstract interface."""
        raise NotImplementedError("LabelEncoder is a Transformer, not a Predictor.")

    def explain(self) -> str:
        if not self.is_fitted:
            return "Model is not fitted yet."
        
        explanation = "--- GlassBox Explanation: LabelEncoder ---\n"
        explanation += f"Learned Classes: {self.classes_.tolist()}\n"
        explanation += "Mapping:\n"
        for idx, label in enumerate(self.classes_):
            explanation += f"  {idx} -> '{label}'\n"
        explanation += "\nArchitecture: Uses pure NumPy binary search (np.searchsorted) to instantly translate categorical strings into mathematically digestible integers."
        return explanation
