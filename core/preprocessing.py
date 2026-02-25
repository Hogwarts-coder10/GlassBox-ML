import numpy as np

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Essential for gradient descent and distance-based algorithms.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Computes the mean and std to be used for later scaling."""
        # Calculate mean and standard deviation for each column (axis=0)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        
        # Prevent division by zero just in case a feature is completely constant
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """Performs standardization by centering and scaling."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call .fit(X) first.")
        
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fits to data, then transforms it."""
        return self.fit(X).transform(X)
    

class PolynomialFeatures:
    """
    Generates polynomial and interaction features.
    Transforms [x] into [x, x^2, x^3, ...] to allow linear models to fit curves.
    """
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        """
        Takes a column vector X and returns a matrix with polynomial columns.
        """
        n_samples, n_features = X.shape
        
        # GlassBox Capacity Warning
        if self.degree > 5:
            print(f"⚠️ GLASSBOX WARNING: Degree {self.degree} polynomial requested.")
            print("Expect extreme variance (Runge's phenomenon) and severe overfitting.")
            
        if self.degree >= n_samples:
            print(f"⚠️ GLASSBOX FATAL: Degree ({self.degree}) >= Number of samples ({n_samples}).")
            print("The model will perfectly memorize the noise. Validation error will explode.")

        # Create the polynomial columns
        X_poly = np.copy(X)
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
            
        return X_poly

    def fit_transform(self, X):
        return self.transform(X)