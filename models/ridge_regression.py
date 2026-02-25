import numpy as np
from core.base_model import GlassBoxModel

class RidgeRegression(GlassBoxModel):
    """
    Transparent Ridge Regression (L2 Regularization).
    Prevents overfitting and handles multicollinearity by penalizing large weights.
    """
    def __init__(self, optimizer, alpha=1.0, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.alpha = alpha  # The regularization strength
        self.epochs = epochs
        self.params = {'w': None, 'b': None}

    def check_assumptions(self, X, y):
        """
        Ridge specific diagnostics: Checks if features are scaled.
        """
        self.failure_modes = []
        
        # Calculate the variance of each feature column
        variances = np.var(X, axis=0)
        max_var = np.max(variances)
        min_var = np.min(variances)
        
        # If the highest variance is over 10x the lowest, the data isn't scaled
        if min_var > 0 and (max_var / min_var) > 10:
            self.failure_modes.append(
                f"Unscaled features detected (Max Var: {max_var:.2f}, Min Var: {min_var:.2f}). "
                "Ridge Regression is highly sensitive to feature scales. "
                "Features with larger scales will be penalized less. Please standardize your X data."
            )
            
        return self.failure_modes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.params['w'] = np.zeros(n_features)
        self.params['b'] = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # 1. Calculate Loss (MSE + L2 Penalty)
            mse_loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l2_penalty = (self.alpha / 2) * np.sum(self.params['w'] ** 2)
            loss = mse_loss + l2_penalty
            
            # 2. Calculate Gradients (MSE gradient + L2 derivative)
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha * self.params['w']),
                'db': (1 / n_samples) * np.sum(y_pred - y) # Bias is not penalized
            }
            
            # 3. Optimizer updates parameters
            self.params = self.optimizer.update(self.params, grads)
            
            # 4. Track the journey
            self._record_step(epoch_loss=loss, epoch_gradient={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]

    def predict(self, X):
        return np.dot(X, self.params['w']) + self.params['b']