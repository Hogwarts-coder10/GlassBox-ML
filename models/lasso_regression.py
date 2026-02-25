import numpy as np
from core.base_model import GlassBoxModel

class LassoRegression(GlassBoxModel):
    """
    Transparent Lasso Regression (L1 Regularization).
    Performs automatic feature selection by forcing useless feature weights to zero.
    """
    def __init__(self, optimizer, alpha=1.0, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.alpha = alpha  
        self.epochs = epochs
        self.params = {'w': None, 'b': None}

    def check_assumptions(self, X, y):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        max_var, min_var = np.max(variances), np.min(variances)
        
        if min_var > 0 and (max_var / min_var) > 10:
            self.failure_modes.append(
                f"Unscaled features detected. Lasso Regression will aggressively "
                "delete features with smaller scales. Please standardize X."
            )
        return self.failure_modes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.params['w'] = np.zeros(n_features)
        self.params['b'] = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # 1. Loss: MSE + L1 Penalty (Sum of absolute weights)
            mse_loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l1_penalty = self.alpha * np.sum(np.abs(self.params['w']))
            loss = mse_loss + l1_penalty
            
            # 2. Gradients: MSE gradient + L1 derivative (sign of w)
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha * np.sign(self.params['w'])),
                'db': (1 / n_samples) * np.sum(y_pred - y)
            }
            
            self.params = self.optimizer.update(self.params, grads)
            self._record_step(epoch_loss=loss, epoch_gradients={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]
        
        # GlassBox Feature Selection: Snap tiny weights to exact 0
        tolerance = 1e-4
        zeroed_out = np.sum(np.abs(self.params['w']) < tolerance)
        self.params['w'][np.abs(self.params['w']) < tolerance] = 0.0
        
        if zeroed_out > 0:
            self.failure_modes.append(
                f"Feature Selection: Lasso forced {zeroed_out} out of {n_features} features to exactly 0.0."
            )

    def predict(self, X):
        return np.dot(X, self.params['w']) + self.params['b']