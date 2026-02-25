import numpy as np
from core.base_model import GlassBoxModel

class LinearRegression(GlassBoxModel):
    """
    Transparent Linear Regression using injected Optimizers.
    Maps input features to a continuous output using a linear combination.
    """
    def __init__(self, optimizer, epochs=1000):
        super().__init__()
        self.optimizer = optimizer
        self.epochs = epochs
        # Grouping parameters so the optimizer can easily update them
        self.params = {'w': None, 'b': None}

    def check_assumptions(self, X, y):
        """
        Diagnoses potential data issues before training begins.
        """
        self.failure_modes = [] # Reset on each run
        n_samples, n_features = X.shape

        if n_samples < n_features:
            self.failure_modes.append(
                f"Underdetermined system: {n_samples} samples for {n_features} features. "
                "The model will likely overfit and memorize the training data."
            )
            
        cond_number = np.linalg.cond(X)
        if cond_number > 30:
            self.failure_modes.append(
                f"High multicollinearity detected (Condition number: {cond_number:.2f}). "
                "Gradients may oscillate and weights will be highly sensitive to noise."
            )
            
        return self.failure_modes

    def fit(self, X, y):
        """
        Trains the model while recording the optimization journey.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters to zeros inside the dictionary
        self.params['w'] = np.zeros(n_features)
        self.params['b'] = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            
            # 1. Calculate Loss
            loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            
            # 2. Calculate Gradients
            grads = {
                'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)),
                'db': (1 / n_samples) * np.sum(y_pred - y)
            }
            
            # 3. Optimizer updates the parameters
            self.params = self.optimizer.update(self.params, grads)
            
            # 4. Track the journey
            # We store a copy of the gradients so users can inspect them later
            self._record_step(epoch_loss=loss, epoch_gradient={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]

    def predict(self, X):
        """ Compute the linear combination: y = Xw + b """
        return np.dot(X, self.params['w']) + self.params['b']