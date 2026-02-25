import numpy as np
from core.base_model import GlassBoxModel

class LogisticRegression(GlassBoxModel):
    """
    Transparent Logistic Regression for binary classification.
    Maps input features to a probability between 0 and 1 using the Sigmoid function.
    Includes an educational toggle to demonstrate why MSE fails for classification.
    """
    def __init__(self, optimizer, epochs=1000, threshold=0.5, loss_function='bce'):
        super().__init__()
        self.optimizer = optimizer
        self.epochs = epochs
        self.threshold = threshold
        
        # 'bce' (Binary Cross-Entropy) or 'mse' (Mean Squared Error)
        self.loss_function = loss_function.lower() 
        self.params = {'w': None, 'b': None}

    def _sigmoid(self, z):
        """Squashes continuous values to the [0, 1] probability range."""
        # np.clip prevents overflow errors in exp() if z gets wildly large or small
        z = np.clip(z, -250, 250) 
        return 1 / (1 + np.exp(-z))

    def check_assumptions(self, X, y):
        """Diagnoses classification-specific data issues and bad hyperparameters."""
        self.failure_modes = []
        
        # 1. The GlassBox MSE Warning
        if self.loss_function == 'mse':
            self.failure_modes.append(
                "EDUCATIONAL WARNING: You are using Mean Squared Error (MSE) for classification. "
                "The loss landscape is now non-convex (wavy), and your gradients will vanish "
                "if the model is 'confidently wrong' due to the sigmoid derivative. "
                "Expect poor convergence compared to Binary Cross-Entropy (BCE)."
            )

        # 2. Class Imbalance Check
        class_1_ratio = np.mean(y)
        if class_1_ratio < 0.1 or class_1_ratio > 0.9:
            self.failure_modes.append(
                f"Severe Class Imbalance detected (Class 1 ratio: {class_1_ratio:.2f}). "
                "The model might trivially learn to just predict the majority class."
            )
            
        # 3. Label Check
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0.0, 1.0]):
             self.failure_modes.append(
                f"Invalid labels detected: {unique_labels}. Logistic Regression requires "
                "binary labels strictly formatted as 0 and 1."
             )
             
        return self.failure_modes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.params['w'] = np.zeros(n_features)
        self.params['b'] = 0.0
        
        self.check_assumptions(X, y)

        for epoch in range(self.epochs):
            # 1. Forward Pass
            linear_model = np.dot(X, self.params['w']) + self.params['b']
            y_pred = self._sigmoid(linear_model)
            
            # 2 & 3. Calculate Loss and Gradients based on the chosen math
            if self.loss_function == 'bce':
                epsilon = 1e-9
                loss = -(1 / n_samples) * np.sum(
                    y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)
                )
                
                # BCE Gradients (Clean and convex)
                grads = {
                    'dw': (1 / n_samples) * np.dot(X.T, (y_pred - y)),
                    'db': (1 / n_samples) * np.sum(y_pred - y)
                }
                
            elif self.loss_function == 'mse':
                loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
                
                # MSE Gradients (The fatal flaw: includes the sigmoid derivative)
                sigmoid_derivative = y_pred * (1 - y_pred)
                error_term = (y_pred - y) * sigmoid_derivative
                
                grads = {
                    'dw': (1 / n_samples) * np.dot(X.T, error_term),
                    'db': (1 / n_samples) * np.sum(error_term)
                }
            else:
                raise ValueError("loss_function must be 'bce' or 'mse'")
            
            # 4. Optimizer updates parameters
            self.params = self.optimizer.update(self.params, grads)
            
            # 5. Track the journey
            self._record_step(epoch_loss=loss, epoch_gradient={'dw': grads['dw'].copy(), 'db': grads['db']})
            
        self.is_fitted = True
        self.training_error = self.loss_history[-1]

    def predict_proba(self, X):
        """Returns the raw probabilities (e.g., 0.87 for class 1)."""
        linear_model = np.dot(X, self.params['w']) + self.params['b']
        return self._sigmoid(linear_model)

    def predict(self, X):
        """Returns the final discrete class prediction (0 or 1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)