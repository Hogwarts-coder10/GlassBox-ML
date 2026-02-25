import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Base class for all optimization algorithms in GlassBoxML."""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    @abstractmethod
    def update(self, params, grads):
        """Updates and returns the new parameters based on the gradients."""
        pass

class GradientDescent(Optimizer):
    """Standard Batch Gradient Descent."""
    def update(self, params, grads):
        # The classic update rule: θ = θ - α∇J(θ)
        params['w'] -= self.lr * grads['dw']
        params['b'] -= self.lr * grads['db']
        return params

class Momentum(Optimizer):
    """Gradient Descent with Momentum to accelerate learning."""
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = None

    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = {key: np.zeros_like(val) for key, val in params.items()}
        
        # v = βv + (1-β)∇J
        self.velocities['w'] = self.beta * self.velocities['w'] + (1 - self.beta) * grads['dw']
        self.velocities['b'] = self.beta * self.velocities['b'] + (1 - self.beta) * grads['db']
        
        params['w'] -= self.lr * self.velocities['w']
        params['b'] -= self.lr * self.velocities['b']
        return params