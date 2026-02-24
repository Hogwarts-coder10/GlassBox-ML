import numpy as np
from abc import ABC, abstractmethod

class GlassBoxModel(ABC):
    """"
    The foundational class for all GlassBoxML models.
    Enforces the philosophy of transparency by requiring models to track
    their learning process, assumptions, and failure modes.
    """

    def __init__(self):
        # Tracking the learning journey
        self.loss_history = []
        self.gradient_history = []

        # Diagnostics
        self.failure_modes = []
        self.is_fitted = False
        self.training_error = None


    @abstractmethod
    def fit(self,X,y):
        """ Train the model and record the optimization steps """
        pass


    @abstractmethod
    def predict(self,X):
        """ Make Predictions using the learned parameters """
        pass

    @abstractmethod
    def check_assumptions(self,X,y):
        """ 
        Inspect the dataset against the mathematical assumptions of the data 
        eg: checking for multi-collinearity in Linear Regression 
        """
        pass

    def _record_step(self,epoch_loss,epoch_gradient = None):
        """
        Helper Method to log training internals at each step/epoch
        """

        self.loss_history.append(epoch_loss)
        if epoch_gradient is not None:
            self.gradient_history.append(epoch_gradient)


    def diagnose(self):
        """
        Returns a human readable summary about what the model learned and 
        where it struggled
        """

        if not self.is_fitted:
            return "Model is not fitted yet, call fit(X,y) first"
        
        report = {
            'Final Training Error' : self.training_error,
            'Optimization Steps' : len(self.loss_history),
            'Logged Failure Modes' : self.failure_modes if self.failure_modes else "None detected",
        }

        return report
    
    def generalization_estimate(self):
        """
        Placeholder for learning theory metrics (e.g., Rademacher complexity, 
        VC dimension bounds, or simple train/val gap heuristics)
        """

        if not self.is_fitted:
            raise ValueError("Fit the model before estimating generalization.")
        
        else:
            return "Generalization bounds not yet computed for this model."


