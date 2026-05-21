import numpy as np
from typing import Any, Union

from glassboxml.core._base_model import GlassBoxModel

class Pipeline(GlassBoxModel):
    """
    Chains multiple Transformers and a final Predictor into a single cohesive unit.

    Ensures that the exact same sequence of data transformations applied
    during training (.fit) is automatically applied during inference (.predict).
    """

    def __init__(self,steps: list[tuple[str,GlassBoxModel]]):
        super().__init__()

        if not steps or not isinstance(steps,list):
            raise ValueError("Pipeline requires a list of (name,model) tuples")

        # Validate all steps before the .transform() method
        for name,step in steps[:-1]:
            if not hasattr(step, 'transform'):
                raise TypeError(f"Intermediate step '{name}' must implement .transform().")


        self.steps = steps
        self.is_fitted = False



    def fit(self, X: Any, y: Any = None) -> Pipeline:
        """
        Passes data throught the pipeline, fitting and transforming at each step,
        until it reaches the final esitmator which is only fitted
        """

        X_transformed = X
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)

        final_steps = self.steps[-1][1]
        final_steps.fit(X_transformed,y)

        self.is_fitted = True
        return self


    def transform(self, X: Any) -> Any:
        """
        Applies transforms sequentially. Fails if the final step is a Predictor.
        """

        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")

        X_transformed = X
        for name, step in self.steps:
            if not hasattr(step, 'transform'):
                    raise TypeError(f"Step '{name}' does not support .transform().")
            X_transformed = step.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """
        Fits all steps and transforms the data in one sequential pass.
        Strictly requires the final step in the pipeline to be a Transformer.
        """
        X_transformed = X

        # Route through all intermediate transformers
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)

        # The final step MUST support transform to use fit_transform on the pipeline
        final_step = self.steps[-1][1]
        if not hasattr(final_step, 'transform'):
            raise TypeError(
                f"Final step '{self.steps[-1][0]}' does not support .transform(). "
                "Pipeline.fit_transform() requires all steps to be Transformers."
            )

        # Apply fit_transform on the final component
        X_transformed = final_step.fit_transform(X_transformed, y)

        self.is_fitted = True
        return X_transformed

    def predict(self, X: Any) -> Any:
        """
        Transforms the data sequentially, then predicts with the final estimator.
        """

        if not self.is_fitted:
            raise ValueError("Call fit() before predict().")

        X_transformed = X

        # Transform the incoming data through the chain
        for name, step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)

        # Pass the mathematically formatted data to the final engine
        final_step = self.steps[-1][1]
        if not hasattr(final_step, 'predict'):
                raise TypeError(f"Final step '{self.steps[-1][0]}' does not support .predict().")

        return final_step.predict(X_transformed)

    def explain(self) -> str:
        if not self.is_fitted:
            return "Pipeline hasn't been fitted yet"


        explanation = "========================================\n"
        explanation += " GlassBox Explanation: Execution Pipeline \n"
        explanation += "========================================\n"

        for i, (name, step) in enumerate(self.steps):
            explanation += f"\n[Step {i + 1}: {name}]\n"
            # Indent the explanation of the inner component for readability
            step_exp = step.explain().replace("\n", "\n    ")
            explanation += f"    {step_exp}\n"

            if i < len(self.steps) - 1:
                explanation += "          |\n          v\n"


        return explanation
