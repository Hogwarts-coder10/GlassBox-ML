import numpy as np
import sys, os


from glassboxml.core import Pipeline
from glassboxml.feature_extraction.text import TfidfVectorizer
from glassboxml.preprocessing import LabelEncoder

# A mock DummyClassifier just so we can test the predict chain
# without importing your actual heavy RandomForest right now.
from glassboxml.core._base_model import GlassBoxModel

class DummyPredictor(GlassBoxModel):
    def fit(self, X, y=None):
        self.is_fitted = True
        return self
    def predict(self, X):
        # Just predicts class 0 for everything
        return np.zeros(X.shape[0], dtype=int)
    def explain(self):
        return "--- Dummy Predictor ---\nArchitecture: Outputs 0 regardless of input."

def main():
    print("==================================================")
    print(" GlassBox-ML Demo: The Architecture Pipeline      ")
    print("==================================================\n")

    # 1. Raw Medical Logs (No manual formatting required!)
    raw_logs = [
        "Inventory alert: The hospital is running low on epinephrine.",
        "Routine check: All systems normal.",
    ]
    raw_targets = ["Critical", "Stable"]

    # 2. Build the System Chassis
    # Look how clean this API is.
    med_pipeline = Pipeline([
        ('nlp_extractor', TfidfVectorizer(mode="sparse")),
        ('classifier', DummyPredictor())
    ])

    target_encoder = LabelEncoder()

    # 3. Train the entire system in two lines of code
    print("Training pipeline...\n")
    y_encoded = target_encoder.fit_transform(raw_targets)
    med_pipeline.fit(raw_logs, y_encoded)

    # 4. Generate the recursive architecture map
    print(med_pipeline.explain())

if __name__ == "__main__":
    main()
