import numpy as np
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decision_tree import DecisionTreeClassifier
from datasets.generators import make_blobs
from diagnostics.learning_curve import LearningCurveAnalyzer

def main():
    print("--- GlassBox-ML Diagnostic: Learning Curves ---\n")
    
    # 1. Generate a noisy dataset that is easy to overfit
    print("Generating noisy classification data...")
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2.5, random_state=42)
    
    # 2. Initialize a model prone to overfitting (An unconstrained Decision Tree)
    print("Initializing Unconstrained Decision Tree...")
    # By setting max_depth=50, we give the tree the power to memorize every single data point.
    model = DecisionTreeClassifier(max_depth=50)
    
    # 3. Generate the Learning Curve
    train_sizes, train_errors, val_errors = LearningCurveAnalyzer.generate_curve(
        model=model, 
        X=X, 
        y=y, 
        is_classifier=True, 
        n_splits=6, 
        test_size=0.2
    )
    
    # 4. Plot the Diagnosis
    print("\nRendering Bias-Variance Diagnostic Plot...")
    LearningCurveAnalyzer.plot_curve(
        train_sizes, 
        train_errors, 
        val_errors, 
        title="Unconstrained Decision Tree (Prone to Overfitting)"
    )

if __name__ == "__main__":
    main()