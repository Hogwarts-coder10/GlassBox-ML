import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import RandomForestClassifier
from glassboxml.datasets import make_blobs 
from glassboxml.diagnostics import FairnessAnalyzer, LearningCurveAnalyzer

def main():
    print("--- GlassBox-ML: Random Forest & Fairness Demo ---\n")
    
    # 1. Generate Noisy Data
    print("Generating noisy classification data...")
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2.5, random_state=42)
    
    # Generate a fake "sensitive attribute" (e.g., 0 = Group A, 1 = Group B)
    # Let's purposefully make the model perform worse on Group B by adding noise to their labels
    sensitive_attr = np.random.choice([0, 1], size=len(y))
    y_biased = y.copy()
    noise_mask = (sensitive_attr == 1) & (np.random.rand(len(y)) < 0.3) # 30% of Group B gets corrupted labels
    y_biased[noise_mask] = np.random.choice([0, 1, 2, 3], size=np.sum(noise_mask))
    
    # 2. Train the Random Forest
    print("\nInitializing Random Forest (15 trees)...")
    rf = RandomForestClassifier(n_trees=15, max_depth=5)
    
    # 3. Generate the Learning Curve to prove we fixed Overfitting!
    train_sizes, train_errors, val_errors = LearningCurveAnalyzer.generate_curve(
        model=rf, X=X, y=y_biased, is_classifier=True, n_splits=5, test_size=0.2
    )
    
    # 4. Run the Fairness Analyzer
    print("\nEvaluating Model Fairness...")
    rf.fit(X, y_biased)
    y_pred = rf.predict(X)
    
    fairness_report = FairnessAnalyzer.check_disparate_impact(
        y_true=y_biased, 
        y_pred=y_pred, 
        sensitive_attribute=sensitive_attr,
        group_names=["Group A", "Group B (Noisy)"]
    )
    print("\n" + fairness_report + "\n")
    print(rf.explain())

    # 5. Plot the cured Learning Curve
    LearningCurveAnalyzer.plot_curve(
        train_sizes, train_errors, val_errors, 
        title="Random Forest (Bagging reduces Variance!)"
    )

if __name__ == "__main__":
    main()
