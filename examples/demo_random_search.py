import numpy as np

from glassboxml.datasets import make_blobs
from glassboxml.tuning import RandomizedSearchCV
# Note: Adjust this import path to match whatever you named your Decision Tree file!
from glassboxml.models import DecisionTreeClassifier 

def main():
    print("==================================================")
    print(" GlassBox-ML Demo: CPU-Safe Hyperparameter Tuning ")
    print("==================================================\n")

    # 1. Generate a tricky, overlapping dataset so the model has to work for it
    print("Generating a messy 4-cluster dataset (500 points)...")
    X, y = make_blobs(n_samples=500, centers=4, cluster_std=2.5, random_state=42)

    # 2. Instantiate the blank, default model
    base_tree = DecisionTreeClassifier()

    # 3. Define the massive search space
    # (5 depths) * (4 split rules) = 20 possible combinations
    param_distributions = {
        'max_depth': [3, 5, 8, 12, 20],
        'min_samples_split': [2, 5, 10, 20]
    }

    # 4. Unleash the CPU-Safe Search Engine
    # We restrict it to n_iter=5. It will randomly pick 5 of the 20 universes.
    print("\nDeploying RandomizedSearchCV (Testing exactly 5 combos)...")
    tuner = RandomizedSearchCV(
        model=base_tree, 
        param_distributions=param_distributions, 
        n_iter=5, 
        random_state=99
    )

    # 5. Let it hunt for the highest accuracy
    best_tree = tuner.fit(X, y)

    print("\n==================================================")
    print("  Search Complete! Winning model is locked in.    ")
    print("==================================================")
    
    # Prove the returned model is fully functional
    if hasattr(best_tree, 'explain'):
        print(best_tree.explain())
    else:
        print(f"Optimal tree depth selected and ready for deployment!")

if __name__ == "__main__":
    main()