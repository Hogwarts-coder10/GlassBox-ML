import numpy as np
import time

from glassboxml.core import train_test_split
from glassboxml.preprocessing import StandardScaler
from glassboxml.core import GradientDescent 
from glassboxml.metrics import accuracy_score
from glassboxml.datasets import make_moons

from glassboxml.models import (
    LogisticRegression,
    KNNClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    SVM,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
def main():
    print("==================================================")
    print("      GlassBox-ML: Classifier Leaderboard         ")
    print("==================================================\n")

    # 1. Generate Non-Linear Data (The Moons)
    print("Generating non-linear 'Moons' dataset (800 rows)...")
    X, y = make_moons(n_samples=800, noise=0.2, random_state=42)

    # 2. Scale and Split!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )

    # 3. Initialize the Models
    # We will test how well they handle curves!
    models = {
        "Logistic Regression (Linear)": LogisticRegression(optimizer=GradientDescent(learning_rate=0.1),epochs=1000),
        "K-Nearest Neighbors (k=5)": KNNClassifier(k=5),
        "Decision Tree (Max Depth=5)": DecisionTreeClassifier(max_depth=5),
        "Random Forest (20 Trees)": RandomForestClassifier(n_trees=20, max_depth=5),
        "Linear SVM (Hard Margin)": SupportVectorMachine(lambda_param=0.001, n_iters=500),
        "AdaBoost (30 Stumps)": AdaBoostClassifier(n_clf=30),
        "Gradient Boosting (50 Trees)": GradientBoostingClassifier(n_trees=50)
    }

    # 4. The Gauntlet
    print("\nRunning the 80/20 split gauntlet...\n")
    print(f"{'Algorithm Name':<30} | {'Test Accuracy':<15} | {'Training Time'}")
    print("-" * 65)

    results = []
    
    for name, model in models.items():
        # Hide the verbose print statements from the models for a clean table
        sys.stdout = open(os.devnull, 'w') 
        
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        
        sys.stdout = sys.__stdout__ # Restore printing
        
        acc = accuracy_score(y_test, y_pred)
        train_time = end_time - start_time
        results.append((acc, name, train_time))
        
        print(f"{name:<30} | {acc:>13.2%} | {train_time:>5.3f} sec")

    # Print the Winner
    results.sort(reverse=True) # Sort by accuracy
    print("-" * 65)
    print(f"\n🏆 WINNER: {results[0][1]} with {results[0][0]:.2%} accuracy!")

if __name__ == "__main__":
    main()
