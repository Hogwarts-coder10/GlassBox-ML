import numpy as np
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model_selection import train_test_split
from metrics.classification import accuracy_score, classification_report
from models.random_forest import RandomForestClassifier
from datasets.generators import make_blobs

def main():
    print("--- GlassBox-ML: The Train/Test Split ---\n")

    # 1. Generate a large dataset
    print("Generating 1,000 data points...")
    X, y = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.0, random_state=42)

    # 2. Split the data! (80% Training, 20% Testing)
    print("Splitting data into the Training Vault and Testing Vault...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    print(f"  -> Training Set Shape: {X_train.shape[0]} rows")
    print(f"  -> Testing Set Shape : {X_test.shape[0]} rows\n")

    # 3. Train strictly on the Training Set
    print("Training Random Forest on the 80% training data...")
    rf = RandomForestClassifier(n_trees=15, max_depth=5)
    rf.fit(X_train, y_train)

    # 4. Predict strictly on the Testing Set
    print("Administering the blind test...")
    y_pred = rf.predict(X_test)

    # 5. Grade the test
    print("\n" + classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()