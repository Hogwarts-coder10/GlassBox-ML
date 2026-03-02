import numpy as np
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diagnostics.inspector import DataInspector
from datasets.generators import make_regression, make_blobs

def main():
    print("--- GlassBox-ML Demo: Data Diagnostics ---\n")
    
    # --- TEST 1: Data Leakage ---
    print("[Test 1: Detecting Data Leakage]")
    X, y, _, _ = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    
    # 🚨 INTENTIONAL TRAP: We sneak the 'y' values back into the feature matrix!
    # We multiply it by 2 to pretend it's a different metric, but the correlation is identical.
    leaked_feature = y * 2.0 
    X_trapped = np.column_stack((X, leaked_feature))
    
    feature_names = ["Square_Footage", "Num_Bedrooms", "Age_of_Home", "Taxes_Paid_On_Value"]
    
    leakage_report = DataInspector.detect_data_leakage(X_trapped, y, feature_names=feature_names)
    print(leakage_report)
    
    # --- TEST 2: Class Imbalance ---
    print("\n[Test 2: Detecting Class Imbalance]")
    # Generate 95 healthy patients and 5 sick patients
    y_imbalanced = np.concatenate([np.zeros(95), np.ones(5)])
    
    imbalance_report = DataInspector.detect_imbalance(y_imbalanced)
    print(imbalance_report)

if __name__ == "__main__":
    main()