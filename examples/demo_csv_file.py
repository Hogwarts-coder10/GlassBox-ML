import numpy as np
import pandas as pd
import  os

# Setup path
from glassboxml.models import RandomForestClassifier
from glassboxml.diagnostics import FairnessAnalyzer

def main():
    print("--- GlassBox-ML: Real CSV Data Pipeline ---\n")
    
    # 1. Load the CSV using Pandas
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_inventory.csv')
    print(f"Loading dataset from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("🚨 Error: Could not find medical_inventory.csv. Did you create it in the data/ folder?")
        return

    print(f"Dataset loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    print("First 3 rows:")
    print(df.head(3).to_string())
    print("-" * 50)

    # 2. Separate Features (X) from Target (y)
    # We want to predict 'needs_restock' (the last column)
    target_column = 'needs_restock'
    
    # Convert to NumPy arrays for GlassBoxML!
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    feature_names = df.drop(columns=[target_column]).columns.tolist()

    # 3. Train the Random Forest
    print("\nTraining Random Forest on inventory data...")
    rf = RandomForestClassifier(n_trees=10, max_depth=3)
    rf.fit(X, y)
    
    # 4. Make a Real-World Prediction
    print("\n[Simulating a live database query]")
    # Imagine retrieving a new item: Usage=25.0/day, Stock=30, LeadTime=3 days, Critical=1
    new_item = np.array([[25.0, 30.0, 3.0, 1.0]])
    
    prediction = rf.predict(new_item)[0]
    confidence_note = "Model vote cast."
    
    print(f"New Item Metrics: {new_item[0]}")
    if prediction == 1:
        print("🚨 PREDICTION: RESTOCK IMMEDIATELY (Class 1)")
    else:
        print("✅ PREDICTION: STOCK LEVELS HEALTHY (Class 0)")
        
    print("\n" + rf.explain())

if __name__ == "__main__":
    main()