import numpy as np
import sys, os
import time

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model_selection import train_test_split
from core.preprocessing import StandardScaler
from metrics.regression import mean_squared_error, r2_score

# Import the GlassBox Arsenal (Regression Edition)
from models.decision_tree import DecisionTreeRegressor
from models.random_forest import RandomForestRegressor
from models.svm import SupportVectorRegressor
from models.gradientboost import GradientBoostingRegressor

def main():
    print("==================================================")
    print("      GlassBox-ML: Regressor Leaderboard          ")
    print("==================================================\n")

    # 1. Generate Non-Linear Continuous Data
    print("Generating complex continuous wave data (800 rows)...")
    rng = np.random.RandomState(42)
    X = np.sort(10 * rng.rand(800, 1), axis=0)
    # Create a messy, curving mathematical wave
    y = np.sin(X).ravel() + np.sin(2 * X).ravel() + (X.ravel() * 0.2)
    y += 0.5 * (0.5 - rng.rand(800)) # Add random static noise

    # 2. Scale and Split! (Always scale for the SVR!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )

    # 3. Initialize the Models
    models = {
        "Decision Tree (Max Depth=5)": DecisionTreeRegressor(max_depth=5),
        "Random Forest (20 Trees)": RandomForestRegressor(n_trees=20, max_depth=5),
        "Support Vector Regressor (SVR)": SupportVectorRegressor(learning_rate=0.01, epsilon=0.1, n_iters=1000),
        "Gradient Boosting (50 Trees)": GradientBoostingRegressor(n_trees=50, learning_rate=0.1, max_depth=3)
    }

    # 4. The Gauntlet
    print("\nRunning the 80/20 split gauntlet...\n")
    print(f"{'Algorithm Name':<32} | {'MSE (Lower=Better)':<18} | {'R² Score (Closer to 1=Better)'}")
    print("-" * 85)

    results = []

    for name, model in models.items():
        # Hide the verbose training printouts for a clean table
        sys.stdout = open(os.devnull, 'w') 
        
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        
        sys.stdout = sys.__stdout__ # Restore printing
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        train_time = end_time - start_time
        
        # We save R2 as the first item in the tuple so we can sort by it to find the winner
        results.append((r2, mse, name, train_time))
        
        print(f"{name:<32} | {mse:<18.4f} | {r2:.4f}")

    # Print the Winner
    results.sort(reverse=True) 
    print("-" * 85)
    print(f"\n🏆 WINNER: {results[0][2]} explaining {results[0][0]:.2%} of the variance!")

if __name__ == "__main__":
    main()