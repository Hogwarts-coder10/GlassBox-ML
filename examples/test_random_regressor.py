import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decision_tree import DecisionTreeRegressor
from models.random_forest import RandomForestRegressor

def main():
    print("--- GlassBox-ML Demo: Random Forest Regressor ---\n")

    # 1. Generate the same noisy continuous data
    print("Generating noisy continuous data...")
    rng = np.random.RandomState(42)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16)) # Add the heavy noise

    # 2. Train the Models (Both with max_depth=5)
    print("Training a Single Tree (Prone to overfitting)...")
    single_tree = DecisionTreeRegressor(max_depth=5)
    single_tree.fit(X, y)

    print("\nTraining a Random Forest (20 trees to average out the noise)...")
    forest = RandomForestRegressor(n_trees=20, max_depth=5)
    forest.fit(X, y)

    print("\n" + forest.explain() + "\n")

    # 3. Predict to draw the lines
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_tree = single_tree.predict(X_test)
    y_forest = forest.predict(X_test)

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    plt.scatter(X, y, s=40, edgecolor="black", c="darkorange", label="Raw Noisy Data")
    
    # Plot the Single Tree (Jagged) vs the Forest (Smooth)
    plt.plot(X_test, y_tree, color="red", label="Single Tree (Overfitting Staircase)", linewidth=2, alpha=0.5)
    plt.plot(X_test, y_forest, color="blue", label="Random Forest (Smoothed Average)", linewidth=3)
    
    plt.xlabel("Feature X")
    plt.ylabel("Target y (Continuous)")
    plt.title("Random Forest Regression: Averaging out the Noise")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    textstr = "Notice how the Blue line (Forest) ignores the massive orange outliers\nbecause the 'majority of the trees' knew it was just noise!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()