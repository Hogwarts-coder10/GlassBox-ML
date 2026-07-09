import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models._decision_tree import DecisionTreeRegressor

def main():
    print("--- GlassBox-ML Demo: Decision Tree Regressor ---\n")

    # 1. Generate a noisy Sine Wave (Non-linear continuous data)
    print("Generating noisy continuous data...")
    rng = np.random.RandomState(42)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16)) # Add some heavy noise to every 5th point

    # 2. Train two trees with different depths
    print("Training Regressors (Depth 2 vs Depth 5)...")
    tree_shallow = DecisionTreeRegressor(max_depth=2)
    tree_deep = DecisionTreeRegressor(max_depth=5)
    
    tree_shallow.fit(X, y)
    tree_deep.fit(X, y)

    print("\n" + tree_shallow.explain() + "\n")

    # 3. Predict across the whole X-axis to draw the lines
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_shallow = tree_shallow.predict(X_test)
    y_deep = tree_deep.predict(X_test)

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot the original noisy data
    plt.scatter(X, y, s=40, edgecolor="black", c="darkorange", label="Raw Noisy Data")
    
    # Plot the tree predictions (The Staircases!)
    plt.plot(X_test, y_shallow, color="cornflowerblue", label="Shallow Tree (max_depth=2)", linewidth=3)
    plt.plot(X_test, y_deep, color="yellowgreen", label="Deep Tree (max_depth=5)", linewidth=2)
    
    plt.xlabel("Feature X")
    plt.ylabel("Target y (Continuous)")
    plt.title("Decision Tree Regression: The Staircase Effect")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    textstr = "Notice how the Deep Tree (green) overfits by jumping up and\ndown to perfectly memorize the noisy orange outliers!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()