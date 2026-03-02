import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path to find the core, models, and datasets folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decision_tree import DecisionTreeClassifier
from datasets.generators import make_blobs

def main():
    print("--- GlassBox-ML Demo: Decision Tree Classifier ---\n")

    # 1. Generate Non-Linear 2D Data (3 distinct classes)
    print("Generating 2D blob data...")
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.2, random_state=42)

    # Notice: NO SCALING! 
    # Decision Trees are entirely scale-invariant. 
    # They just look for threshold splits, so StandardScaler isn't necessary.

    # 2. Initialize and Train the Model
    # We restrict max_depth to 3 so we can actually read the logic rules it prints!
    print("Initializing Decision Tree (max_depth=3)...")
    tree = DecisionTreeClassifier(max_depth=3)
    
    print("Growing the tree...\n")
    tree.fit(X, y)

    # 3. Open the Glass Box (Diagnostics & Explanation)
    print("Diagnostics:")
    report = tree.diagnose()
    for key, value in report.items():
        if isinstance(value, list) and value:
            for item in value:
                print(f"  ⚠️ {item}")
        else:
            print(f"  {key}: {value}")

    # Print the recursive logic gates!
    print("\n" + tree.explain() + "\n")

    # 4. Map the Decision Boundaries (The "City Grid")
    print("Mapping the logic boundaries (this takes a second)...")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = tree.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 5. Visualization
    plt.figure(figsize=(10, 8))
    
    # Draw the blocky decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='brg')
    
    # Plot training data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolor='k', s=40)
    
    plt.title("Decision Tree Classification (max_depth=3)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Add a little text box explaining the shapes
    textstr = "Notice the orthogonal (blocky) boundaries.\nTrees split one axis at a time!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()