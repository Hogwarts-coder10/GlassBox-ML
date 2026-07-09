import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path to find the core, models, and datasets folders

from glassboxml.models import DecisionTreeClassifier
from glassboxml.datasets._generators import make_blobs

def main():
    print("--- GlassBox-ML Showdown: Gini vs. Entropy ---\n")

    # 1. Generate Non-Linear 2D Data
    print("Generating 2D blob data...")
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.5, random_state=42)

    # 2. Initialize Both Trees (Restricting depth to 3 so we can read the logic)
    print("Initializing trees...")
    tree_gini = DecisionTreeClassifier(max_depth=3, criterion='gini')
    tree_entropy = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    
    # 3. Train Both Models
    print("Growing the Gini tree...")
    tree_gini.fit(X, y)
    
    print("Growing the Entropy tree...\n")
    tree_entropy.fit(X, y)

    # Print the logic for both so you can compare the text output!
    print(tree_gini.explain())
    print("-" * 40 + "\n")
    print(tree_entropy.explain())

    # 4. Map the Decision Boundaries
    print("Mapping the logic boundaries (this takes a second)...")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict with both trees
    Z_gini = tree_gini.predict(grid_points).reshape(xx.shape)
    Z_entropy = tree_entropy.predict(grid_points).reshape(xx.shape)

    # 5. Head-to-Head Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Gini Impurity
    ax1.contourf(xx, yy, Z_gini, alpha=0.3, cmap='brg')
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolor='k', s=40)
    ax1.set_title("Tree 1: Gini Impurity (Faster)")
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: Information Entropy
    ax2.contourf(xx, yy, Z_entropy, alpha=0.3, cmap='brg')
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolor='k', s=40)
    ax2.set_title("Tree 2: Information Entropy (Logarithmic)")
    ax2.set_xlabel("Feature 0")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()