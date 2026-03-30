import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import GradientBoostingClassifier
from glassboxml.datasets import make_donut

def main():
    print("--- GlassBox-ML Demo: Gradient Boosting Classifier ---\n")

    # 1. Generate non-linear 'Donut' data (Impossible for a straight line)
    print("Generating 2D donut data from scratch...")
    X, y = make_donut(n_samples=250, noise=0.15, random_state=42)

    # 2. Train the GBM Classifier
    print("Training the ensemble to map the probability gradients...")
    gbm = GradientBoostingClassifier(n_trees=50, learning_rate=0.1, max_depth=3)
    gbm.fit(X, y)
    
    print("\n" + gbm.explain() + "\n")

    # 3. Visualization
    print("Mapping the decision boundaries...")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Let's map the actual PROBABILITIES to see the gradient, not just the hard prediction!
    Z = gbm.predict_proba(grid_points).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    
    # Use contourf to show the smooth probability gradients (Log-Loss in action!)
    contour = plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm', levels=20)
    plt.colorbar(contour, label="Probability of Class 1")
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
    
    plt.title("Gradient Boosting Classifier: Probability Gradients")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    
    textstr = "Notice how the color smoothly transitions! The model didn't just\ndraw a line; it mapped the literal probability of being in the outer ring."
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
