import numpy as np
import matplotlib.pyplot as plt

# Setup path
from glassboxml.models import AdaBoostClassifier
from glassboxml.datasets import make_moons  

def main():
    print("--- GlassBox-ML Demo: AdaBoost ---\n")

    # 1. Generate non-linear 'Moons' data
    print("Generating interlocking moon data from scratch...")
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

    # 2. Train AdaBoost
    print("Training 30 sequential weak stumps...")
    boost = AdaBoostClassifier(n_clf=30)
    boost.fit(X, y)
    
    print("\n" + boost.explain() + "\n")

    # 3. Visualization
    print("Mapping the sequential boundaries...")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = boost.predict(grid_points).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=40)
    
    plt.title("AdaBoost: 30 Weak Stumps chained together!")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    
    textstr = "Notice the blocky 'staircase' edges. Those are the individual\nstraight lines drawn by the 30 1-depth Decision Stumps!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
