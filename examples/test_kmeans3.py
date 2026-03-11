import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Route Python to your GlassBoxML root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.generators import make_circles
# Import your v2 K-Means!
from models.kmeans2 import KMeansClustering # (Adjust import name if necessary)

def main():
    print("==================================================")
    print("    GlassBox-ML Demo: K-Means vs The Bullseye     ")
    print("==================================================\n")

    # 1. Generate the Concentric Circles
    print("Generating 'Circles' dataset (400 points)...")
    # factor=0.5 means the inner circle is half the size of the outer circle
    X, _ = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)

    # 2. Unleash K-Means v2 (We know there are 2 circles, so K=2)
    print("Deploying K-Means (K=2, trying 10 initializations)...")
    # Using your exact v2 API here!
    model = KMeansClustering(k=2, max_iters=100, n_init=10)
    model.fit(X)

    # 3. Visualization Showdown
    plt.figure(figsize=(8, 8)) # Square figure so the circles don't look like ovals
    
    # Plot the clustered points
    unique_labels = np.unique(model.labels_)
    colors = ['#8800ff', '#00cfb5'] # Let's make it look sharp
    
    for i, label in enumerate(unique_labels):
        plt.scatter(X[model.labels_ == label, 0], X[model.labels_ == label, 1], 
                    color=colors[i], edgecolor='k', s=50, label=f'Cluster {label}')
        
    # Plot the exact coordinates K-Means chose for its anchors
    centroids = np.array(model.centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                color='red', marker='X', s=200, edgecolor='black', 
                linewidth=2, label='K-Means Anchors')
            
    plt.title("The Bullseye Failure: K-Means slicing through empty space")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add an explanatory text box
    textstr = (
        "Notice the straight-line decision boundary.\n"
        "Because K-Means only understands Euclidean\n"
        "distance, it cannot separate a ring from its core."
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()