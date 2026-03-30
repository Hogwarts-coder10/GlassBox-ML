import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import KMeansClustering 
from glassboxml.datasets import make_blobs

def main():
    print("--- GlassBox-ML Demo: K-Means Clustering ---\n")

    # 1. Generate UNLABELED Data
    # We ask for 4 centers, but we THROW AWAY the 'y' labels!
    print("Generating raw, unlabeled spatial data...")
    X, _ = make_blobs(n_samples=400, n_features=2, centers=4, cluster_std=1.5, random_state=42)

    # 2. Train the Unsupervised Model
    # We tell it to look for 4 clusters. Notice we only pass X!
    print("Unleashing K-Means to find the centers of gravity...")
    kmeans = KMeansClustering(k=4, max_iters=100)
    kmeans.fit(X)
    
    print("\n" + kmeans.explain() + "\n")

    # 3. Visualization
    plt.figure(figsize=(10, 7))
    
    # Plot the data points, colored by the clusters K-Means INVENTED
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6, edgecolor='k', s=50)
    
    # Plot the final Centroids as massive Red X's
    centroids = kmeans.centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=250, edgecolor='black', 
                linewidth=2, label='Final Centroids')
    
    plt.title("K-Means Clustering: Finding Structure in Chaos")
    plt.xlabel("Feature 0 (e.g., Longitude)")
    plt.ylabel("Feature 1 (e.g., Latitude)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    textstr = "The model was never told where the clusters were.\nIt dropped 4 random anchors and mathematically\npulled them to the centers of gravity!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
