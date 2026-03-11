import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Route Python to your GlassBoxML root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.generators import make_blobs
# Import YOUR K-Means model!
from models.kmeans2 import KMeansClustering 
def main():
    print("==================================================")
    print("      GlassBox-ML Utility: The Elbow Method       ")
    print("==================================================\n")

    # 1. Generate a dataset with exactly 4 hidden clusters
    print("Generating mystery dataset...")
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
    
    # 2. Run K-Means 10 times and record the Inertia
    max_k = 10
    inertias = []
    
    print(f"Running K-Means for K=1 through K={max_k}...")
    for k in range(1, max_k + 1):
        # We suppress print statements in the loop so it doesn't spam your terminal
        model = KMeansClustering(k=k) 
        model.fit(X)
        inertias.append(model.inertia_)
        print(f"  -> K={k} | Inertia: {model.inertia_:.2f}")

    # 3. Plot the Elbow Graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("The Elbow Method: Finding the Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    
    # Highlight the "Elbow"
    plt.axvline(x=4, color='red', linestyle='--', label='The Elbow (Optimal K=4)')
    
    plt.xticks(range(1, max_k + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()