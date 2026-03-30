import numpy as np
import matplotlib.pyplot as plt

from glassboxml.datasets import make_circles
from glassboxml.preprocessing import StandardScaler

from glassboxml.models import DBSCAN 

def main():
    print("==================================================")
    print("   GlassBox-ML Demo: DBSCAN vs The Bullseye       ")
    print("==================================================\n")

    # 1. Generate the exact same Concentric Circles
    print("Generating 'Circles' dataset (400 points)...")
    X, _ = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)
    
    # CRITICAL: Remember your check_assumptions() rule! 
    # DBSCAN needs scaled data so the Epsilon radius stays a perfect circle.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Unleash DBSCAN
    # We use a tighter Epsilon here (0.15) because the rings are close together
    print("Deploying DBSCAN (Epsilon=0.15, Min Neighbors=5)...")
    dbscan = DBSCAN(eps=0.15, min_samples=5)
    dbscan.fit(X_scaled)
    
    print("\n" + dbscan.explain() + "\n")

    # 3. Visualization Showdown
    plt.figure(figsize=(8, 8))
    
    labels = dbscan.labels_
    unique_labels = np.unique(labels)
    colors = ['#8800ff', '#00cfb5', '#ff0055', '#ffaa00'] # Extra colors just in case
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points get the black X
            plt.scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                        c='black', marker='x', label='Noise', alpha=0.6)
        else:
            # Valid density chains get colored
            color_idx = i % len(colors)
            plt.scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                        color=colors[color_idx], edgecolor='k', s=50, label=f'Cluster {label}')
            
    plt.title("DBSCAN Victory: Mapping shapes via Breadth-First Search")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    textstr = (
        "Unlike K-Means, DBSCAN has no concept of 'center'.\n"
        "It simply follows the continuous path of high density,\n"
        "allowing it to perfectly isolate the inner core\n"
        "from the outer ring!"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
