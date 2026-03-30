import numpy as np
import matplotlib.pyplot as plt

from glassboxml.datasets import make_moons
from glassboxml.preprocessing import StandardScaler
from glassboxml.models import DBSCAN

def main():
    print("==================================================")
    print("      GlassBox-ML Demo: DBSCAN vs Non-Linearity   ")
    print("==================================================\n")

    # 1. Generate the Interlocking Moons
    print("Generating non-linear 'Moons' dataset (300 points)...")
    X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
    
    # CRITICAL: Distance-based algorithms (like the Epsilon radius) 
    # require scaled data, otherwise the 'circle' becomes a squished oval!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Unleash DBSCAN
    print("Deploying DBSCAN (Epsilon=0.3, Min Neighbors=5)...")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X_scaled)
    print("\n" + dbscan.explain() + "\n")

    # 3. Visualization Showdown
    plt.figure(figsize=(10, 6))
    
    # We use a custom color map to ensure 'Noise' (-1) points show up differently
    labels = dbscan.labels_
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Plot Noise as black X's
            plt.scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                        c='black', marker='x', label='Noise (Ignored)', alpha=0.6)
        else:
            # Plot actual dense clusters
            plt.scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                        color=colors[i], edgecolor='k', s=50, label=f'Cluster {label}')
            
    plt.title("DBSCAN traversing non-linear data via BFS")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add an explanatory text box
    textstr = (
        "Notice how it perfectly maps the winding shapes.\n"
        "Any points sitting out in the middle of nowhere\n"
        "failed the density test and were marked as Noise!"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
