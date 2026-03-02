import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing import StandardScaler
from models.pca import PCA
from models.lda import LDA
from datasets.generators import make_stretched_blobs

def main():
    print("--- GlassBox-ML Showdown: PCA vs. LDA ---\n")

    # 1. Generate "Trick" 3D Data using our GlassBox generator!
    print("Generating stretched 'trick' dataset...")
    X_data, y = make_stretched_blobs(n_samples_per_class=150, random_state=42)

    # 2. Train PCA (Unsupervised)
    print("\nTraining PCA (Reducing 3D -> 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    print(pca.explain() + "\n")

    # 3. Train LDA (Supervised)
    print("-" * 50)
    print("Training LDA (Reducing 3D -> 2D)...")
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_data, y)
    print(lda.explain() + "\n")

    # 4. Epic Visualization
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Original 3D Data
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(X_data[:, 0], X_data[:, 1], X_data[:, 2], 
                           c=y, cmap='brg', alpha=0.6, edgecolor='k')
    ax1.set_title("Original 3D Data (The Trap)")
    ax1.set_xlabel("Feature X (Massive Variance)")
    ax1.set_ylabel("Feature Y (Class Separation)")
    ax1.set_zlabel("Feature Z")

    # Plot 2: PCA Projection
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=y, cmap='brg', alpha=0.6, edgecolor='k')
    ax2.set_title("PCA Projection (Unsupervised)")
    ax2.set_xlabel("Principal Component 1 (Fell for X-axis)")
    ax2.set_ylabel("Principal Component 2")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: LDA Projection
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(X_lda[:, 0], X_lda[:, 1], 
                           c=y, cmap='brg', alpha=0.6, edgecolor='k')
    ax3.set_title("LDA Projection (Supervised)")
    ax3.set_xlabel("Linear Discriminant 1 (Found the Y-axis!)")
    ax3.set_ylabel("Linear Discriminant 2")
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()