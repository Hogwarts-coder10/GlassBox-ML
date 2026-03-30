import numpy as np
import matplotlib.pyplot as plt

from glassboxml.preprocessing import StandardScaler
from glassboxml.models import LDA
from glassboxml.datasets import make_blobs

def main():
    print("--- GlassBox-ML Test: Linear Discriminant Analysis (LDA) ---\n")

    # 1. Generate 3D Data with 3 distinct classes
    print("Generating 3D synthetic data...")
    X_3d, y = make_blobs(n_samples=300, n_features=3, centers=3, cluster_std=1.5, random_state=42)

    # 2. Scale the Data (Always a good practice for distance/variance algorithms)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)

    # 3. Apply LDA
    # We have 3 classes, so max components = 3 - 1 = 2.
    print("\nApplying LDA (Reducing from 3D to 2D)...")
    lda = LDA(n_components=2)
    
    # Fit and transform
    X_2d = lda.fit_transform(X_scaled, y)

    # 4. Open the Glass Box (Diagnostics & Explanation)
    print("\nDiagnostics:")
    report = lda.diagnose()
    for key, value in report.items():
        if isinstance(value, list) and value:
            for item in value:
                print(f"  ⚠️ {item}")
        else:
            print(f"  {key}: {value}")

    print("\n" + lda.explain() + "\n")

    # 5. Visualization
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: The Original 3D Data
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                           c=y, cmap='brg', alpha=0.7, edgecolor='k')
    ax1.set_title("Original 3D Data (3 Classes)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_zlabel("Feature 3")

    # Plot 2: The 2D LDA Projection
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                           c=y, cmap='brg', alpha=0.7, edgecolor='k')
    ax2.set_title("2D LDA Projection")
    ax2.set_xlabel("Linear Discriminant 1 (Most Separability)")
    ax2.set_ylabel("Linear Discriminant 2")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
