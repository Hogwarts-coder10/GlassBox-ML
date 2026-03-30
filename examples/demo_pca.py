import numpy as np
import matplotlib.pyplot as plt

from glassboxml.preprocessing import StandardScaler
from glassboxml.models import PCA

def main():
    print("--- GlassBox-ML Demo: Principal Component Analysis (PCA) ---\n")

    # 1. Generate 3D Data (A flat plane tilted in 3D space with slight noise)
    np.random.seed(42)
    n_samples = 300
    
    # Feature 1 & 2 have high variance
    x = np.random.normal(0, 5, n_samples)
    y = np.random.normal(0, 3, n_samples)
    
    # Feature 3 is mostly just a combination of x and y with tiny noise.
    # This means the 3rd dimension is practically redundant!
    z = 0.5 * x + 0.2 * y + np.random.normal(0, 0.5, n_samples)
    
    X_3d = np.column_stack((x, y, z))

    # 2. Scale the Data (To avoid the GlassBox variance warning)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)

    # 3. Apply PCA
    print("Applying PCA (Reducing from 3D to 2D)...")
    pca = PCA(n_components=2)
    
    # Notice we use our newly created fit_transform method!
    X_2d = pca.fit_transform(X_scaled)
  # ---------------------------------------------------------
    # THIS IS DIAGNOSTICS
    # Right after fitting, before explaining.
    # ---------------------------------------------------------
    print("\nDiagnostics:")
    report = pca.diagnose()
    for key, value in report.items():
        # If the value is a list (like our failure modes), print each one nicely
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    ⚠️  {item}")
        else:
            print(f"  {key}: {value}")
    # ---------------------------------------------------------

    # Print the beautiful mathematical explanation
    print("\n" + pca.explain() + "\n")

    # 4. Visualization
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: The Original 3D Data
    ax1 = fig.add_subplot(121, projection='3d')
    # We color the points by their original Z value so we can track them
    scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                           c=X_scaled[:, 2], cmap='viridis', alpha=0.7)
    ax1.set_title("Original 3D Data (Scaled)")
    ax1.set_xlabel("Feature X")
    ax1.set_ylabel("Feature Y")
    ax1.set_zlabel("Feature Z")

    # Plot 2: The 2D PCA Projection
    ax2 = fig.add_subplot(122)
    # We use the EXACT same colors to prove the structure survived the smash!
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                           c=X_scaled[:, 2], cmap='viridis', alpha=0.7, edgecolor='k')
    ax2.set_title("2D Projection (Principal Components 1 & 2)")
    ax2.set_xlabel("Principal Component 1 (Most Variance)")
    ax2.set_ylabel("Principal Component 2 (2nd Most Variance)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
