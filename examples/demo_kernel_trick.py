import numpy as np
import matplotlib.pyplot as plt
import sys, os

from glassboxml.core.preprocessing import StandardScaler
from glassboxml.models.svm import SupportVectorMachine
from glassboxml.datasets.generators import make_donut

def main():
    print("--- GlassBox-ML Demo: The Kernel Trick ---\n")

    # 1. Generate 2D Donut Data (Linearly Inseparable!)
    print("Generating 2D donut data...")
    X_2d, y = make_donut(n_samples=300, noise=0.15, random_state=42)

    # 2. THE KERNEL TRICK (Manual Feature Mapping)
    # We calculate the squared distance from the center (X^2 + Y^2) to create a 3rd dimension (Z)
    print("Warping 2D data into 3D space (Z = X² + Y²)...")
    Z = X_2d[:, 0]**2 + X_2d[:, 1]**2
    X_3d = np.column_stack((X_2d, Z))

    # 3. Scale the 3D data
    print("Scaling 3D features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)

    # 4. Train our standard Linear SVM on the 3D data!
    print("\nTraining Linear SVM to find a flat plane in 3D...")
    svm = SupportVectorMachine(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X_scaled, y)
    print("\n" + svm.explain() + "\n")

    # 5. Epic 3D Visualization
    print("Rendering the 3D Kernel Trick...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D data points
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
               c=y, cmap='bwr', s=40, edgecolors='k', alpha=0.8)

    # Draw the flat 3D hyperplane: w0*x + w1*y + w2*z - b = 0  =>  z = (b - w0*x - w1*y) / w2
    xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 20),
                         np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 20))
    zz = (svm.b - svm.w[0] * xx - svm.w[1] * yy) / svm.w[2]

    # Plot the slicing plane
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3, edgecolor='none')

    ax.set_title("The Kernel Trick: Slicing warped data with a flat 2D plane!")
    ax.set_xlabel("Feature 0 (X)")
    ax.set_ylabel("Feature 1 (Y)")
    ax.set_zlabel("Warped Dimension (Z = X² + Y²)")
    
    # Adjust the camera angle for the perfect view
    ax.view_init(elev=15, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()