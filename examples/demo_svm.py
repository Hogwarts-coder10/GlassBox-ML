import numpy as np
import matplotlib.pyplot as plt

from glassboxml.core.preprocessing import StandardScaler
from glassboxml.models.svm import SupportVectorMachine
from glassboxml.datasets.generators import make_blobs


def main():
    print("--- GlassBox-ML Demo: Linear SVM ---\n")
    
    # 1. Generate 2D Linearly Separable Data
    print("Generating binary classification data...")
    X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    
    # 2. Scale the data (CRITICAL for SVMs!)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train the SVM
    svm = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_scaled, y)
    
    print("\n" + svm.explain() + "\n")
    
    # 4. Visualization (Drawing the Street)
    print("Mapping the maximum margin hyperplane...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the data points
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', s=50, edgecolors='k', alpha=0.7)
    
    # --- The Geometry of the Margin ---
    # The line equation is: w0*x0 + w1*x1 - b = offset
    # We solve for x1 to plot it: x1 = (offset + b - w0*x0) / w1
    def get_hyperplane_value(x, w, b, offset):
        return (offset + b - w[0] * x) / w[1]
        
    # Get the min and max of the X-axis to draw the lines across the whole plot
    x0_min = np.amin(X_scaled[:, 0]) - 0.5
    x0_max = np.amax(X_scaled[:, 0]) + 0.5
    x0_range = np.array([x0_min, x0_max])
    
    # Calculate the Y-coordinates for the center, top, and bottom lines
    x1_center = get_hyperplane_value(x0_range, svm.w, svm.b, 0)
    x1_pos_margin = get_hyperplane_value(x0_range, svm.w, svm.b, 1)
    x1_neg_margin = get_hyperplane_value(x0_range, svm.w, svm.b, -1)
    
    # Draw the lines
    ax.plot(x0_range, x1_center, 'k-', linewidth=2, label="Decision Boundary (Hyperplane)")
    ax.plot(x0_range, x1_pos_margin, 'k--', linewidth=1.5, label="Positive Margin (+1)")
    ax.plot(x0_range, x1_neg_margin, 'k--', linewidth=1.5, label="Negative Margin (-1)")
    
    # Lock the Y-axis limits so the lines don't stretch the plot to infinity
    ax.set_ylim([np.amin(X_scaled[:, 1]) - 1, np.amax(X_scaled[:, 1]) + 1])
    
    ax.set_title("Support Vector Machine: Maximum Margin Street")
    ax.set_xlabel("Feature 0 (Scaled)")
    ax.set_ylabel("Feature 1 (Scaled)")
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Add educational text
    textstr = "Notice how the dotted margin lines rest directly\non the closest data points (The Support Vectors!)."
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()