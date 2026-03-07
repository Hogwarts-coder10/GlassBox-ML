import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing import StandardScaler
from models.svm import SupportVectorMachine
from datasets.generators import make_blobs

def main():
    print("--- GlassBox-ML Demo: Hard vs. Soft Margin SVM ---\n")
    
    # 1. Generate data and intentionally inject some nasty OUTLIERS
    print("Generating data with intentional outliers...")
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.2, random_state=42)
    
    # Inject two outliers that cross the center line!
    X = np.vstack([X, [[0, 3], [1, -5]]])
    y = np.append(y, [1, 0]) # Give them the 'wrong' labels
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train a "Hard" Margin SVM (Low Lambda)
    print("Training Rigid SVM (Hard Margin approximation)...")
    svm_hard = SupportVectorMachine(learning_rate=0.001, lambda_param=0.001, n_iters=1000)
    svm_hard.fit(X_scaled, y)
    
    # 3. Train a "Soft" Margin SVM (High Lambda)
    print("Training Flexible SVM (Soft Margin)...")
    svm_soft = SupportVectorMachine(learning_rate=0.001, lambda_param=1.0, n_iters=1000)
    svm_soft.fit(X_scaled, y)
    
    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    def plot_margin(ax, svm_model, title):
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', s=50, edgecolors='k')
        
        x0_range = np.array([np.amin(X_scaled[:, 0]) - 0.5, np.amax(X_scaled[:, 0]) + 0.5])
        
        # Calculate lines
        center = (svm_model.b - svm_model.w[0] * x0_range) / svm_model.w[1]
        pos_margin = (1 + svm_model.b - svm_model.w[0] * x0_range) / svm_model.w[1]
        neg_margin = (-1 + svm_model.b - svm_model.w[0] * x0_range) / svm_model.w[1]
        
        ax.plot(x0_range, center, 'k-', linewidth=2)
        ax.plot(x0_range, pos_margin, 'k--', linewidth=1.5)
        ax.plot(x0_range, neg_margin, 'k--', linewidth=1.5)
        
        ax.set_ylim([np.amin(X_scaled[:, 1]) - 1, np.amax(X_scaled[:, 1]) + 1])
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.4)

    # Plot both
    plot_margin(ax1, svm_hard, "Approaching Hard Margin (Lambda = 0.001)\nTwisted by outliers, thin street.")
    plot_margin(ax2, svm_soft, "Soft Margin (Lambda = 1.0)\nIgnores outliers, wide healthy street!")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()