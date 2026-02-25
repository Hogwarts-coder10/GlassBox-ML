import numpy as np
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.optimizer import Momentum
from models.linear_regression import LinearRegression
from models.ridge_regression import RidgeRegression

def main():
    print("--- GlassBox-ML Demo: Ridge vs. Standard Linear Regression ---")
    
    # 1. Generate "Bad" Synthetic Data
    np.random.seed(42)
    n_samples = 200
    
    # Feature 1: Small scale (0 to 1)
    x1 = np.random.rand(n_samples, 1)
    
    # Feature 2: Large scale (0 to 100) - THIS IS UNSCALED
    x2 = 100 * np.random.rand(n_samples, 1)
    
    # Feature 3: Highly correlated with x1 (x1 + some tiny noise) - MULTICOLLINEARITY
    x3 = x1 + 0.05 * np.random.randn(n_samples, 1)
    
    # Combine into feature matrix X
    X = np.hstack((x1, x2, x3))
    
    # True equation: y = 5*x1 + 0.1*x2 + 0*x3 + 2.0 + noise
    # Notice x3 actually has 0 impact in reality!
    true_weights = np.array([5.0, 0.1, 0.0])
    true_bias = 2.0
    y = np.dot(X, true_weights) + true_bias + (0.5 * np.random.randn(n_samples))

    # 2. Train Standard Linear Regression
    print("\n[1] Training Standard Linear Regression...")
    opt_lin = Momentum(learning_rate=0.0001, beta=0.9) # Very small LR needed because data is unscaled
    model_lin = LinearRegression(optimizer=opt_lin, epochs=500)
    model_lin.fit(X, y)
    
    print("Standard Diagnostics:")
    for key, value in model_lin.diagnose().items():
        print(f"  {key}: {value}")

    # 3. Train Ridge Regression
    print("\n[2] Training Ridge Regression (alpha=50.0)...")
    opt_ridge = Momentum(learning_rate=0.0001, beta=0.9)
    # Using a high alpha to really force the weights to shrink
    model_ridge = RidgeRegression(optimizer=opt_ridge, alpha=50.0, epochs=500)
    model_ridge.fit(X, y)
    
    print("Ridge Diagnostics:")
    for key, value in model_ridge.diagnose().items():
        print(f"  {key}: {value}")

    # 4. Compare the Learned Weights
    print("\n--- Weight Comparison ---")
    print(f"True Weights:    {true_weights}")
    print(f"Standard Learns: {model_lin.params['w']}")
    print(f"Ridge Learns:    {model_ridge.params['w']}")

    # 5. Visualize the Difference
    plt.figure(figsize=(10, 6))

    labels = ['Weight 1 (Small Scale)', 'Weight 2 (Large Scale)', 'Weight 3 (Correlated)']
    x_pos = np.arange(len(labels))
    width = 0.25

    # Plot True vs Standard vs Ridge
    plt.bar(x_pos - width, true_weights, width, label='True Weights', color='gray', alpha=0.5)
    plt.bar(x_pos, model_lin.params['w'], width, label='Standard Linear', color='red', alpha=0.7)
    plt.bar(x_pos + width, model_ridge.params['w'], width, label='Ridge (L2 Penalty)', color='blue', alpha=0.8)
    
    plt.xticks(x_pos, labels)
    plt.title("Effect of L2 Regularization on Problematic Data")
    plt.ylabel("Parameter Value")
    plt.axhline(0, color='black', linewidth=1) # Draw a line at y=0
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()