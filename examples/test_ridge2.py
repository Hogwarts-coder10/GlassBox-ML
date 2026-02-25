import numpy as np
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.optimizer import Momentum
from models.linear_regression import LinearRegression
from models.ridge_regression import RidgeRegression
from core.preprocessing import StandardScaler



def main():
    print("--- GlassBox-ML Demo: Ridge vs. Standard Linear Regression ---")
    np.random.seed(42)
    n_samples = 200
    
    x1 = np.random.rand(n_samples, 1)
    x2 = 100 * np.random.rand(n_samples, 1) # Unscaled
    x3 = x1 + 0.05 * np.random.randn(n_samples, 1) # Multicollinear
    
    X = np.hstack((x1, x2, x3))
    
    true_weights = np.array([5.0, 0.1, 0.0])
    true_bias = 2.0
    
    # We must calculate y using the ORIGINAL X so the true relationship exists
    y = np.dot(X, true_weights) + true_bias + (0.5 * np.random.randn(n_samples))

    # --- THE FIX: SCALE THE DATA ---
    print("\nScaling the data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Train Standard Linear Regression
    print("\n[1] Training Standard Linear Regression...")
    # Because the data is scaled, we can use a massive learning rate!
    opt_lin = Momentum(learning_rate=0.1, beta=0.9) 
    model_lin = LinearRegression(optimizer=opt_lin, epochs=100)
    model_lin.fit(X_scaled, y) # Pass X_scaled here!

    print("Standard Diagnostics:")
    for key, value in model_lin.diagnose().items():
        print(f"  {key}: {value}")

    print("\n[2] Training Ridge Regression (alpha=10.0)...")
    opt_ridge = Momentum(learning_rate=0.1, beta=0.9)
    model_ridge = RidgeRegression(optimizer=opt_ridge, alpha=10.0, epochs=100)
    model_ridge.fit(X_scaled, y) # Pass X_scaled here!

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