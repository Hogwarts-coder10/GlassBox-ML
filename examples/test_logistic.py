import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.optimizer import Momentum
from core.preprocessing import StandardScaler
from models.logistic_regression import LogisticRegression

def main():
    print("--- GlassBox-ML Demo: Logistic Regression (BCE vs MSE) ---")
    
    # 1. Generate Synthetic Classification Data (Two distinct blobs)
    np.random.seed(42)
    n_samples = 100
    
    # Class 0: Centered around (-2, -2)
    X0 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    y0 = np.zeros(n_samples)
    
    # Class 1: Centered around (2, 2)
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    y1 = np.ones(n_samples)
    
    # Combine the data
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Train Model 1: Binary Cross-Entropy (The Right Way)
    print("\n[1] Training with Binary Cross-Entropy (BCE)...")
    opt_bce = Momentum(learning_rate=0.5, beta=0.9)
    model_bce = LogisticRegression(optimizer=opt_bce, epochs=100, loss_function='bce')
    model_bce.fit(X_scaled, y)
    
    print("BCE Diagnostics:")
    for key, value in model_bce.diagnose().items():
        print(f"  {key}: {value}")

    # 3. Train Model 2: Mean Squared Error (The Wrong Way)
    print("\n[2] Training with Mean Squared Error (MSE)...")
    opt_mse = Momentum(learning_rate=0.5, beta=0.9)
    model_mse = LogisticRegression(optimizer=opt_mse, epochs=100, loss_function='mse')
    model_mse.fit(X_scaled, y)
    
    print("MSE Diagnostics:")
    for key, value in model_mse.diagnose().items():
        print(f"  {key}: {value}")

    # 4. Extract Gradient Magnitudes for Visualization
    # We take the average absolute value of the 'dw' gradients at each epoch
    bce_grad_magnitudes = [np.mean(np.abs(step['dw'])) for step in model_bce.gradient_history]
    mse_grad_magnitudes = [np.mean(np.abs(step['dw'])) for step in model_mse.gradient_history]

    # 5. Visualize the Difference
    plt.figure(figsize=(15, 5))

    # Plot 1: The Loss Curves
    # (Note: We plot them separately because BCE and MSE are on different scales)
    plt.subplot(1, 3, 1)
    plt.plot(model_bce.loss_history, color='green', linewidth=2, label='BCE Loss')
    plt.plot(model_mse.loss_history, color='red', linewidth=2, label='MSE Loss', linestyle='--')
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot 2: THE GLASSBOX FEATURE - Gradient Magnitudes
    plt.subplot(1, 3, 2)
    plt.plot(bce_grad_magnitudes, color='green', linewidth=2, label='BCE Gradients')
    plt.plot(mse_grad_magnitudes, color='red', linewidth=2, label='MSE Gradients', linestyle='--')
    plt.title("Proof of Vanishing Gradients")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Gradient Magnitude")
    plt.legend()

    # Plot 3: The Data and the Learned Boundaries
    plt.subplot(1, 3, 3)
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], color='blue', alpha=0.5, label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], color='orange', alpha=0.5, label='Class 1')
    
    # Draw decision boundaries (where probability = 0.5, so Xw + b = 0)
    # x2 = -(w1*x1 + b) / w2
    x_vals = np.array([-2, 2])
    
    w_bce = model_bce.params['w']
    b_bce = model_bce.params['b']
    y_vals_bce = -(w_bce[0] * x_vals + b_bce) / w_bce[1]
    
    w_mse = model_mse.params['w']
    b_mse = model_mse.params['b']
    y_vals_mse = -(w_mse[0] * x_vals + b_mse) / w_mse[1]
    
    plt.plot(x_vals, y_vals_bce, color='green', linewidth=3, label='BCE Boundary')
    plt.plot(x_vals, y_vals_mse, color='red', linewidth=3, linestyle='--', label='MSE Boundary')
    
    plt.title("Decision Boundaries")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.ylim(-3, 3)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()