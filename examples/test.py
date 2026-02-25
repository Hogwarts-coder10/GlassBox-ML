import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.optimizer import Momentum
from models.linear_regression import LinearRegression

def main():
    print("--- GlassBox-ML Demo: Linear Regression ---")
    
    # 1. Generate Synthetic Data
    # True equation: y = 3x + 4 + noise
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Flatten y to (n_samples,) to match our math cleanly
    y = y.flatten()

    # 2. Initialize Optimizer and Model
    print("\nInitializing Momentum Optimizer (lr=0.1, beta=0.09)...")
    optimizer = Momentum(learning_rate=0.1, beta=0.09)
    
    print("Initializing Model...")
    model = LinearRegression(optimizer=optimizer, epochs=100)

    # 3. Train the Model
    print("Training...\n")
    model.fit(X, y)

    # 4. Open the Glass Box (Diagnostics)
    print("--- Diagnostic Report ---")
    report = model.diagnose()
    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nTrue Parameters: Weight = [3.0], Bias = 4.0")
    print(f"Learned Parameters: Weight = {model.params['w']}, Bias = {model.params['b']:.4f}")

    # 5. Visualize the Learning Journey
    plt.figure(figsize=(12, 5))

    # Plot 1: The Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, color='blue', linewidth=2)
    plt.title("Loss History (MSE) Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: The Final Fit
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, color='gray', label='Noisy Data', alpha=0.6)
    
    # Generate points for the regression line
    x_line = np.array([[0], [2]])
    y_line = model.predict(x_line)
    
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Learned Fit')
    plt.title("Data vs. Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()