import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.optimizer import Momentum
from models.linear_regression import LinearRegression

def main():
    print("--- GlassBox-ML Demo: Multiple Linear Regression ---")
    
    # 1. Generate Synthetic Data (3 Features)
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    
    # X is now a matrix with 3 columns
    X = 2 * np.random.rand(n_samples, n_features)
    
    # True parameters we want the model to learn
    true_weights = np.array([3.5, -1.2, 5.0]) 
    true_bias = 4.2
    
    # y = w1*x1 + w2*x2 + w3*x3 + b + noise
    y = np.dot(X, true_weights) + true_bias + (0.5 * np.random.randn(n_samples))

    # 2. Initialize Optimizer and Model
    print("\nInitializing Momentum Optimizer (lr=0.1, beta=0.9)...")
    # Let's put beta back to 0.9 to speed through the 3 features!
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    
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

    print(f"\nTrue Parameters: Weights = {true_weights}, Bias = {true_bias}")
    print(f"Learned Parameters: Weights = {model.params['w']}, Bias = {model.params['b']:.4f}")

    # 5. Visualize the Learning Journey
    plt.figure(figsize=(12, 5))

    # Plot 1: The Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, color='blue', linewidth=2)
    plt.title("Loss History (MSE) Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: True vs Learned Parameters Bar Chart
    plt.subplot(1, 2, 2)
    labels = ['w1', 'w2', 'w3', 'bias']
    true_vals = [true_weights[0], true_weights[1], true_weights[2], true_bias]
    learned_vals = [model.params['w'][0], model.params['w'][1], model.params['w'][2], model.params['b']]

    x_pos = np.arange(len(labels))
    width = 0.35

    plt.bar(x_pos - width/2, true_vals, width, label='True Parameters', color='gray', alpha=0.6)
    plt.bar(x_pos + width/2, learned_vals, width, label='Learned Parameters', color='red', alpha=0.8)
    
    plt.xticks(x_pos, labels)
    plt.title("Did the Model Learn the True Equation?")
    plt.ylabel("Parameter Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()