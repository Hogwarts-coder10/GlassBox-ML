import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.optimizer import Momentum
from models.linear_regression import LinearRegression
from datasets.generators import make_regression

def main():
    print("--- GlassBox-ML Demo: Linear Regression ---")
    
    # 1. Generate Synthetic Data using GlassBox Generators
    print("Generating synthetic 1D regression data...")
    X, y, true_weights, true_bias = make_regression(n_samples=100, n_features=1, noise=1.0, random_state=42)
    y = y.flatten()

    # 2. Initialize Optimizer and Model
    print("\nInitializing Momentum Optimizer (lr=0.1, beta=0.09)...")
    optimizer = Momentum(learning_rate=0.1, beta=0.09)
    
    print("Initializing Model...")
    model = LinearRegression(optimizer=optimizer, epochs=100)

    # 3. Train the Model
    print("Training...\n")
    model.fit(X, y)

    # 4. Open the Glass Box (Diagnostics & Explanation)
    print("--- Diagnostic Report ---")
    report = model.diagnose()
    for key, value in report.items():
        if isinstance(value, list) and value:
            print(f"{key}:")
            for item in value:
                print(f"  ⚠️ {item}")
        else:
            print(f"{key}: {value}")

    print("\n" + model.explain() + "\n")

    print(f"True Parameters: Weight = {np.round(true_weights, 4)}, Bias = {true_bias:.4f}")
    print(f"Learned Parameters: Weight = {np.round(model.coef_, 4)}, Bias = {model.intercept_:.4f}")

    # 5. Visualize the Learning Journey
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, color='blue', linewidth=2)
    plt.title("Loss History (MSE) Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.scatter(X, y, color='gray', label='Noisy Data', alpha=0.6)
    
    # Draw the fit line across the range of X
    x_line = np.array([[X.min()], [X.max()]])
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
