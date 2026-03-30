import numpy as np
import matplotlib.pyplot as plt

from glassboxml.core import Momentum
from glassboxml.models import LinearRegression
from datasets.generators import make_regression

def main():
    print("--- GlassBox-ML Demo: Multiple Linear Regression ---")
    
    # 1. Generate Synthetic Data using GlassBox Generators
    print("Generating synthetic 3D regression data...")
    X, y, true_weights, true_bias = make_regression(n_samples=200, n_features=3, noise=0.5, random_state=42)

    # 2. Initialize Optimizer and Model
    print("\nInitializing Momentum Optimizer (lr=0.1, beta=0.9)...")
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
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

    print(f"True Parameters: Weights = {np.round(true_weights, 4)}, Bias = {true_bias:.4f}")
    print(f"Learned Parameters: Weights = {np.round(model.coef_, 4)}, Bias = {model.intercept_:.4f}")

    # 5. Visualize the Learning Journey
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, color='blue', linewidth=2)
    plt.title("Loss History (MSE) Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    labels = ['w1', 'w2', 'w3', 'bias']
    true_vals = [true_weights[0], true_weights[1], true_weights[2], true_bias]
    learned_vals = [model.coef_[0], model.coef_[1], model.coef_[2], model.intercept_]

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
