import numpy as np
import matplotlib.pyplot as plt

from glassboxml.core import Momentum
from glassboxml.models import LinearRegression, RidgeRegression
from glassboxml.preprocessing import StandardScaler

def main():
    print("--- GlassBox-ML Demo: Ridge vs. Standard Linear Regression ---")
    np.random.seed(42)
    n_samples = 200
    
    x1 = np.random.rand(n_samples, 1)
    x2 = 100 * np.random.rand(n_samples, 1) 
    x3 = x1 + 0.05 * np.random.randn(n_samples, 1) 
    
    X = np.hstack((x1, x2, x3))
    
    true_weights = np.array([5.0, 0.1, 0.0])
    true_bias = 2.0
    
    y = np.dot(X, true_weights) + true_bias + (0.5 * np.random.randn(n_samples))

    print("\nScaling the data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[1] Training Standard Linear Regression...")
    opt_lin = Momentum(learning_rate=0.1, beta=0.9) 
    model_lin = LinearRegression(optimizer=opt_lin, epochs=100)
    model_lin.fit(X_scaled, y) 

    print("Standard Diagnostics:")
    for key, value in model_lin.diagnose().items():
        if isinstance(value, list) and value:
            for item in value:
                print(f"  ⚠️ {item}")
        else:
            print(f"  {key}: {value}")

    print("\n[2] Training Ridge Regression (alpha=10.0)...")
    opt_ridge = Momentum(learning_rate=0.1, beta=0.9)
    model_ridge = RidgeRegression(optimizer=opt_ridge, alpha=10.0, epochs=100)
    model_ridge.fit(X_scaled, y) 

    print("Ridge Diagnostics:")
    for key, value in model_ridge.diagnose().items():
        print(f"  {key}: {value}")
        
    print("\n" + model_ridge.explain() + "\n")

    # Updated to use coef_
    print("\n--- Weight Comparison ---")
    print(f"True Weights:    {true_weights}")
    print(f"Standard Learns: {np.round(model_lin.coef_, 4)}")
    print(f"Ridge Learns:    {np.round(model_ridge.coef_, 4)}")

    # 5. Visualize
    plt.figure(figsize=(10, 6))

    labels = ['Weight 1 (Small Scale)', 'Weight 2 (Large Scale)', 'Weight 3 (Correlated)']
    x_pos = np.arange(len(labels))
    width = 0.25

    plt.bar(x_pos - width, true_weights, width, label='True Weights', color='gray', alpha=0.5)
    # Updated to use coef_
    plt.bar(x_pos, model_lin.coef_, width, label='Standard Linear', color='red', alpha=0.7)
    # Updated to use coef_
    plt.bar(x_pos + width, model_ridge.coef_, width, label='Ridge (L2 Penalty)', color='blue', alpha=0.8)
    
    plt.xticks(x_pos, labels)
    plt.title("Effect of L2 Regularization on Problematic Data")
    plt.ylabel("Parameter Value")
    plt.axhline(0, color='black', linewidth=1) 
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
