import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.optimizer import Momentum
from core.preprocessing import StandardScaler
from models.logistic_regression import LogisticRegression
from datasets.generators import make_blobs

def main():
    print("--- GlassBox-ML Demo: Logistic Regression (BCE vs MSE) ---")
    
    # 1. Generate Synthetic Classification Data
    print("Generating 2-class blob data...")
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.5, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Train Model 1 (BCE)
    print("\n[1] Training with Binary Cross-Entropy (BCE)...")
    opt_bce = Momentum(learning_rate=0.5, beta=0.9)
    model_bce = LogisticRegression(optimizer=opt_bce, epochs=100, loss_function='bce')
    model_bce.fit(X_scaled, y)
    
    print("BCE Diagnostics:")
    for key, value in model_bce.diagnose().items():
        print(f"  {key}: {value}")
    print("\n" + model_bce.explain() + "\n")

    # 3. Train Model 2 (MSE)
    print("\n[2] Training with Mean Squared Error (MSE)...")
    opt_mse = Momentum(learning_rate=0.5, beta=0.9)
    model_mse = LogisticRegression(optimizer=opt_mse, epochs=100, loss_function='mse')
    model_mse.fit(X_scaled, y)
    
    print("MSE Diagnostics:")
    for key, value in model_mse.diagnose().items():
        if isinstance(value, list) and value:
            for item in value:
                print(f"  ⚠️ {item}")
        else:
            print(f"  {key}: {value}")

    # 4. Extract Gradients
    bce_grad_magnitudes = [np.mean(np.abs(step['dw'])) for step in model_bce.gradient_history]
    mse_grad_magnitudes = [np.mean(np.abs(step['dw'])) for step in model_mse.gradient_history]

    # 5. Visualize
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(model_bce.loss_history, color='green', linewidth=2, label='BCE Loss')
    plt.plot(model_mse.loss_history, color='red', linewidth=2, label='MSE Loss', linestyle='--')
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(bce_grad_magnitudes, color='green', linewidth=2, label='BCE Gradients')
    plt.plot(mse_grad_magnitudes, color='red', linewidth=2, label='MSE Gradients', linestyle='--')
    plt.title("Proof of Vanishing Gradients")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Gradient Magnitude")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], color='blue', alpha=0.5, label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], color='orange', alpha=0.5, label='Class 1')
    
    # Boundary plotting
    x_vals = np.array([X_scaled[:, 0].min(), X_scaled[:, 0].max()])
    
    w_bce = model_bce.coef_
    b_bce = model_bce.intercept_
    y_vals_bce = -(w_bce[0] * x_vals + b_bce) / w_bce[1]
    
    w_mse = model_mse.coef_
    b_mse = model_mse.intercept_
    y_vals_mse = -(w_mse[0] * x_vals + b_mse) / w_mse[1]
    
    plt.plot(x_vals, y_vals_bce, color='green', linewidth=3, label='BCE Boundary')
    plt.plot(x_vals, y_vals_mse, color='red', linewidth=3, linestyle='--', label='MSE Boundary')
    
    plt.title("Decision Boundaries")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    
    # Keep the limits somewhat tight to the data
    plt.ylim(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
