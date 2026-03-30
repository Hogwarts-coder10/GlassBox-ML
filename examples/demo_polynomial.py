import numpy as np
import matplotlib.pyplot as plt

from glassboxml.core import Momentum
from glassboxml.preprocessing import StandardScaler, PolynomialFeatures
from glassboxml.models import LinearRegression

def main():
    print("--- GlassBox-ML Demo: Polynomial Capacity & Overfitting ---")
    
    # 1. Generate Non-Linear (U-Shaped) Synthetic Data
    np.random.seed(42)
    n_samples = 50
    
    # X values between -3 and 3
    X = 6 * np.random.rand(n_samples, 1) - 3
    
    # True equation: y = 0.5 * x^2 + x + 2 + noise (This is a parabola)
    y = 0.5 * X**2 + X + 2 + np.random.randn(n_samples, 1)
    y = y.flatten()

    # Create a dense set of X values just for drawing smooth lines on the plot later
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

    # --- MODEL 1: Standard Linear Regression (Underfitting) ---
    print("\n[1] Training Standard Linear Regression (Degree 1)...")
    model_1 = LinearRegression(optimizer=Momentum(learning_rate=0.01), epochs=200)
    
    # We still scale X to be safe
    scaler_1 = StandardScaler()
    X_scaled = scaler_1.fit_transform(X)
    model_1.fit(X_scaled, y)
    
    # Predict for the smooth plot line
    y_plot_1 = model_1.predict(scaler_1.transform(X_plot))

    # --- MODEL 2: Perfect Polynomial (Degree 2) ---
    print("\n[2] Training Polynomial Regression (Degree 2)...")
    poly_2 = PolynomialFeatures(degree=2)
    X_poly_2 = poly_2.transform(X)
    
    scaler_2 = StandardScaler()
    X_poly_2_scaled = scaler_2.fit_transform(X_poly_2)
    
    model_2 = LinearRegression(optimizer=Momentum(learning_rate=0.1), epochs=200)
    model_2.fit(X_poly_2_scaled, y)
    
    y_plot_2 = model_2.predict(scaler_2.transform(poly_2.transform(X_plot)))

    # --- MODEL 3: Extreme Polynomial (Degree 15) - OVERFITTING ---
    print("\n[3] Training Extreme Polynomial Regression (Degree 15)...")
    poly_15 = PolynomialFeatures(degree=15)
    X_poly_15 = poly_15.transform(X) # This should trigger your GlassBox Warning!
    
    scaler_15 = StandardScaler()
    X_poly_15_scaled = scaler_15.fit_transform(X_poly_15)
    
    # We use a smaller learning rate because 15 degrees creates chaotic gradients
    model_15 = LinearRegression(optimizer=Momentum(learning_rate=0.01), epochs=500)
    model_15.fit(X_poly_15_scaled, y)
    
    y_plot_15 = model_15.predict(scaler_15.transform(poly_15.transform(X_plot)))

    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 5))

    # Plot 1: Underfit
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, color='gray', alpha=0.6, label='Data')
    plt.plot(X_plot, y_plot_1, color='red', linewidth=2, label='Degree 1 Fit')
    plt.title("Underfitting (High Bias)")
    plt.ylim(-2, 12)
    plt.legend()

    # Plot 2: Perfect Fit
    plt.subplot(1, 3, 2)
    plt.scatter(X, y, color='gray', alpha=0.6, label='Data')
    plt.plot(X_plot, y_plot_2, color='green', linewidth=2, label='Degree 2 Fit')
    plt.title("Optimal Capacity")
    plt.ylim(-2, 12)
    plt.legend()

    # Plot 3: Overfit
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, color='gray', alpha=0.6, label='Data')
    plt.plot(X_plot, y_plot_15, color='blue', linewidth=2, label='Degree 15 Fit')
    plt.title("Overfitting (High Variance)")
    plt.ylim(-2, 12)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
