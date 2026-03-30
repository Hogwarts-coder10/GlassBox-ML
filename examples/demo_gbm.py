import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import GradientBoostingRegressor

def main():
    print("--- GlassBox-ML Demo: Gradient Boosting ---\n")

    # 1. Generate a complex, curvy, noisy continuous dataset from scratch
    print("Generating complex continuous data...")
    rng = np.random.RandomState(42)
    X = np.sort(10 * rng.rand(150, 1), axis=0)
    # A sine wave that gets taller and faster as it moves right
    y = np.sin(X).ravel() + np.sin(2 * X).ravel() + (X.ravel() * 0.2)
    y += 0.5 * (0.5 - rng.rand(150)) # Add noise

    # 2. Train three different GBMs to show the "learning" process
    print("Training ensembles to show the evolution of the curve...")
    gbm_early = GradientBoostingRegressor(n_trees=2, learning_rate=0.2, max_depth=3)
    gbm_mid = GradientBoostingRegressor(n_trees=15, learning_rate=0.2, max_depth=3)
    gbm_final = GradientBoostingRegressor(n_trees=100, learning_rate=0.1, max_depth=3)
    
    gbm_early.fit(X, y)
    gbm_mid.fit(X, y)
    gbm_final.fit(X, y)

    print("\n" + gbm_final.explain() + "\n")

    # 3. Predict across the whole X-axis
    X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
    y_early = gbm_early.predict(X_test)
    y_mid = gbm_mid.predict(X_test)
    y_final = gbm_final.predict(X_test)

    # 4. Visualization
    plt.figure(figsize=(12, 7))
    
    plt.scatter(X, y, s=30, edgecolor="black", c="lightgray", label="Raw Noisy Data", alpha=0.8)
    
    plt.plot(X_test, y_early, color="red", label="After 2 Trees (Underfitting)", linewidth=2, linestyle='--')
    plt.plot(X_test, y_mid, color="orange", label="After 15 Trees (Learning)", linewidth=2, linestyle='-.')
    plt.plot(X_test, y_final, color="blue", label="After 100 Trees (Perfect Fit)", linewidth=3)
    
    plt.xlabel("Feature X")
    plt.ylabel("Target y (Continuous)")
    plt.title("Gradient Boosting: Learning by fixing previous mistakes!")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    textstr = "Watch the progression: The red line is barely trying.\nThe orange line finds the major peaks.\nThe blue line uses 100 trees to perfectly map the residuals!"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
