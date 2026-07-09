import numpy as np
import matplotlib.pyplot as plt

from glassboxml.models import DecisionTreeClassifier
from glassboxml.datasets import make_blobs
from glassboxml.core import train_test_split


def main():
    print("🌳 GlassBoxML Demo: Decision Tree Classifier")

    # 1. Generate synthetic data
    X, y = make_blobs(n_samples=200, centers=3, random_state=42)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Initialize model
    model = DecisionTreeClassifier(max_depth=5)

    # 4. Train
    model.fit(X_train, y_train)

    # 5. Predict
    preds = model.predict(X_test)

    # 6. Accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # 7. GlassBox diagnostics
    print("\n--- Diagnostics ---")
    print(model.diagnose())

    print("\n--- Explanation ---")
    print(model.explain())

    # 8. Visualization (optional)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.title("Decision Tree Classification (Data View)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


    xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

if __name__ == "__main__":
    main()

