import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup path to find the core and models folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.knn import KNNClassifier

def main():
    print("--- GlassBox-ML Test: KNN with Boundary Mapping ---\n")
    
    # 1. Generate a Simple Toy Dataset (Two clusters)
    np.random.seed(42)
    
    # Class 0: Centered around (2, 2)
    X0 = np.random.randn(20, 2) + np.array([2, 2])
    y0 = np.zeros(20)
    
    # Class 1: Centered around (7, 7)
    X1 = np.random.randn(20, 2) + np.array([7, 7])
    y1 = np.ones(20)
    
    X_train = np.vstack((X0, X1))
    y_train = np.concatenate((y0, y1))

    # 2. Initialize and Fit the Model
    print("Initializing KNN (k=3, p=2)...")
    knn = KNNClassifier(k=3, p=2)
    knn.fit(X_train, y_train)
    
    # Print our brand new explanation method!
    print("\n" + knn.explain() + "\n")

    # 3. Create 4 specific Test Points to classify
    X_test = np.array([
        [1.0, 1.0],   # Deep inside Class 0 territory
        [8.0, 8.0],   # Deep inside Class 1 territory
        [4.5, 4.5],   # Right in the middle (Battleground!)
        [7.0, 2.0],    # Bottom right corner
        [5.0,6.0],
        [3.0,5.0],
        [4.0,7.0]
    ])

    print("--- Predictions for New Points ---")
    predictions = knn.predict(X_test)
    for point, pred in zip(X_test, predictions):
        print(f"Point {point}  -->  Class {int(pred)}")

    # 4. Calculate the Decision Boundary Map
    print("\nMapping the decision boundaries (this takes a second)...")
    x_min, x_max = -1, 10
    y_min, y_max = -1, 10
    
    # Create a dense grid of points covering the entire map
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict the class for every single microscopic point on that grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 5. Visualize the Results
    plt.figure(figsize=(10, 8))
    
    # Draw the boundary map!
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot training data
    plt.scatter(X0[:, 0], X0[:, 1], color='blue', alpha=0.6, edgecolor='k', label='Class 0 (Training)')
    plt.scatter(X1[:, 0], X1[:, 1], color='red', alpha=0.6, edgecolor='k', label='Class 1 (Training)')
    
    # Plot the new test points as giant stars
    for i, point in enumerate(X_test):
        color = 'blue' if predictions[i] == 0 else 'red'
        label = 'New Test Point' if i == 0 else ""
        plt.scatter(point[0], point[1], color=color, marker='*', s=400, edgecolor='black', linewidth=2, label=label)
        plt.text(point[0] + 0.3, point[1], f"Pred: {int(predictions[i])}", fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    plt.title("KNN Classification with Boundary Map (k=3)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()