import numpy as np
import matplotlib.pyplot as plt

from glassboxml.datasets.generators import make_blobs
from glassboxml.models.naive_bayes import GaussianNaiveBayes

def main():
    print("==================================================")
    print("   GlassBox-ML Demo: Gaussian Naive Bayes         ")
    print("==================================================\n")

    # 1. Generate 3 distinct classes of data
    print("Generating pure-Numpy blob dataset...")
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

    # 2. Train the Model
    print("Training the probability engine...")
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X, y)

    # 3. Test its accuracy on the same data
    predictions = nb_model.predict(X)
    accuracy = np.sum(predictions == y) / len(y)
    print(f"-> Training Accuracy: {accuracy * 100:.2f}%\n")
    nb_model.explain()
    
    # 4. Visualization: Drawing the Decision Boundaries
    print("Plotting the probabilistic decision boundaries...")
    plt.figure(figsize=(10, 6))

    # Create a mesh grid across the entire plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the probability for every single pixel on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nb_model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Paint the background regions based on the winning probability
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Plot the actual data points on top
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis', s=50)
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.title(f"Gaussian Naive Bayes\nCalculated Accuracy: {accuracy * 100:.2f}%")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    textstr = (
        "The background colors show the regions where\n"
        "the model calculates a higher probability for\n"
        "a specific class using pure Gaussian distributions."
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.05, textstr, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()