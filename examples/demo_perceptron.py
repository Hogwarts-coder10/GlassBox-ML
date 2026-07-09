import numpy as np

from glassboxml.models import Perceptron

def main():
    print("==================================================")
    print(" GlassBox-ML Demo: The Rosenblatt Perceptron      ")
    print("==================================================\n")

    # 1. Generate a perfectly linearly separable dataset
    # The pure Perceptron ONLY works if a single straight line can separate the data.
    print("Generating a linearly separable 2D dataset (200 points)...")
    rng = np.random.default_rng(42)
    
    # Class 0: Centered around (-2, -2)
    X_0 = rng.normal(-2, 0.5, (100, 2))
    y_0 = np.zeros(100)
    
    # Class 1: Centered around (2, 2)
    X_1 = rng.normal(2, 0.5, (100, 2))
    y_1 = np.ones(100)
    
    # Combine and shuffle
    X = np.vstack((X_0, X_1))
    y = np.hstack((y_0, y_1))

    indices = np.arange(200)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # Standard 80/20 Train-Test Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Instantiate and Train the Perceptron
    print("Booting up the 1957 Artificial Neuron...")
    # We use a small learning rate so it doesn't wildly overshoot the boundary
    neuron = Perceptron(learning_rate=0.01, n_iters=1000, random_state=99)
    
    print("Training on 160 samples...")
    neuron.fit(X_train, y_train)

    # 3. Test its predictive power
    print("Running forward pass on 40 unseen test samples...")
    predictions = neuron.predict(X_test)
    
    accuracy = np.sum(predictions == y_test) / len(y_test)

    # 4. The Verdict
    print("\n==================================================")
    print(f"-> Final Accuracy: {accuracy * 100:.2f}%")
    print("==================================================\n")
    
    # Display the learned mathematical boundary
    print(neuron.explain())

if __name__ == "__main__":
    main()