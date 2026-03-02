import numpy as np

def make_regression(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """
    Generates a random linear dataset for regression testing.
    Returns: X (features), y (target), true_weights, true_bias
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = 2 * np.random.rand(n_samples, n_features)
    
    # Generate random true underlying weights and bias
    true_weights = np.random.uniform(-5, 5, n_features)
    true_bias = np.random.uniform(-5, 5)
    
    # y = Xw + b + noise
    y = np.dot(X, true_weights) + true_bias
    y += noise * np.random.randn(n_samples)
    
    return X, y, true_weights, true_bias

def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, random_state=None):
    """
    Generates distinct clusters of points for classification or clustering testing.
    Returns: X (features), y (cluster labels)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = []
    y = []
    
    samples_per_center = n_samples // centers
    
    for i in range(centers):
        # Pick a random center point in space
        center_coords = np.random.uniform(-10, 10, n_features)
        
        # Generate a cloud of points around that center
        cluster_points = center_coords + cluster_std * np.random.randn(samples_per_center, n_features)
        
        X.append(cluster_points)
        y.append(np.full(samples_per_center, i))
        
    # Stack everything into neat arrays
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle the dataset so classes aren't strictly ordered
    shuffle_indices = np.random.permutation(len(X))
    return X[shuffle_indices], y[shuffle_indices]


def make_stretched_blobs(n_samples_per_class=150, random_state=None):
    """
    Generates 3 classes of 3D data for dimensionality reduction testing.
    The classes are perfectly separated along the Y-axis, but massively stretched 
    along the X-axis. This acts as a trap to test if an algorithm prioritizes 
    pure variance (PCA) or class separation (LDA).
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Class 0 (Top)
    x0 = np.random.normal(0, 15, n_samples_per_class)  # Massive X variance
    y0 = np.random.normal(5, 1, n_samples_per_class)   # Y = 5
    z0 = np.random.normal(0, 1, n_samples_per_class)
    
    # Class 1 (Middle)
    x1 = np.random.normal(0, 15, n_samples_per_class)
    y1 = np.random.normal(0, 1, n_samples_per_class)   # Y = 0
    z1 = np.random.normal(0, 1, n_samples_per_class)
    
    # Class 2 (Bottom)
    x2 = np.random.normal(0, 15, n_samples_per_class)
    y2 = np.random.normal(-5, 1, n_samples_per_class)  # Y = -5
    z2 = np.random.normal(0, 1, n_samples_per_class)
    
    X = np.vstack([np.column_stack([x0, y0, z0]), 
                   np.column_stack([x1, y1, z1]), 
                   np.column_stack([x2, y2, z2])])
    
    y = np.concatenate([np.zeros(n_samples_per_class), 
                        np.ones(n_samples_per_class), 
                        np.full(n_samples_per_class, 2)])
    
    return X, y