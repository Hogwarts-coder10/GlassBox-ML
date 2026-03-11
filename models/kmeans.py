import numpy as np
from core.base_model import GlassBoxModel

class KMeansClustering(GlassBoxModel):
    """
    Transparent K-Means Clustering.
    An unsupervised learning algorithm that discovers hidden groupings in unlabeled data
    by iteratively moving 'Centroids' to the mathematical center of local point clusters.
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        super().__init__()
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # Tolerance for stopping (if centroids barely move, we stop)
        self.centroids = None
        self.clusters = None

    def check_assumptions(self, X, y=None):
        self.failure_modes = []
        variances = np.var(X, axis=0)
        if np.max(variances) / (np.min(variances) + 1e-9) > 10:
            self.failure_modes.append(
                "[WARNING] K-Means measures physical Euclidean distance! If your features "
                "are on different scales (e.g., GPS coordinates vs. inventory counts), the distance "
                "math will be completely warped. Always use StandardScaler first."
            )
        return self.failure_modes

    def _euclidean_distance(self, point, data):
        """Calculates the straight-line distance between a point and an array of points."""
        return np.sqrt(np.sum((data - point) ** 2, axis=1))

    def fit(self, X, y=None):  # Note: y is totally ignored!
        n_samples, n_features = X.shape
        
        # 1. Randomly initialize the centroids by picking K random data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        print(f"K-Means: Dropping {self.k} random centroids and finding gravity centers...")
        
        for i in range(self.max_iters):
            # 2. Assign every point to the closest centroid
            self.clusters = [[] for _ in range(self.k)]
            cluster_labels = np.zeros(n_samples)
            
            for idx, point in enumerate(X):
                distances = self._euclidean_distance(point, self.centroids)
                closest_centroid_idx = np.argmin(distances)
                self.clusters[closest_centroid_idx].append(point)
                cluster_labels[idx] = closest_centroid_idx
                
            # 3. Calculate new centroid locations (the mean of the clusters)
            old_centroids = np.copy(self.centroids)
            
            for cluster_idx, cluster_points in enumerate(self.clusters):
                # If a centroid accidentally ended up with no points, leave it where it is
                if len(cluster_points) == 0:
                    continue
                self.centroids[cluster_idx] = np.mean(cluster_points, axis=0)
                
            # Check if the centroids stopped moving (Convergence)
            if np.all(old_centroids == self.centroids):
                print(f"  -> Converged successfully after {i+1} iterations!")
                break

        # ====================================================================
        # [NEW ADDITION] 4. Calculate total Inertia (WCSS) after training ends
        # ====================================================================
        self.inertia_ = 0.0
        
        for cluster_idx, cluster_points in enumerate(self.clusters):
            if len(cluster_points) > 0:
                # Convert list of points to a numpy array for math operations
                points_array = np.array(cluster_points)
                
                # Calculate straight-line distance from each point to its centroid
                distances = np.linalg.norm(points_array - self.centroids[cluster_idx], axis=1)
                
                # Square the distances and add to the total tension (Inertia)
                self.inertia_ += np.sum(distances ** 2)
                
        # Lock in the final labels and mark the model as trained
        self.labels_ = cluster_labels
        self.is_fitted = True

    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")
            
        # For new data, just measure which established centroid it falls closest to
        predictions = []
        for point in X:
            distances = self._euclidean_distance(point, self.centroids)
            predictions.append(np.argmin(distances))
            
        return np.array(predictions)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
            
        return (
            "--- GlassBox Explanation: K-Means Clustering ---\n"
            f"Clusters (K): {self.k}\n\n"
            "Interpretation: The model grouped the completely unlabeled data into "
            f"{self.k} distinct territories. It did this by dropping anchors, drawing borders "
            "based on pure distance, and shifting the anchors until they rested in the perfect "
            "mathematical center of their respective communities."
        )