import numpy as np

from glassboxml.core._base_model import GlassBoxModel
from glassboxml.models._decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class RandomForestClassifier(GlassBoxModel):
    """
    Transparent Random Forest Classifier.
    An ensemble method that trains multiple Decision Trees on random subsets
    of the data to prevent overfitting (Bagging).
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        super().__init__()
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.n_trees < 2:
            self.failure_modes.append(
                "[WARNING] A Random Forest with fewer than 2 trees is just a single Decision Tree! "
                "You need multiple trees to benefit from the 'Wisdom of the Crowds'."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)

        print(f"⚡ Booting Stochastic Micro-Batch Forest ({self.n_trees} trees)...")
        self.trees = []  # Reset trees on fresh fit

        n_samples = X.shape[0]
        # The Magic Number: Give each tree only 3,000 random rows.
        # This forces the tree to train instantly while maintaining ensemble diversity.
        sample_size = min(3000, n_samples)

        for i in range(self.n_trees):
            # 1. Bootstrap Sub-Sampling
            np.random.seed(42 + i)  # Ensures reproducible randomness
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_batch, y_batch = X[indices], y[indices]

            # 2. Train the base tree on the micro-batch
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gini",
            )
            tree.fit(X_batch, y_batch)
            self.trees.append(tree)

            # Print a clean progress bar for the terminal so you know it's not frozen
            print(f"  [>] Tree {i + 1}/{self.n_trees} planted.")

        self.is_fitted = True

        # Ensemble training error: fraction of training rows the majority
        # vote gets wrong, so diagnose() / generalization_estimate have a
        # real numeric signal to work with instead of silently staying None.
        train_preds = self.predict(X)
        self.training_error = float(np.mean(train_preds != y))

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError(
                "GlassBox Error: Model is not fitted yet. Call .fit() first."
            )

        # Get predictions from every single tree
        # Shape will be (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Swap axes to (n_samples, n_trees) so we can vote row-by-row
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # Majority Vote!
        y_pred = [np.bincount(preds.astype(int)).argmax() for preds in tree_preds]
        return np.array(y_pred)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."

        return (
            "--- GlassBox Explanation: Random Forest ---\n"
            f"Ensemble Size: {self.n_trees} Independent Trees\n"
            f"Max Depth per Tree: {self.max_depth}\n\n"
            "Interpretation: The model used Bootstrapping to train dozens of trees on "
            "random variations of the training data. To make a prediction, it asks every "
            "tree for an answer and takes a democratic majority vote. This prevents the "
            "overfitting seen in single Decision Trees."
        )

    def _get_effective_params(self):
        """
        Tree ensembles aren't well described by the base class's coef_-based
        default (which falls back to n_features and would report near-zero
        complexity regardless of ensemble size or depth). Approximate capacity
        instead as the total number of leaf nodes across all trees, summed --
        each leaf is roughly one degree of freedom the ensemble can use to
        memorize training data.
        """
        if not self.trees:
            return self.dataset_stats_.get("n_features", 1)

        def count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf_node():
                return 1
            return count_leaves(node.left) + count_leaves(node.right)

        total_leaves = sum(count_leaves(tree.root) for tree in self.trees)
        return max(total_leaves, 1)


class RandomForestRegressor(GlassBoxModel):
    """
    Transparent Random Forest Regressor.
    Trains multiple Decision Tree Regressors on random subsets of the data
    and averages their continuous predictions to smooth out the 'staircase' effect.
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        super().__init__()
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.n_trees < 2:
            self.failure_modes.append(
                "[WARNING] A Random Forest with 1 tree is just a single Decision Tree! "
                "Increase n_trees to smooth out the regression predictions."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        self.trees = []
        n_samples = X.shape[0]

        print(f"Planting a regression forest of {self.n_trees} trees...")
        for i in range(self.n_trees):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )

            # BOOTSTRAPPING: Pick random rows with replacement
            bootstrap_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idxs]
            y_bootstrap = y[bootstrap_idxs]

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        self.is_fitted = True
        self.training_error = "N/A (Ensemble Averaging)"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet.")

        # Get predictions from every single tree
        # Shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # REGRESSION TWEAK: Take the mathematical average of all predictions!
        y_pred = np.mean(tree_preds, axis=0)
        return y_pred

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."

        return (
            "--- GlassBox Explanation: Random Forest Regressor ---\n"
            f"Ensemble Size: {self.n_trees} Independent Regression Trees\n"
            f"Max Depth per Tree: {self.max_depth}\n\n"
            "Interpretation: The model trained dozens of decision trees on random variations "
            "of the data. Because each tree saw different noise, their individual 'staircase' "
            "predictions were erratic. By averaging them all together, the model canceled out "
            "the noise and found the true continuous curve."
        )
