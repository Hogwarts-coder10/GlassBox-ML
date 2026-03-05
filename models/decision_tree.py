import numpy as np
from core.base_model import GlassBoxModel

class Node:
    """
    A single decision tree node or leaf in a tree
    """

    def __init__(self, feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTreeClassifier(GlassBoxModel):
    """
    Transparent Decision Tree Classifier.
    Uses Information Theory (Entropy) to recursively split data.
    """

    def __init__(self,min_samples_split = 2,max_depth = 100):
        super().__init__()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def check_assumptions(self, X, y):
        self.failure_modes = []

        # Check for overfitting
        if self.max_depth > 10:
            self.failure_modes.append(
                f"High max_depth detected ({self.max_depth}). Decision Trees are highly prone to "
                "overfitting. They will memorize the training data down to the last noisy outlier. "
                "Consider pruning by lowering max_depth."
            )

        # check for scaling: Not needed for Decision trees
        variances = np.var(X,axis = 0)
        if np.max(variances) / (np.min(variances) + 1e-9) > 10:
            self.failure_modes.append(
                "EDUCATIONAL NOTE: Massive variance gap detected, but it doesn't matter! "
                "Unlike PCA or KNN, Decision Trees are scale-invariant. A split at x > 5 works "
                "the exact same way whether x is measured in inches or miles."
            )

        return self.failure_modes
    

    def fit(self,X,y):
        self._store_dataset_stats(X)
        self.check_assumptions(X,y)

        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X,y)
        self.is_fitted = True
        self.training_error = "N/A Information Theory"

    def _grow_tree(self,X,y,depth = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria (Purity, Max Depth, or Min Samples)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Greedily search for the best split
        feat_idxs = np.arange(n_feats)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Grow the children recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate the Information Gain for this specific split
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        # 1. Calculate parent entropy
        parent_entropy = self._entropy(y)

        # 2. Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # 3. Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 4. Information Gain
        return parent_entropy - child_entropy
    
    def _split(self,X_column,split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs,right_idxs
    
    def _entropy(self,y):
        # H(s) = -sum(p_i * log2(p_i))
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self,y):
        return np.bincount(y.astype(int)).argmax()
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model is not fitted yet. Call .fit() first.")
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def explain(self):
        """
        Recursively prints the logic gates learned by the tree.
        """
        if not self.is_fitted:
            return "Model is not fitted."

        explanation = "--- GlassBox Explanation: Decision Tree Logic ---\n"
        
        def print_tree(node, depth=0):
            indent = "  " * depth
            if node.is_leaf_node():
                return f"{indent}└── Predicted Class: {node.value}\n"
            
            tree_str = f"{indent}├── IF Feature {node.feature} <= {node.threshold:.4f}:\n"
            tree_str += print_tree(node.left, depth + 1)
            tree_str += f"{indent}└── ELSE (Feature {node.feature} > {node.threshold:.4f}):\n"
            tree_str += print_tree(node.right, depth + 1)
            return tree_str

        explanation += print_tree(self.root)
        return explanation


class DecisionTreeRegressor(GlassBoxModel):
    """
    Transparent Decision Tree Regressor.
    Uses Variance Reduction (Mean Squared Error) to recursively split data
    and predicts the mean value at the leaves.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        super().__init__()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def check_assumptions(self, X, y):
        self.failure_modes = []
        if self.max_depth > 10:
            self.failure_modes.append(
                f"[WARNING] High max_depth ({self.max_depth}). Regression trees are highly prone "
                "to overfitting. They will build a 'staircase' that perfectly memorizes the noise."
            )
        return self.failure_modes

    def fit(self, X, y):
        self._store_dataset_stats(X)
        self.check_assumptions(X, y)
        self.root = self._grow_tree(X, y)
        self.is_fitted = True
        self.training_error = "N/A (Variance Reduction)"

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape

        # Stopping criteria: Max depth, min samples, or variance is 0 (all values identical)
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0):
            # REGRESSION TWEAK 1: Predict the Mean!
            leaf_value = np.mean(y) 
            return Node(value=leaf_value)

        feat_idxs = np.arange(n_feats)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # If no valid split found (e.g. all remaining features are identical)
        if best_feat is None:
            return Node(value=np.mean(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_variance_reduction = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                reduction = self._variance_reduction(y, X_column, thr)

                if reduction > best_variance_reduction:
                    best_variance_reduction = reduction
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _variance_reduction(self, y, X_column, threshold):
        # REGRESSION TWEAK 2: Use Variance instead of Entropy/Gini
        parent_variance = np.var(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        var_l, var_r = np.var(y[left_idxs]), np.var(y[right_idxs])

        # Calculate weighted average variance of the children
        child_variance = (n_l / n) * var_l + (n_r / n) * var_r

        # Information Gain is just the reduction in variance!
        return parent_variance - child_variance

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("GlassBox Error: Model not fitted yet.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def explain(self):
        if not self.is_fitted:
            return "Model is not fitted."
        
        explanation = "--- GlassBox Explanation: Decision Tree Regressor ---\n"
        def print_tree(node, depth=0):
            indent = "  " * depth
            if node.is_leaf_node():
                return f"{indent}└── Predict Value (Mean): {node.value:.4f}\n"
            
            tree_str = f"{indent}├── IF Feature {node.feature} <= {node.threshold:.4f}:\n"
            tree_str += print_tree(node.left, depth + 1)
            tree_str += f"{indent}└── ELSE (Feature {node.feature} > {node.threshold:.4f}):\n"
            tree_str += print_tree(node.right, depth + 1)
            return tree_str

        explanation += print_tree(self.root)
        return explanation
        
