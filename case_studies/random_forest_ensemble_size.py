"""
Case study: Random Forest ensemble size, a structural blind spot, and two
genuine bugs.

Reproduces Table 4 and the numbers cited in GlassBoxML_paper.{md,tex} §5.4.

Trains a single DecisionTreeClassifier (max_depth=50) as an in-run baseline,
then RandomForestClassifier at n_trees in {1, 10, 50} (also max_depth=50),
all on the same train/test split of a fresh 200-sample, two-class, 8%-noise
dataset generated with the same recipe as §5.3.

Run from the repo root:
    python case_studies/random_forest_ensemble_size.py
"""

import numpy as np

from glassboxml.core._model_selection import train_test_split
from glassboxml.metrics._classification import accuracy_score
from glassboxml.models._decision_tree import DecisionTreeClassifier
from glassboxml.models._random_forest import RandomForestClassifier


def make_dataset(n=200, noise_frac=0.08, seed=11):
    rng = np.random.RandomState(seed)
    n0, n1 = n // 2, n - n // 2
    X0 = rng.randn(n0, 2) * 1.2 + np.array([0, 0])
    X1 = rng.randn(n1, 2) * 1.2 + np.array([1.5, 1.5])
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    n_flip = int(noise_frac * n)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=11
    )

    print("=== Single Decision Tree (depth=50), in-run baseline ===")
    tree = DecisionTreeClassifier(max_depth=50, min_samples_split=2, criterion="gini")
    tree.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, tree.predict(X_train))
    te_acc = accuracy_score(y_test, tree.predict(X_test))
    print(
        f"Train acc: {tr_acc:.3f}  Test acc: {te_acc:.3f}  Gap: {tr_acc - te_acc:+.3f}"
    )
    print("failure_modes:", tree.failure_modes)
    print()

    for n_trees in [1, 10, 50]:
        print(f"=== RandomForestClassifier(n_trees={n_trees}, max_depth=50) ===")
        rf = RandomForestClassifier(n_trees=n_trees, max_depth=50, min_samples_split=2)
        rf.check_assumptions(X_train, y_train)
        rf.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, rf.predict(X_train))
        te_acc = accuracy_score(y_test, rf.predict(X_test))
        gap = tr_acc - te_acc
        print(f"Train acc: {tr_acc:.3f}  Test acc: {te_acc:.3f}  Gap: {gap:+.3f}")
        print("failure_modes:", rf.failure_modes)
        print("diagnose():", rf.diagnose())
        print("generalization_estimate:", rf.generalization_estimate)
        print()


if __name__ == "__main__":
    main()
