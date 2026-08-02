"""
Case study: K-Nearest Neighbors, a real effect, and a threshold that
missed it by one.

Reproduces the numbers cited in GlassBoxML_paper.{md,tex} §5.6 and
generates Figure 5 (knn_dimensionality.png).

Fits KNNClassifier (k=5, Euclidean) on 200 samples with a fixed
2-dimensional class signal padded with an increasing number of
unit-variance noise dimensions, at d in {2, 5, 10, 15, 20, 30, 50} total
features -- isolating dimensionality as the only variable, since every
dimension shares the same scale (so the separate unscaled-features check
never fires).

Run from the repo root:
    python case_studies/knn_dimensionality.py
Writes knn_dimensionality.png to the current working directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from glassboxml.core._model_selection import train_test_split
from glassboxml.metrics._classification import accuracy_score
from glassboxml.models._knn import KNNClassifier

DIMS = [2, 5, 10, 15, 20, 30, 50]


def make_dataset(n_features, n=200, seed=5):
    rng = np.random.RandomState(seed)
    n0, n1 = n // 2, n - n // 2
    informative0 = rng.randn(n0, 2) - 1.0
    informative1 = rng.randn(n1, 2) + 1.0
    noise0 = rng.randn(n0, n_features - 2)
    noise1 = rng.randn(n1, n_features - 2)
    X0 = np.hstack([informative0, noise0])
    X1 = np.hstack([informative1, noise1])
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def main():
    test_accs, flagged = [], []
    for d in DIMS:
        X, y = make_dataset(d)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=5
        )
        model = KNNClassifier(k=5, p=2)
        model.check_assumptions(X_train, y_train)
        model.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, model.predict(X_train))
        te_acc = accuracy_score(y_test, model.predict(X_test))
        print(
            f"d={d:3d}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}  "
            f"failure_modes={model.failure_modes}"
        )
        test_accs.append(te_acc)
        flagged.append(bool(model.failure_modes))

    print("\n=== diagnose() sanity check ===")
    X, y = make_dataset(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=5
    )
    m = KNNClassifier(k=5)
    m.fit(X_train, y_train)
    print(m.diagnose())
    print(m.generalization_estimate)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#c0392b" if f else "#2e7d5b" for f in flagged]
    ax.plot(DIMS, test_accs, color="#888888", linewidth=1, zorder=1)
    ax.scatter(DIMS, test_accs, c=colors, s=70, zorder=2)
    ax.axvline(15, color="#888888", linestyle="--", linewidth=1)
    ax.text(
        15.5,
        min(test_accs) - 0.01,
        "check_assumptions()\nflags n_features \u2265 15",
        fontsize=8.5,
        color="#555555",
    )
    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2e7d5b",
            markersize=9,
            label="No warning",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c0392b",
            markersize=9,
            label="Curse-of-dimensionality warning",
        ),
    ]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.set_xlabel("Number of features (2 informative + noise)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("GlassBoxML KNN (k=5): test accuracy vs. dimensionality")
    ax.set_ylim(0.65, 0.95)
    plt.tight_layout()
    plt.savefig("knn_dimensionality.png", dpi=200, facecolor="white")
    print("\nSaved knn_dimensionality.png")


if __name__ == "__main__":
    main()
