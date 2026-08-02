"""
Case study: SVM, a warning confirmed, and a missing optimization trace.

Reproduces the accuracy/weight numbers cited in GlassBoxML_paper.{md,tex}
§5.5 and generates Figure 4 (svm_loss_curves.png).

Fits SVM once on raw features and once after StandardScaler, on a
300-sample, two-class dataset where one feature carries the class signal
(unit variance) and a second, uninformative feature is pure noise at 80x
the scale -- exactly the condition SVM.check_assumptions() warns about.

Run from the repo root:
    python case_studies/svm_feature_scaling.py
Writes svm_loss_curves.png to the current working directory.
"""

import matplotlib.pyplot as plt
import numpy as np

from glassboxml.core._model_selection import train_test_split
from glassboxml.metrics._classification import accuracy_score
from glassboxml.models._svm import SVM
from glassboxml.preprocessing import StandardScaler


def make_dataset(seed=3):
    rng = np.random.RandomState(seed)
    n = 300
    n0, n1 = n // 2, n - n // 2
    X0 = np.column_stack([rng.randn(n0) * 1.0 - 1.5, rng.randn(n0) * 80])
    X1 = np.column_stack([rng.randn(n1) * 1.0 + 1.5, rng.randn(n1) * 80])
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=3
    )

    print("=== SVM on unscaled features ===")
    svm_raw = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm_raw.check_assumptions(X_train, y_train)
    svm_raw.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, svm_raw.predict(X_train))
    te_acc = accuracy_score(y_test, svm_raw.predict(X_test))
    print("failure_modes:", svm_raw.failure_modes)
    print(f"Train acc: {tr_acc:.3f}  Test acc: {te_acc:.3f}")
    print("weights (w):", svm_raw.w, " bias:", svm_raw.b)
    print("loss_history[0], [-1]:", svm_raw.loss_history[0], svm_raw.loss_history[-1])
    print()

    print("=== SVM on standardized features ===")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm_scaled = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm_scaled.check_assumptions(X_train_s, y_train)
    svm_scaled.fit(X_train_s, y_train)
    tr_acc_s = accuracy_score(y_train, svm_scaled.predict(X_train_s))
    te_acc_s = accuracy_score(y_test, svm_scaled.predict(X_test_s))
    print("failure_modes:", svm_scaled.failure_modes)
    print(f"Train acc: {tr_acc_s:.3f}  Test acc: {te_acc_s:.3f}")
    print("weights (w):", svm_scaled.w, " bias:", svm_scaled.b)
    print(
        "loss_history[0], [-1]:",
        svm_scaled.loss_history[0],
        svm_scaled.loss_history[-1],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        svm_raw.loss_history, color="#c0392b", label="Unscaled features", linewidth=1.2
    )
    ax.plot(
        svm_scaled.loss_history,
        color="#2e7d5b",
        label="Standardized features",
        linewidth=1.2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regularized hinge loss")
    ax.set_title("GlassBoxML SVM: loss_history, unscaled vs. standardized features")
    ax.legend()
    plt.tight_layout()
    plt.savefig("svm_loss_curves.png", dpi=200, facecolor="white")
    print("\nSaved svm_loss_curves.png")


if __name__ == "__main__":
    main()
