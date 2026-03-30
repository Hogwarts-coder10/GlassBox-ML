import numpy as np
# Setup path

from glassboxml.core import train_test_split
from glassboxml.metrics import classification_report, plot_confusion_matrix
from glassboxml.models import RandomForestClassifier
from glassboxml.datasets import make_blobs

def main():
    print("--- GlassBox-ML: The Confusion Matrix ---\n")

    # 1. Generate a noisy dataset with overlapping classes
    print("Generating overlapping data to force some mistakes...")
    X, y = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=3.5, random_state=42)

    # 2. Split into 80% Training / 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # 3. Train the Random Forest
    print("Training the Random Forest...")
    rf = RandomForestClassifier(n_trees=15, max_depth=5)
    rf.fit(X_train, y_train)

    # 4. Predict on the Test Set
    print("Administering the blind test...")
    y_pred = rf.predict(X_test)

    # 5. Print the text report
    print("\n" + classification_report(y_test, y_pred))

    # 6. Render the Visual Matrix!
    print("Rendering the Confusion Matrix plot...")
    plot_confusion_matrix(y_test, y_pred, title="Random Forest: Test Set Mistakes")

if __name__ == "__main__":
    main()
