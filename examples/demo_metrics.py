import numpy as np

from glassboxml.metrics import classification_report, confusion_matrix

def main():
    print("--- GlassBox-ML Demo: Metrics & The Accuracy Paradox ---")

    # 1. Generate an incredibly imbalanced dataset
    # Imagine a medical test: 950 healthy people (Class 0), 50 sick people (Class 1)
    np.random.seed(42)
    y_true = np.array([0] * 950 + [1] * 50)
    np.random.shuffle(y_true)

    # ---------------------------------------------------------
    # 2. Model A: The "Dumb" Model (Predicts 0 every single time)
    # ---------------------------------------------------------
    print("\n[Model A] The 'Always Healthy' Predictor (Predicts 0 unconditionally)")
    y_pred_dumb = np.zeros(1000)

    tp, fp, tn, fn = confusion_matrix(y_true, y_pred_dumb)
    print(f"Confusion Matrix -> TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(classification_report(y_true, y_pred_dumb))

    # ---------------------------------------------------------
    # 3. Model B: A "Realistic" Model 
    # ---------------------------------------------------------
    print("\n[Model B] A Realistic Predictor (Makes some mistakes, but tries)")
    y_pred_smart = np.copy(y_true)
    
    # Let's introduce some realistic errors to the smart model:
    # 10 False Positives (Healthy people told they are sick)
    zero_indices = np.where(y_true == 0)[0]
    y_pred_smart[np.random.choice(zero_indices, 10, replace=False)] = 1

    # 5 False Negatives (Sick people told they are healthy)
    one_indices = np.where(y_true == 1)[0]
    y_pred_smart[np.random.choice(one_indices, 5, replace=False)] = 0

    tp, fp, tn, fn = confusion_matrix(y_true, y_pred_smart)
    print(f"Confusion Matrix -> TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(classification_report(y_true, y_pred_smart))

if __name__ == "__main__":
    main()
