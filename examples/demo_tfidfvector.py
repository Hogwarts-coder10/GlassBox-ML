import numpy as np
import sys, os


from glassboxml.feature_extraction.text import TfidfVectorizer

def main():
    print("==================================================")
    print(" GlassBox-ML Demo: Dual-Engine TF-IDF Vectorizer  ")
    print("==================================================\n")

    # 1. The AnchorMed Synthetic Corpus
    medical_logs = [
        "The patient requires an immediate dose of epinephrine.",
        "Inventory alert: The hospital is running low on o-negative blood.",
        "The patient is stable. No further epinephrine is needed at this time.",
        "Routine check: The patient is resting. Blood pressure is normal.",
        "Critical alert: Severe shortage of o-negative blood in the trauma ward."
    ]

    print("--- Booting Dual Engines ---")
    
    # 2. Instantiate Both Engines
    sparse_engine = TfidfVectorizer(mode="sparse")
    dense_engine = TfidfVectorizer(mode="dense")
    
    print("Training SciPy Sparse Engine (Production)...")
    sparse_matrix = sparse_engine.fit_transform(medical_logs)
    
    print("Training NumPy Dense Engine (Educational)...")
    dense_matrix = dense_engine.fit_transform(medical_logs)

    # 3. The Ultimate Parity Check
    print("\n--- Running Mathematical Parity Check ---")
    # Convert sparse to dense locally just for the validation check
    if np.allclose(sparse_matrix.toarray(), dense_matrix):
        print("✅ [PASS] Both engines produced the exact same mathematical geometry.")
    else:
        print("❌ [FAIL] Geometry mismatch detected! Check L2 Normalization math.")
        return

    # 4. Display the Upgraded Explain() Method
    print("\n" + sparse_engine.explain())
    print("\n" + dense_engine.explain())

    # 5. Feature Extraction Deep Dive (Using the Production Engine)
    print("\n--- Deep Dive: Extracting critical terms from Log 4 ---")
    print(f"Original Text: '{medical_logs[4]}'")
    
    # Extract row 4 from the sparse matrix
    doc_4_scores = sparse_matrix[4].toarray()[0]
    
    word_scores = []
    for word, idx in sparse_engine.vocabulary_.items():
        if doc_4_scores[idx] > 0:
            word_scores.append((word, doc_4_scores[idx]))
            
    # Sort by mathematical importance (highest TF-IDF score first)
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    for word, score in word_scores:
        print(f" -> '{word}': {score:.4f}")

if __name__ == "__main__":
    main()