import numpy as np
import scipy.sparse as sp
import re
from typing import Optional
from collections import Counter

from glassboxml.core._base_model import GlassBoxModel

class TfidfVectorizer(GlassBoxModel):
    """
    Transforms a collection of raw text documents to a matrix of TF-IDF features.

    Features a Dual-Engine architecture:
    - mode="sparse" (Default): Uses SciPy CSR matrices for production-grade memory efficiency.
    - mode="dense": Uses standard NumPy arrays for educational transparency and explainability.
    """

    def __init__(self, smooth_idf: bool = True, mode: str = "sparse"):
        super().__init__()
        self.smooth_idf = smooth_idf

        if mode not in ["dense", "sparse"]:
            raise ValueError("mode must be either 'dense' or 'sparse'")
        self.mode = mode

        # State attributes
        self.vocabulary_: Optional[dict[str, int]] = None
        self.idf_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.is_fitted: bool = False

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r'(?u)\b\w\w+\b', text)

    def fit(self, raw_documents: list[str], y=None) -> "TfidfVectorizer":
        if not isinstance(raw_documents, list) or not all(isinstance(d, str) for d in raw_documents):
            raise TypeError("Input must be a list of strings.")

        n_samples = len(raw_documents)
        df_counts = Counter()
        vocabulary = {}
        current_vocab_idx = 0

        for doc in raw_documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df_counts[token] += 1
                if token not in vocabulary:
                    vocabulary[token] = current_vocab_idx
                    current_vocab_idx += 1

        self.vocabulary_ = vocabulary
        self.n_features_ = len(vocabulary)

        self.idf_ = np.zeros(self.n_features_, dtype=np.float64)
        for token, idx in self.vocabulary_.items():
            df = df_counts[token]
            if self.smooth_idf:
                idf_val = np.log((1 + n_samples) / (1 + df)) + 1
            else:
                idf_val = np.log(n_samples / df) + 1
            self.idf_[idx] = idf_val

        self.is_fitted = True
        return self

    def transform(self, raw_documents: list[str]) -> Union[np.ndarray, sp.csr_matrix]:
        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")

        n_samples = len(raw_documents)

        # =========================================================
        # ENGINE 1: The Production Route (SciPy Sparse)
        # =========================================================
        if self.mode == "sparse":
            rows, cols, data = [], [], []
            for doc_idx, doc in enumerate(raw_documents):
                tokens = self._tokenize(doc)
                term_counts = Counter(tokens)
                for token, count in term_counts.items():
                    if token in self.vocabulary_:
                        vocab_idx = self.vocabulary_[token]
                        rows.append(doc_idx)
                        cols.append(vocab_idx)
                        data.append(count * self.idf_[vocab_idx])

            tfidf_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, self.n_features_), dtype=np.float64)

            # Vectorized Sparse L2 Normalization
            sq_sum = tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)
            row_norms = np.sqrt(sq_sum).A1
            row_norms[row_norms == 0] = 1.0
            tfidf_matrix.data /= np.repeat(row_norms, np.diff(tfidf_matrix.indptr))

            return tfidf_matrix

        # =========================================================
        # ENGINE 2: The Educational Route (NumPy Dense)
        # =========================================================
        else:
            tfidf_matrix = np.zeros((n_samples, self.n_features_), dtype=np.float64)
            for doc_idx, doc in enumerate(raw_documents):
                tokens = self._tokenize(doc)
                term_counts = Counter(tokens)
                for token, count in term_counts.items():
                    if token in self.vocabulary_:
                        vocab_idx = self.vocabulary_[token]
                        tfidf_matrix[doc_idx, vocab_idx] = count * self.idf_[vocab_idx]

                # Standard Dense L2 Normalization
                row_norm = np.linalg.norm(tfidf_matrix[doc_idx, :])
                if row_norm > 0:
                    tfidf_matrix[doc_idx, :] = tfidf_matrix[doc_idx, :] / row_norm

            return tfidf_matrix

    def fit_transform(self, raw_documents: List[str], y=None) -> Union[np.ndarray, sp.csr_matrix]:
        self.fit(raw_documents)
        return self.transform(raw_documents)
    
    def predict(self,X):
        """
        Transformers map data into new feature spaces; they do not make predictions.
        """

        raise NotImplementedError(
            "TfidfVectorizer is a Transformer, not a Predictor. "
            "Use .transform() or .fit_transform() instead."
        )
    

    def explain(self) -> str:
        if not self.is_fitted:
            return "Model is not fitted yet."

        engine_type = "SciPy CSR (Optimized)" if self.mode == "sparse" else "NumPy Dense (Educational)"

        explanation = "--- GlassBox Explanation: TF-IDF Vectorizer ---\n"
        explanation += f"Active Engine: {engine_type}\n"
        explanation += f"Vocabulary Size: {self.n_features_} unique terms.\n"
        explanation += f"Matrix shape: {self.n_features_} features\n"  # 🚀 Your upgrade

        sorted_vocab = sorted(self.vocabulary_.keys(), key=lambda x: self.idf_[self.vocabulary_[x]])
        explanation += f"Most common (lowest weight) terms: {sorted_vocab[:3]}\n"
        explanation += f"Most specific (highest weight) terms: {sorted_vocab[-3:]}\n"

        if self.mode == "sparse":
            explanation += "Architecture: Converts text into an ultra-efficient SciPy CSR matrix, bypassing millions of zeroes for raw scaling power."
        else:
            explanation += "Architecture: Converts text into a raw NumPy grid. Excellent for studying the underlying geometry, but memory-heavy for massive datasets."

        return explanation
