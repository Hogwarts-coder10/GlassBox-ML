import numpy as np
import re
from typing import Optional
from collections import Counter

from glassboxml.core._base_model import GlassBoxModel

class TfidfVectorizer(GlassBoxModel):
    """
    Transforms a collection of raw text docs to a matrix of TF-IDF features

    Uses standard L2 normalization and smoothed Inverse Document Frequency (IDF)
    to prevent zero-division errors for out-of-vocabulary terms.
    """

    def __init__(self, smooth_idf: bool = True):
        super().__init__()
        self.smooth_idf = smooth_idf

        # State attributes
        self.vocabulary_: Optional[dict[str, int]] = None
        self.idf_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.is_fitted: bool = False

    def _tokenize(self,text: str) -> list[str]:
        """
        Mimics standard tokenization: lowercase, extract words of 2+ characters.
        """

        text = text.lower()
        return re.findall(r'(?u)\w\w+\b',text)

    def fit(self,raw_docs: list[str],y=None) -> "TfidfVectorizer":
        if not isinstance(raw_docs,list) or not all(isinstance(d,str) for d in raw_docs):
            raise TypeError("Input must be list of Strings")

        n_samples = len(raw_docs)

        # Building Vocabulary and calculate Document Frequency (DF)
        df_count = Counter()
        vocabulary = {}
        current_vocab_idx = 0

        for doc in raw_docs:
            tokens - self._tokenize(doc)
            unique_tokens = set(tokens)

            for token in unique_tokens:
                df_counts[token] += 1
                if token not in vocabulary:
                    vocabulary[token] = current_vocab_idx
                    current_vocab_idx += 1


        self.vocabulary_ = vocabulary
        self.n_features_ = len(vocabulary)

        # Calculating the smoothed IDF vector
        # Formula: log((1 + N) / (1 + df)) + 1

        self.idf_ = np.zeroes(self.n_features_,dtype = np.float64)

        for token, idx in self.vocabulary_.items():
            df = df_count[token]
            if self.smooth_idf:
                idf_val = np.log((1 + n_samples) / (1 + df)) + 1
            else:
                idf_val = np.log(n_samples / df) + 1

            self.idf_[idx] = idf_val

        self.is_fitted = True
        return self


    def transform(self,raw_docs:list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() before transform().")

        if not isinstance(raw_documents, list) or not all(isinstance(d, str) for d in raw_documents):
            raise TypeError("Input must be a list of strings.")

        n_samples = len(raw_documents)

        # Initialize dense matrix (Note: Scikit-learn uses sparse matrices here for memory efficiency,
        # but we use dense NumPy arrays for architectural simplicity in GlassBoxML).
        tfidf_matrix = np.zeros((n_samples, self.n_features_), dtype=np.float64)

        for doc_idx, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            term_counts = Counter(tokens)

            # Calculate TF and multiply by IDF
            for token, count in term_counts.items():
                if token in self.vocabulary_:
                    vocab_idx = self.vocabulary_[token]
                    # Raw count Term Frequency
                    tf = count
                    tfidf_matrix[doc_idx, vocab_idx] = tf * self.idf_[vocab_idx]

            # Apply L2 Normalization (row-wise)
            # This ensures document length doesn't artificially inflate the scores
            row_norm = np.linalg.norm(tfidf_matrix[doc_idx, :])
            if row_norm > 0:
                tfidf_matrix[doc_idx, :] = tfidf_matrix[doc_idx, :] / row_norm

        return tfidf_matrix

    def fit_transform(self,raw_docs: list[str],y = None) -> np.ndarray:
        self.fit(raw_docs)
        return self.transform(raw_docs)

    def explain(self) -> str:
        if not self.is_fitted:
            return "Model not fitted yet"

        explanation = "--- GlassBox Explanation: TF-IDF Vectorizer ---\n"
        explanation += f"Vocabulary Size: {self.n_features_} unique terms.\n"

        # Find the 3 most "important" (highest IDF) and "common" (lowest IDF) words
        sorted_vocab = sorted(self.vocabulary_.keys(), key=lambda x: self.idf_[self.vocabulary_[x]])
        common_words = sorted_vocab[:3]
        rare_words = sorted_vocab[-3:]

        explanation += f"Most common (ignored) terms: {common_words}\n"
        explanation += f"Most specific (highly weighted) terms: {rare_words}\n"
        explanation += "Interpretation: Converts text into a mathematical matrix.\nIt penalizes words that appear everywhere and rewards rare words that give a document its specific meaning."
        return explanation
