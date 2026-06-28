import os
import re
import pickle
import math
from langchain_core.documents import Document

class BM25Index:
    """
    A lightweight, self-contained implementation of the Okapi BM25 lexical search algorithm.
    Allows serializing/deserializing the corpus index to disk.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: dict[str, Document] = {}  # chunk_id -> Document
        self.doc_lengths: dict[str, int] = {}  # chunk_id -> doc_length (tokens)
        self.doc_tfs: dict[str, dict[str, int]] = {}  # chunk_id -> {term -> count}
        self.df: dict[str, int] = {}  # term -> document count
        self.idf: dict[str, float] = {}  # term -> idf score
        self.avgdl: float = 0.0
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # Lowercase and replace non-alphanumeric characters with space
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    def fit(self, documents: list[Document]):
        """
        Indexes a list of Document objects, calculating term frequencies,
        document frequencies, average document length, and term IDFs.
        """
        if not documents:
            return

        self.corpus = {}
        self.doc_lengths = {}
        self.doc_tfs = {}
        self.df = {}
        self.idf = {}
        self.num_docs = len(documents)

        total_length = 0
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("All documents must have a unique 'chunk_id' in metadata.")

            self.corpus[chunk_id] = doc
            
            # Tokenize document content
            tokens = self._tokenize(doc.page_content)
            doc_len = len(tokens)
            self.doc_lengths[chunk_id] = doc_len
            total_length += doc_len

            # Calculate term frequencies for this doc
            tf = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1
            self.doc_tfs[chunk_id] = tf

            # Calculate document frequencies
            for term in tf.keys():
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_length / self.num_docs if self.num_docs > 0 else 0.0

        # Calculate IDF for each term in the corpus
        for term, doc_freq in self.df.items():
            # Standard BM25 IDF formulation with smoothing
            self.idf[term] = math.log((self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def score(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """
        Calculates BM25 score for the query against all documents in the corpus.
        Returns the top_k matching Documents with their BM25 scores.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []

        scores = []
        for chunk_id, doc in self.corpus.items():
            score = 0.0
            doc_len = self.doc_lengths[chunk_id]
            tf_dict = self.doc_tfs[chunk_id]

            for term in query_tokens:
                if term not in self.idf:
                    continue
                
                term_tf = tf_dict.get(term, 0)
                if term_tf == 0:
                    continue

                # Okapi BM25 formula
                numerator = term_tf * (self.k1 + 1.0)
                denominator = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += self.idf[term] * (numerator / denominator)

            if score > 0.0:
                scores.append((doc, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, filepath: str):
        """Serializes the BM25 index state to disk using pickle."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath: str):
        """Deserializes the BM25 index state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BM25 index file not found at '{filepath}'.")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.__dict__.update(state)
