"""BM25 keyword retrieval via the battle-tested `rank-bm25` (Okapi BM25)."""

import logging

from rank_bm25 import BM25Okapi

from .tokenizer import tokenize

logger = logging.getLogger("retrieval")


class BM25Retriever:
    """Lexical retrieval — great at exact terms, acronyms, and formulas that
    embeddings often miss (e.g. `O(log n)`, `ATP`, `F=ma`)."""

    def __init__(self, texts):
        self.texts = texts
        self._tokenized = [tokenize(t) for t in texts]
        # rank-bm25 needs a non-empty corpus of non-empty docs.
        self.bm25 = BM25Okapi(self._tokenized) if any(self._tokenized) else None

    def search(self, query, k):
        if self.bm25 is None or not self.texts:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.texts)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:k]]
