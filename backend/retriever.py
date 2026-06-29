"""
Lightweight retrieval for RAG — chunking + TF-IDF cosine similarity.

Pure Python (no extra dependencies), so it runs anywhere and deploys on a free
tier. The interface is intentionally swappable: replace `_vec` / similarity with
a neural-embedding backend later without touching the pipeline.
"""

import re
import math
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(s: str):
    return _WORD.findall(s.lower())


def chunk_text(text: str, target_chars: int = 700, overlap_chars: int = 120):
    """
    Split text into ~target_chars passages with a small overlap. Word-based so
    it handles punctuation-less input (e.g. auto-generated YouTube transcripts).
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    words = text.split(" ")
    chunks = []
    cur, cur_len = [], 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= target_chars:
            chunks.append(" ".join(cur))
            # keep a tail for overlap so context isn't cut mid-thought
            tail, tl = [], 0
            for ww in reversed(cur):
                tail.insert(0, ww)
                tl += len(ww) + 1
                if tl >= overlap_chars:
                    break
            cur, cur_len = tail, sum(len(x) + 1 for x in tail)
    if cur:
        joined = " ".join(cur)
        if not chunks or joined != chunks[-1]:
            chunks.append(joined)
    return chunks


class Retriever:
    """TF-IDF retriever over the chunks of a single source document."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        self._tokens = [_tokenize(c) for c in self.chunks]
        n = len(self.chunks)
        df = Counter()
        for toks in self._tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vectors = [self._vec(toks) for toks in self._tokens]

    def _vec(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {t: (c / total) * self.idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(self, query: str, k: int = 8):
        """Return the top-k most relevant chunks (in document order) as
        [{"id": 1-based, "text": ...}]."""
        if not self.chunks:
            return []
        q = self._vec(_tokenize(query))
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: self._cosine(q, self.vectors[i]),
            reverse=True,
        )
        top = sorted(ranked[: min(k, len(self.chunks))])
        return [{"id": i + 1, "text": self.chunks[i]} for i in top]

    def sample(self, max_chars: int = 16000) -> str:
        """An even spread of chunks across the document (breadth for planning)."""
        if not self.chunks:
            return ""
        avg = max(1, sum(len(c) for c in self.chunks) // len(self.chunks))
        budget = max(1, max_chars // avg)
        if len(self.chunks) <= budget:
            selected = self.chunks
        else:
            step = len(self.chunks) / budget
            selected = [self.chunks[int(i * step)] for i in range(budget)]
        return "\n\n".join(selected)[:max_chars]
