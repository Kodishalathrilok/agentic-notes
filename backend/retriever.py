"""
Retrieval for RAG — pluggable embedding backend.

- "gemini": neural/semantic embeddings via Google's REST API (when GEMINI_API_KEY
  is set). Better at matching meaning/paraphrase. Uses `requests` (no new dep).
- "tfidf": pure-Python lexical fallback (no key, no network). Always available.

The Retriever picks the backend automatically and falls back to TF-IDF if the
embedding API is unavailable, so it never breaks.
"""

import os
import re
import math
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip()
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

_WORD = re.compile(r"[a-z0-9]+")


def _safe_err(exc) -> str:
    """Strip the API key out of error messages before logging."""
    return str(exc).split("?key=")[0]


def active_embedding_backend() -> str:
    """What the retriever will try first."""
    return "gemini" if GEMINI_API_KEY else "tfidf"


def _tokenize(s: str):
    return _WORD.findall(s.lower())


def chunk_text(text: str, target_chars: int = 700, overlap_chars: int = 120):
    """Split text into ~target_chars passages with small overlap (word-based, so
    it handles punctuation-less transcripts)."""
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


# ---------------------------------------------------------------------------
# Gemini embeddings (REST)
# ---------------------------------------------------------------------------

def _embed_documents(texts):
    """Batch-embed passages (RETRIEVAL_DOCUMENT). Returns list of float vectors."""
    url = f"{_GEMINI_BASE}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    vectors = []
    for i in range(0, len(texts), 100):  # API allows up to 100 per batch
        batch = texts[i : i + 100]
        body = {
            "requests": [
                {
                    "model": f"models/{GEMINI_EMBED_MODEL}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
                for t in batch
            ]
        }
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        for emb in resp.json().get("embeddings", []):
            vectors.append(emb.get("values", []))
    return vectors


def _embed_query(text):
    url = f"{_GEMINI_BASE}/models/{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    body = {
        "model": f"models/{GEMINI_EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json().get("embedding", {}).get("values", [])


def _cosine_list(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Retrieves the most relevant chunks of a single source document.

    backend: "auto" (default) | "gemini" | "tfidf"
    """

    def __init__(self, text: str, backend: str = "auto"):
        self.chunks = chunk_text(text)
        self.backend = "tfidf"
        self.embeddings = []

        want_gemini = backend in ("auto", "gemini") and active_embedding_backend() == "gemini"
        if self.chunks and want_gemini:
            try:
                self.embeddings = _embed_documents(self.chunks)
                if self.embeddings and all(self.embeddings):
                    self.backend = "gemini"
            except Exception as exc:  # noqa: BLE001
                print(f"[retriever] Gemini embeddings failed ({_safe_err(exc)}); using TF-IDF.")

        if self.backend == "tfidf":
            self._build_tfidf()

    # ---- TF-IDF backend ----
    def _build_tfidf(self):
        self._tokens = [_tokenize(c) for c in self.chunks]
        n = len(self.chunks)
        df = Counter()
        for toks in self._tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vectors = [self._tfidf_vec(toks) for toks in self._tokens]

    def _tfidf_vec(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {t: (c / total) * self.idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _tfidf_cosine(a, b):
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    # ---- public API ----
    def retrieve(self, query: str, k: int = 8):
        if not self.chunks:
            return []

        if self.backend == "gemini":
            try:
                q = _embed_query(query)
                scores = [_cosine_list(q, ev) for ev in self.embeddings]
            except Exception as exc:  # noqa: BLE001
                print(f"[retriever] Gemini query embedding failed ({_safe_err(exc)}); using TF-IDF.")
                self._build_tfidf()
                self.backend = "tfidf"
                return self.retrieve(query, k)
            ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        else:
            q = self._tfidf_vec(_tokenize(query))
            ranked = sorted(
                range(len(self.chunks)),
                key=lambda i: self._tfidf_cosine(q, self.vectors[i]),
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
