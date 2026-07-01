"""Semantic retrieval via Gemini embeddings (REST) + cosine similarity."""

import os
import re
import math
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("retrieval")

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip()
_BASE = "https://generativelanguage.googleapis.com/v1beta"


def semantic_available() -> bool:
    return bool(GEMINI_API_KEY)


def _safe(exc) -> str:
    return re.split(r"[?&]key=", str(exc))[0]


def _embed_documents(texts):
    """Batch-embed passages (RETRIEVAL_DOCUMENT). Returns list of float vectors."""
    url = f"{_BASE}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    vectors = []
    for i in range(0, len(texts), 100):
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
    url = f"{_BASE}/models/{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    body = {
        "model": f"models/{GEMINI_EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json().get("embedding", {}).get("values", [])


def _cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticRetriever:
    """Neural retrieval. `available` flips to False (with a clear log) if the
    embedding API can't be reached — the pipeline then continues on BM25."""

    def __init__(self, texts):
        self.texts = texts
        self.embeddings = None
        self.available = semantic_available()

    def index(self):
        if not self.available or not self.texts:
            self.available = False
            return
        try:
            self.embeddings = _embed_documents(self.texts)
            if not (self.embeddings and all(self.embeddings)):
                raise ValueError("empty embeddings returned")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini document embedding failed (%s).", _safe(exc))
            self.available = False
            self.embeddings = None

    def search(self, query, k):
        if not self.available or not self.embeddings:
            return []
        try:
            q = _embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini query embedding failed (%s).", _safe(exc))
            self.available = False
            return []
        scores = [_cosine(q, ev) for ev in self.embeddings]
        ranked = sorted(range(len(self.texts)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:k]]
