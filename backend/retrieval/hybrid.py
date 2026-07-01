"""Hybrid retrieval: run semantic + BM25, fuse with RRF, emit metrics.

Failure handling: if semantic can't run, it does NOT fail silently — it logs a
clear message and the pipeline continues on BM25.
"""

import time
import logging

from .config import RetrievalConfig
from .bm25 import BM25Retriever
from .semantic import SemanticRetriever
from .fusion import reciprocal_rank_fusion

logger = logging.getLogger("retrieval")


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


class HybridRetriever:
    """Coordinates the configured retrieval methods and fuses their results.

    Returns a candidate pool (up to `initial_retrieval_k`) in relevance order,
    each annotated with per-method scores and a selection trace, plus metrics.
    """

    def __init__(self, texts, config: RetrievalConfig):
        self.texts = texts
        self.config = config
        self.semantic = SemanticRetriever(texts) if config.mode in ("semantic", "hybrid") else None
        self.bm25 = BM25Retriever(texts) if config.mode in ("bm25", "hybrid") else None
        self._index()

    def _index(self):
        t = time.perf_counter()
        # BM25 index is built in BM25Retriever.__init__; time the embed step.
        bm_build_ms = _ms(time.perf_counter() - t)
        embed_ms = 0.0
        if self.semantic is not None:
            t = time.perf_counter()
            self.semantic.index()
            embed_ms = _ms(time.perf_counter() - t)
            if not self.semantic.available:
                if self.config.mode == "hybrid":
                    logger.warning("Gemini embeddings unavailable. Running BM25 retrieval only.")
                    if self.bm25 is None:
                        self.bm25 = BM25Retriever(self.texts)
                elif self.config.mode == "semantic":
                    logger.warning("Gemini embeddings unavailable. Falling back to BM25 retrieval only.")
                    self.bm25 = BM25Retriever(self.texts)
        logger.info("Indexed %d chunks (bm25 %sms, embed %sms).", len(self.texts), bm_build_ms, embed_ms)

    def search(self, query):
        cfg = self.config
        k = cfg.initial_retrieval_k

        sem_ranked, bm_ranked = [], []
        sem_ms = bm_ms = fus_ms = 0.0

        if self.semantic is not None and self.semantic.available:
            t = time.perf_counter()
            sem_ranked = self.semantic.search(query, k)
            sem_ms = _ms(time.perf_counter() - t)

        if self.bm25 is not None:
            t = time.perf_counter()
            bm_ranked = self.bm25.search(query, k)
            bm_ms = _ms(time.perf_counter() - t)

        sem_scores = dict(sem_ranked)
        bm_scores = dict(bm_ranked)

        rankings = []
        if sem_ranked:
            rankings.append([i for i, _ in sem_ranked])
        if bm_ranked:
            rankings.append([i for i, _ in bm_ranked])

        t = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings, cfg.rrf_k) if rankings else []
        fus_ms = _ms(time.perf_counter() - t)

        candidates = []
        for idx, rrf in fused[:k]:
            in_sem, in_bm = idx in sem_scores, idx in bm_scores
            source = "both" if in_sem and in_bm else ("semantic" if in_sem else "bm25")
            candidates.append(
                {
                    "index": idx,
                    "semantic_score": sem_scores.get(idx),
                    "bm25_score": bm_scores.get(idx),
                    "rrf_score": rrf,
                    "source": source,
                }
            )

        metrics = {
            "mode": cfg.mode,
            "semantic_candidates": len(sem_ranked),
            "bm25_candidates": len(bm_ranked),
            "fusion_candidates": len(candidates),
            "semantic_latency_ms": sem_ms,
            "bm25_latency_ms": bm_ms,
            "fusion_latency_ms": fus_ms,
        }
        return candidates, metrics
