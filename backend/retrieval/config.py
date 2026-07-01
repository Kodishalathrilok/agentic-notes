"""Retrieval configuration — a single lightweight config object, env-driven."""

import os
from dataclasses import dataclass

VALID_MODES = ("semantic", "bm25", "hybrid")


@dataclass(frozen=True)
class RetrievalConfig:
    """All retrieval settings in one place. Switching modes is env-only."""

    mode: str = "hybrid"  # semantic | bm25 | hybrid
    initial_retrieval_k: int = 30  # candidates pulled per method and fused (pool)
    final_context_k: int = 8  # chunks handed to the LLM
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        def _int(name, default):
            try:
                return int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        mode = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
        if mode not in VALID_MODES:
            mode = "hybrid"
        return cls(
            mode=mode,
            initial_retrieval_k=_int("INITIAL_RETRIEVAL_K", 30),
            final_context_k=_int("FINAL_CONTEXT_K", 8),
            rrf_k=_int("RRF_K", 60),
        )

    def with_mode(self, mode: str) -> "RetrievalConfig":
        m = (mode or self.mode).strip().lower()
        if m not in VALID_MODES:
            m = self.mode
        return RetrievalConfig(
            mode=m,
            initial_retrieval_k=self.initial_retrieval_k,
            final_context_k=self.final_context_k,
            rrf_k=self.rrf_k,
        )
