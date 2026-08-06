"""
Public retrieval facade — the single entry point for the rest of the app.

Coordinates the hybrid retrieval pipeline:

    chunk_document(text)
        -> HybridRetriever (semantic + BM25, fused by RRF)
        -> candidate processing stage   [reranker plugs in here later]
        -> final context (relevance order)

Internal strategies live in the `retrieval/` package. Callers (agent.py,
main.py) keep the exact same interface as before:

    Retriever(text, mode=None)
    .retrieve(query, k=None) -> [{"id", "text"}]   (relevance order, stable IDs)
    .sample(max_chars) -> str
    chunk_text(text) -> list[str]
    active_embedding_backend() -> "gemini" | "none"

Rich per-chunk metadata (offsets, per-method scores, selection trace) and
retrieval metrics are preserved *internally* on the instance (`.last_context`,
`.last_metrics`) for debugging, benchmarking, and future observability — they
are NOT returned through the public API or emitted to the frontend.
"""

import re
import time
import logging

from retrieval.config import RetrievalConfig
from retrieval.hybrid import HybridRetriever
from retrieval.semantic import semantic_available

logger = logging.getLogger("retrieval")


def active_embedding_backend() -> str:
    """Embedding backend used for semantic retrieval."""
    return "gemini" if semantic_available() else "none"


# ---------------------------------------------------------------------------
# Chunking (with character offsets in the normalized text)
# ---------------------------------------------------------------------------

def normalize(text) -> str:
    """The one definition of the text that chunk offsets are measured against.

    Anything that needs to locate something inside a document — page spans,
    highlighting, citation verification — must measure in this same space, or
    it drifts by however much whitespace got collapsed.
    """
    return re.sub(r"\s+", " ", (text or "").strip())


def page_spans(page_texts):
    """Where each page lands in the NORMALIZED document.

    Pages that normalize to nothing (blank, or image-only with no text layer)
    get a zero-width span and do not advance the cursor — matching the
    extractor, which drops empty pages before joining. The +1 is the single
    space each join collapses to.
    """
    spans, cursor = [], 0
    for number, raw in enumerate(page_texts, start=1):
        norm = normalize(raw)
        if not norm:
            spans.append({"page": number, "start": cursor, "end": cursor})
            continue
        spans.append({"page": number, "start": cursor, "end": cursor + len(norm)})
        cursor += len(norm) + 1
    return spans


def page_for_offset(spans, offset):
    """Which page a normalized-text offset falls on, or None if unknowable.

    Falls back to the last page that starts at or before the offset, so an
    offset landing in the join between two pages still resolves.
    """
    if not spans:
        return None
    best = None
    for span in spans:
        if span["start"] <= offset < span["end"]:
            return span["page"]
        if span["start"] <= offset:
            best = span["page"]
    return best


def chunk_document(text, target_chars: int = 700, overlap_chars: int = 120):
    """Split into ~target_chars passages with overlap. Returns metadata dicts:
    {chunk_id, text, start_offset, end_offset}. Offsets index the whitespace-
    normalized text (used later for highlighting/citation verification)."""
    norm = normalize(text)
    if not norm:
        return []

    words, pos = [], 0
    for w in norm.split(" "):
        words.append((w, pos))
        pos += len(w) + 1

    def _emit(group):
        start = group[0][1]
        last_word, last_start = group[-1]
        return " ".join(w for w, _ in group), start, last_start + len(last_word)

    chunks = []
    cur, cur_len = [], 0
    for word, start in words:
        cur.append((word, start))
        cur_len += len(word) + 1
        if cur_len >= target_chars:
            body, st, en = _emit(cur)
            if not chunks or body != chunks[-1]["text"]:
                chunks.append({"chunk_id": len(chunks) + 1, "text": body, "start_offset": st, "end_offset": en})
            tail, tl = [], 0
            for pair in reversed(cur):
                tail.insert(0, pair)
                tl += len(pair[0]) + 1
                if tl >= overlap_chars:
                    break
            cur, cur_len = tail, sum(len(p[0]) + 1 for p in tail)
    if cur:
        body, st, en = _emit(cur)
        if not chunks or body != chunks[-1]["text"]:
            chunks.append({"chunk_id": len(chunks) + 1, "text": body, "start_offset": st, "end_offset": en})
    return chunks


def chunk_text(text, target_chars: int = 700, overlap_chars: int = 120):
    """Backward-compatible: just the chunk strings."""
    return [c["text"] for c in chunk_document(text, target_chars, overlap_chars)]


# ---------------------------------------------------------------------------
# Retriever facade
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, text, mode=None):
        self.config = RetrievalConfig.from_env()
        if mode:
            self.config = self.config.with_mode(mode)
        self.chunks_meta = chunk_document(text)
        self.texts = [c["text"] for c in self.chunks_meta]
        self.hybrid = HybridRetriever(self.texts, self.config)
        self.last_metrics = {}
        self.last_context = []  # full chunk metadata for the returned context (internal)

    def _process_candidates(self, candidates):
        """Isolated candidate-processing stage. Currently the identity function.
        The upcoming Cross-Encoder reranker plugs in HERE — it reorders the fused
        candidate pool before the final top-K cut, without touching any caller."""
        return candidates

    def retrieve(self, query, k=None):
        if not self.texts:
            self.last_metrics = {"mode": self.config.mode, "returned_context": 0}
            self.last_context = []
            return []

        t0 = time.perf_counter()
        candidates, metrics = self.hybrid.search(query)
        candidates = self._process_candidates(candidates)
        final_k = k or self.config.final_context_k
        selected = candidates[:final_k]

        context = []
        for c in selected:
            meta = self.chunks_meta[c["index"]]
            context.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                    "start_offset": meta["start_offset"],
                    "end_offset": meta["end_offset"],
                    "semantic_score": c["semantic_score"],
                    "bm25_score": c["bm25_score"],
                    "rrf_score": c["rrf_score"],
                    "source": c["source"],
                }
            )
        self.last_context = context

        def _avg(key):
            vals = [x[key] for x in context if x[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        self.last_metrics = {
            **metrics,
            "returned_context": len(context),
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "average_semantic_score": _avg("semantic_score"),
            "average_bm25_score": _avg("bm25_score"),
            "average_rrf_score": _avg("rrf_score"),
        }
        logger.info(
            "retrieval mode=%s sem=%d bm25=%d fused=%d returned=%d total=%sms",
            self.last_metrics["mode"],
            self.last_metrics.get("semantic_candidates", 0),
            self.last_metrics.get("bm25_candidates", 0),
            self.last_metrics.get("fusion_candidates", 0),
            self.last_metrics["returned_context"],
            self.last_metrics["total_latency_ms"],
        )

        # Public shape is unchanged: id + text, in RELEVANCE order (not document
        # order). Citation IDs are chunk identity, independent of ordering.
        return [{"id": x["chunk_id"], "text": x["text"]} for x in context]

    def sample(self, max_chars: int = 16000) -> str:
        """An even spread of chunks across the document (breadth for planning)."""
        if not self.texts:
            return ""
        avg = max(1, sum(len(c) for c in self.texts) // len(self.texts))
        budget = max(1, max_chars // avg)
        if len(self.texts) <= budget:
            selected = self.texts
        else:
            step = len(self.texts) / budget
            selected = [self.texts[int(i * step)] for i in range(budget)]
        return "\n\n".join(selected)[:max_chars]
