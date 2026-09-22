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


# Upper bound on page spans accepted for one document. /api/extract-pdf emits
# one per page and parses at most MAX_PDF_PAGES pages (default 500), so this
# leaves ample headroom while bounding the per-span work below.
MAX_PAGE_SPANS = 5000

# Hard ceiling on chunks per document (defence in depth). Unbounded chunking of
# a 300k-char document yields ~520 chunks; page-bounded chunking adds at most
# one partial chunk per page. Anything past this is not a real document.
MAX_CHUNKS = 6000

# How far past the end of the normalized text a span may reach. Matches the
# +-2 tolerance of _spans_match.
_SPAN_SLACK = 2


def valid_page_spans(spans, norm_len: int) -> bool:
    """Are these spans STRUCTURALLY what page_spans() produces for a text of
    normalized length `norm_len`?

    page_spans() emits dicts {page, start, end} of ints, pages strictly
    increasing from 1, 0 <= start <= end, each span starting at or after the
    previous one's end (no overlap; blank pages are zero-width), and none
    reaching past the text. Spans that break any of this cannot have come
    from /api/extract-pdf; honouring them would let one request multiply the
    chunking/embedding work (the same region re-chunked once per span).
    """
    if not isinstance(spans, list) or len(spans) > MAX_PAGE_SPANS:
        return False
    prev_page, prev_end = 0, 0
    for s in spans:
        if not isinstance(s, dict):
            return False
        page, start, end = s.get("page"), s.get("start"), s.get("end")
        # `type(...) is int`, not isinstance: bool is an int subclass.
        if type(page) is not int or type(start) is not int or type(end) is not int:
            return False
        if page <= prev_page or start < prev_end or end < start:
            return False
        if end > norm_len + _SPAN_SLACK:
            return False
        prev_page, prev_end = page, end
    return True


def _spans_match(norm: str, spans) -> bool:
    """Do these page spans actually describe THIS text?

    page_spans lays pages out as `p1 + " " + p2 + …`, so the last page's end is
    exactly len(norm). If the user edited the extracted text before generating,
    or it was truncated after the spans were measured, that no longer holds —
    and silently chunking against stale spans would label every chunk with a
    confidently wrong page. Better to fall back to unbounded chunking and cite
    passage numbers than to cite the wrong page.
    """
    if not valid_page_spans(spans, len(norm)):
        logger.warning("ignoring %d malformed page span(s); chunking without pages",
                       len(spans) if isinstance(spans, list) else -1)
        return False
    covered = max((s["end"] for s in spans), default=0)
    return bool(covered) and abs(covered - len(norm)) <= _SPAN_SLACK


def _walk_chunks(segment, base, target_chars, overlap_chars, page, chunks):
    """Emit ~target_chars chunks for `segment`, appending to `chunks`.

    `base` is where segment[0] sits in the normalized document, so the offsets
    recorded stay document-global even when the caller is walking one page at a
    time.
    """
    if not segment:
        return

    words, pos = [], 0
    for w in segment.split(" "):
        words.append((w, pos))
        pos += len(w) + 1

    def _emit(group):
        start = group[0][1]
        last_word, last_start = group[-1]
        return " ".join(w for w, _ in group), start, last_start + len(last_word)

    def _add(body, st, en):
        if chunks and body == chunks[-1]["text"]:
            return
        entry = {
            "chunk_id": len(chunks) + 1,
            "text": body,
            "start_offset": base + st,
            "end_offset": base + en,
        }
        if page is not None:
            # Authoritative: the chunk was cut from inside this page, so no
            # offset-to-page inference happens anywhere downstream.
            entry["page"] = page
            entry["pages"] = [page]
        chunks.append(entry)

    cur, cur_len = [], 0
    for word, start in words:
        cur.append((word, start))
        cur_len += len(word) + 1
        if cur_len >= target_chars:
            _add(*_emit(cur))
            tail, tl = [], 0
            for pair in reversed(cur):
                tail.insert(0, pair)
                tl += len(pair[0]) + 1
                if tl >= overlap_chars:
                    break
            cur, cur_len = tail, sum(len(p[0]) + 1 for p in tail)
    if cur:
        _add(*_emit(cur))


def chunk_document(text, target_chars: int = 700, overlap_chars: int = 120, spans=None):
    """Split into ~target_chars passages with overlap. Returns metadata dicts:
    {chunk_id, text, start_offset, end_offset} plus {page, pages} when `spans`
    are supplied. Offsets index the whitespace-normalized text.

    With `spans`, chunking is PAGE-BOUNDED: a chunk is cut from within a single
    page and never straddles a boundary, so its page is a fact rather than
    something inferred from an offset afterwards.

    That inference is what made citations wrong. Chunks default to 700 chars
    while a slide averages ~280, so a chunk covered ~3 slides and was labelled
    with the first of them — every citation landed 1-3 pages early, never late.

    A page longer than target_chars still splits into several chunks; a page
    shorter than it becomes one chunk. Overlap does not cross pages.
    """
    norm = normalize(text)
    if not norm:
        return []

    chunks = []
    if spans and _spans_match(norm, spans):
        for span in spans:
            if span["end"] <= span["start"]:
                continue  # blank or image-only page: no text layer to chunk
            _walk_chunks(
                norm[span["start"]:span["end"]], span["start"],
                target_chars, overlap_chars, span["page"], chunks,
            )
            if len(chunks) > MAX_CHUNKS:
                break
        if chunks and len(chunks) <= MAX_CHUNKS:
            return chunks
        if chunks:
            logger.warning("page-bounded chunking passed %d chunks; chunking without pages",
                           MAX_CHUNKS)
        chunks = []

    _walk_chunks(norm, 0, target_chars, overlap_chars, None, chunks)
    if len(chunks) > MAX_CHUNKS:
        # Only reachable with a tiny target_chars on a huge text; never for
        # real inputs (see MAX_CHUNKS). Truncate rather than blow up.
        logger.warning("document produced %d chunks; keeping the first %d",
                       len(chunks), MAX_CHUNKS)
        del chunks[MAX_CHUNKS:]
    return chunks


def chunk_text(text, target_chars: int = 700, overlap_chars: int = 120):
    """Backward-compatible: just the chunk strings."""
    return [c["text"] for c in chunk_document(text, target_chars, overlap_chars)]


# ---------------------------------------------------------------------------
# Retriever facade
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, text, mode=None, spans=None):
        self.config = RetrievalConfig.from_env()
        if mode:
            self.config = self.config.with_mode(mode)
        # Page spans make chunking page-bounded, which is what makes a chunk's
        # page authoritative instead of inferred. Without them (pasted text, a
        # URL, a transcript) chunks simply carry no page and the UI cites
        # passage numbers, exactly as before.
        self.spans = spans or []
        self.chunks_meta = chunk_document(text, spans=self.spans)
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
        # Page-bounded chunks are often far smaller than target_chars (one
        # short slide each), so a fixed count would hand the writer a fraction
        # of the evidence it used to get. The budget is expressed in characters
        # so smaller chunks simply mean more of them.
        budget = final_k * self.config.chunk_target_chars
        by_index = {c["index"]: c for c in candidates}

        def entry(i):
            """Candidate record for chunk i, synthesised if ranking never saw it.

            The fused pool is capped at initial_retrieval_k, so a chunk can be
            absent from `candidates` while still being needed for coverage.
            Reading only from the pool is what silently capped the writer's
            evidence at 30 chunks no matter how much budget was left.
            """
            return by_index.get(i, {"index": i, "semantic_score": None,
                                    "bm25_score": None, "rrf_score": None,
                                    "source": "coverage"})

        size = lambda i: len(self.chunks_meta[i]["text"])  # noqa: E731
        all_indices = range(len(self.chunks_meta))

        if sum(size(i) for i in all_indices) <= budget:
            # The WHOLE document fits in the context we were going to spend.
            # Ranking can only lose material here: it was dropping substantive
            # pages purely because one query ranked them low, and WHICH pages
            # vanished changed with the query. When everything fits, selection
            # has no job to do.
            #
            # Ranked candidates stay FIRST, in relevance order — this is a
            # retrieval API and its ordering is measured (MRR). The chunks the
            # pool never saw are appended; the writer re-sorts into document
            # order itself, as the sectioned path already did.
            selected = list(candidates) + [entry(i) for i in all_indices if i not in by_index]
        else:
            # Too big to send whole, so rank and cut. Above SECTION_DOC_THRESHOLD
            # the writer runs the sectioned path instead, which issues one query
            # PER OUTLINE SECTION and unions the results — that, not a spread
            # heuristic here, is what gives long documents their coverage.
            order = {c["index"]: rank for rank, c in enumerate(candidates)}
            chosen, used = set(), 0
            for cand in candidates:
                if used >= budget:
                    break
                if cand["index"] in chosen:
                    continue
                chosen.add(cand["index"])
                used += size(cand["index"])
            selected = [entry(i) for i in sorted(chosen, key=order.__getitem__)]

        context = []
        for c in selected:
            meta = self.chunks_meta[c["index"]]
            context.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                    "start_offset": meta["start_offset"],
                    "end_offset": meta["end_offset"],
                    "page": meta.get("page"),
                    "pages": meta.get("pages") or ([meta["page"]] if meta.get("page") else []),
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
        return [
            {"id": x["chunk_id"], "text": x["text"], "page": x["page"], "pages": x["pages"]}
            for x in context
        ]

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
