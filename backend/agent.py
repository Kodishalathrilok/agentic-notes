"""
Multi-agent notes pipeline.

run_agent(...) drives the whole pipeline as a synchronous generator that
yields event dicts of the shape: { type, step, content, data }.

Standalone helpers (used by the regenerate / chat / title endpoints):
    generate_quiz, generate_flashcards, rewrite_notes,
    generate_title, chat_about_notes_stream
"""

import os
import re
import json
import time
import uuid
import logging
import threading
import queue as _queue
from concurrent.futures import ThreadPoolExecutor

from models import (call_model, call_model_stream, safe_json, helper_model, IncompleteStreamError,
                    UserFacingError, ProvidersUnavailableError, log_unexpected_error,
                    PipelineCancelled, cancel_scope, raise_if_cancelled,
                    call_stats_scope, current_call_stats, _effective_cancel)
from retriever import Retriever, page_for_offset

# Quality thresholds for the self-improvement loop
REVISE_THRESHOLD = 8  # revise until score reaches this (1-10)
MAX_REVISION_ROUNDS = 2  # cap revision passes to bound latency

# Long-document handling: above this size, notes are written SECTION BY SECTION
# (map-reduce) — each outline section gets its own retrieval over the whole
# document, so no part of a large source is left out.
SECTION_DOC_THRESHOLD = 12000  # chars
# SUPERSEDED by COVERAGE_SUPPLEMENT_K: sections are document windows now, and
# retrieval supplements a window rather than defining it. Kept so an existing
# deployment setting it does not break on import.
SECTION_RETRIEVAL_K = 6

# Single-pass (small-doc) path: how many chunks the writer sees, scaled by the
# requested length. The old fixed k=8 (~5.6k chars) starved "long" notes — the
# writer can't produce 1000-1500 words from < 1000 words of context. These pull
# a much larger slice of the document (candidate pool is 30) so the length
# targets are actually reachable.
SINGLE_PASS_K = {
    "short": 8,
    "medium": 14,
    "long": 24,
}
# Cap sections so a very large doc can't fire an unbounded burst of model calls
# (protects free-tier rate limits). Override with SECTION_MAX_COUNT.
# SUPERSEDED by COVERAGE_MAX_WINDOWS, which bounds generation calls the same
# way. Kept for import compatibility.
SECTION_MAX_COUNT = int(os.getenv("SECTION_MAX_COUNT", "8"))
# How many sections are written CONCURRENTLY on the map-reduce path. Sections
# are independent (each has its own retrieved context), so overlapping the
# model calls cuts wall-clock time; keep this modest to respect free-tier
# rate limits (failover still covers 429s). Override with SECTION_CONCURRENCY.
# This is PER RUN: server-wide, at most MAX_CONCURRENT_GENERATIONS (main.py)
# x SECTION_CONCURRENCY section streams are in flight at once.
SECTION_CONCURRENCY = int(os.getenv("SECTION_CONCURRENCY", "2"))

_logger = logging.getLogger("agentic")


def _concurrency(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# Independent model calls run with at most this many in flight PER RUN.
# Grounding batches and digest segments never read each other's results, so
# overlapping them changes wall-clock time only; results are always merged in
# input order. Server-wide the ceiling is MAX_CONCURRENT_GENERATIONS x this.
GROUNDING_CONCURRENCY = _concurrency("GROUNDING_CONCURRENCY", 4)
DIGEST_CONCURRENCY = _concurrency("DIGEST_CONCURRENCY", 3)


def _cancel_kw(cancel):
    # Only pass `cancel` when there is one, so fakes of the writer/stream
    # functions without that parameter keep working (same trick as `strict`).
    return {"cancel": cancel} if cancel is not None else {}


def _carry_scopes(fn):
    """Bind fn to the CALLING thread's ambient cancel flag and call-stats sink.

    Both are thread-locals (models.cancel_scope / call_stats_scope), so a pool
    worker would otherwise run with neither: its model calls would ignore a
    cancelled run and go uncounted in the timing log.
    """
    cancel, sink = _effective_cancel(), current_call_stats()

    def run(*args, **kwargs):
        with cancel_scope(cancel), call_stats_scope(sink):
            return fn(*args, **kwargs)
    return run


def _run_bounded(fn, items, limit, name):
    """Return [(result, exc)] for fn(item) over `items`, in INPUT order.

    At most `limit` calls are in flight. An Exception from one item is
    returned in its slot (the caller keeps its own fail-open handling);
    PipelineCancelled is a BaseException and propagates, and items not yet
    started are then never started. limit 1 (or a single item) runs inline,
    exactly the old sequential loop.
    """
    if limit <= 1 or len(items) <= 1:
        out = []
        for item in items:
            try:
                out.append((fn(item), None))
            except Exception as exc:  # noqa: BLE001
                out.append((None, exc))
        return out

    cancel = _effective_cancel()
    work = _carry_scopes(fn)
    pool = ThreadPoolExecutor(max_workers=min(limit, len(items)), thread_name_prefix=name)
    try:
        futures = [pool.submit(work, item) for item in items]
        out = []
        for fut in futures:
            try:
                out.append((fut.result(), None))
            except Exception as exc:  # noqa: BLE001
                out.append((None, exc))
        return out
    finally:
        # Queued items are always dropped on the way out. On cancellation also
        # wait for the running ones (they stop before their next provider
        # attempt), so no worker is still spending tokens once the run ends.
        pool.shutdown(wait=cancel is not None and cancel.is_set(), cancel_futures=True)

# Corrective re-retrieval (CRAG-style): when the critique reports missing
# topics, run a fresh retrieval PER TOPIC and add those chunks to the context
# before revising — the reviser can then actually fix omissions instead of
# being asked to add material it was never shown.
CORRECTIVE_TOPICS_MAX = 4  # cap topics queried per revision round
CORRECTIVE_K_PER_TOPIC = 3  # chunks retrieved per missing topic

# Above this size, the planner's even sample covers too little of the document
# (e.g. ~8% of a 100-page PDF), so a topic on only 2-3 pages can be invisible
# to it. Fix: a full-coverage DIGEST scan — the cheap helper model reads the
# ENTIRE document in large segments and lists each segment's topics, and the
# planner outlines from that 100%-coverage inventory instead of a sample.
DIGEST_DOC_THRESHOLD = int(os.getenv("DIGEST_DOC_THRESHOLD", "60000"))
DIGEST_SEGMENT_CHARS = 25000
DIGEST_MAX_SEGMENTS = 12  # 12 x 25k = full coverage of the 300k input cap


# ---------------------------------------------------------------------------
# Document coverage: windows that PARTITION the source
# ---------------------------------------------------------------------------
#
# The sectioned writer used to build its context from one retrieval per outline
# topic, so the most a large document could ever show the writer was
# SECTION_MAX_COUNT x SECTION_RETRIEVAL_K chunks — 48, whatever the document's
# size. Measured on synthetic documents: 78% of chunks at 10 pages, 23% at 40,
# 9% at 100, 4% at 200. Whether a passage was seen depended on how a handful of
# queries happened to rank it.
#
# Windows fix that structurally. Every chunk belongs to exactly one window and
# every window is written, so covering the document stops being a retrieval
# outcome and becomes an arithmetic one. Retrieval still runs — it adds
# cross-section evidence to each window — but it no longer decides what exists.

# Cost is bounded by the NUMBER of windows, not their size: each window is one
# generation call. A bigger document therefore gets BIGGER windows rather than
# more of them, which is where a large provider context window actually earns
# its keep.
COVERAGE_MAX_WINDOWS = int(os.getenv("COVERAGE_MAX_WINDOWS", "12"))
COVERAGE_WINDOW_CHARS = int(os.getenv("COVERAGE_WINDOW_CHARS", "6000"))
# Supplemental retrieval per window: related material from ELSEWHERE in the
# document (a definition introduced in an earlier section, say).
COVERAGE_SUPPLEMENT_K = int(os.getenv("COVERAGE_SUPPLEMENT_K", "3"))

# HARD ceiling on the rendered context handed to ONE window write, in chars.
#
# This is the budget windows are BUILT to fit, not a slice applied afterwards.
# The writer used to end its prompt with `context[:16000]`, so a window could be
# assigned pages 1-17 and forward only pages 1-5 - selection coverage of 100%
# with writer coverage of 62%. Silent, and invisible to every test because the
# tests stopped at selection.
#
# 40000 is derived, not guessed:
#   - the single-pass writer has shipped with 40000 chars of context on these
#     same providers and models, so that much input is proven in production;
#   - worst-case prompt scaffolding measured at 3600 chars (longest doc-type
#     rule, full checklist, custom instructions), so a full prompt is ~43600
#     chars, about 10900 tokens;
#   - output is reserved separately: SECTION_MAX_TOKENS tops out at 1600 plus
#     1024 reasoning headroom = 2624 tokens;
#   - NVIDIA accepted ~400K prompt tokens in testing and Gemini 3.6 Flash
#     carries 1M, so this uses roughly 2.7% of the smaller proven figure.
# The headroom is deliberate: the point is that a window NEVER overflows, not
# that it uses as much of the provider as it can.
WRITER_CONTEXT_CHARS = int(os.getenv("WRITER_CONTEXT_CHARS", "40000"))

# _format_context renders each chunk as "[id] text" joined by a blank line, so
# budgeting on raw text alone would under-count. 12 covers a 4-digit id, the
# brackets, a space and the separator.
_CHUNK_RENDER_CHARS = 12
# Supplements are optional extras; reserving room for them keeps them from
# pushing a window over budget, so a window always fits its OWN pages.
_SUPPLEMENT_RESERVE = COVERAGE_SUPPLEMENT_K * 1000


class _WindowProducedNothing(RuntimeError):
    """Every provider returned an empty stream for this window."""


def _failure_slug(exc) -> str:
    """Non-secret failure category for the coverage record.

    Derived from the exception TYPE only. Provider messages routinely carry the
    request URL, which carries an API key, so they must never reach the client.
    """
    if isinstance(exc, _WindowProducedNothing):
        return "empty_output"
    if isinstance(exc, IncompleteStreamError):
        return exc.reason  # "interrupted" / "max_tokens" - fixed slugs
    if isinstance(exc, TimeoutError):
        return "timeout"
    return type(exc).__name__.lower()


def _page_ranges(pages) -> str:
    """Render [3,4,6,7,8] as '3-4, 6-8' for a human-readable gap notice."""
    runs = []
    for pg in sorted(set(pages)):
        if runs and pg == runs[-1][1] + 1:
            runs[-1][1] = pg
        else:
            runs.append([pg, pg])
    return ", ".join(str(a) if a == b else f"{a}\u2013{b}" for a, b in runs)


def _window_title(chunks, index, total):
    """Name a window by the pages it covers — a claim about the document, not
    about its subject, so it cannot mislabel the content."""
    pages = sorted({c["page"] for c in chunks if c.get("page")})
    if not pages:
        return f"Part {index} of {total}"
    if pages[0] == pages[-1]:
        return f"Page {pages[0]}"
    return f"Pages {pages[0]}–{pages[-1]}"


def _document_windows(chunks_meta, max_windows=None, window_chars=None,
                      max_chars=None):
    """Partition chunks into contiguous, document-ordered windows.

    Returns [(title, [chunk...])]. Every chunk appears in exactly one window.

    The per-window budget grows with the document so the call count stays
    bounded: a 200-page source becomes a dozen large windows rather than a
    hundred small ones.
    """
    chunks = sorted(chunks_meta, key=lambda c: c["chunk_id"])
    if not chunks:
        return []

    max_windows = max_windows or COVERAGE_MAX_WINDOWS
    window_chars = window_chars or COVERAGE_WINDOW_CHARS
    budget = max_chars or (WRITER_CONTEXT_CHARS - _SUPPLEMENT_RESERVE)
    total = sum(len(c["text"]) + _CHUNK_RENDER_CHARS for c in chunks)
    target = max(window_chars, -(-total // max(1, max_windows)))

    # The budget is a HARD ceiling. When splitting into max_windows would make
    # windows larger than the writer can accept, the window COUNT grows instead:
    # the overflow moves into another window that is actually written, which is
    # the whole point. Capping the count here is what used to force the last
    # window to absorb the remainder and then lose it to the prompt slice.
    bounded = target > budget
    if bounded:
        target = budget
    limit = None if bounded else max_windows

    windows, current, used = [], [], 0
    for chunk in chunks:
        size = len(chunk["text"]) + _CHUNK_RENDER_CHARS
        # Break BEFORE adding when the window is already full, so a single
        # oversized chunk still gets a window of its own rather than being lost.
        if current and used + size > target and (limit is None or len(windows) < limit - 1):
            windows.append(current)
            current, used = [], 0
        current.append(chunk)
        used += size
    if current:
        windows.append(current)

    return [(_window_title(w, i, len(windows)), w) for i, w in enumerate(windows, 1)]


def _window_context(window_chunks, retriever, mode, supplement_k=None,
                    max_chars=None):
    """A window's own chunks (guaranteed) plus related evidence from elsewhere.

    The window's own material is never displaced by retrieval; supplements are
    appended and de-duplicated. If retrieval fails — a Gemini 429 degrades the
    hybrid retriever to BM25, and even that could return nothing — the window
    still has its own chunks, so a section can never vanish because embeddings
    were unavailable.
    """
    supplement_k = COVERAGE_SUPPLEMENT_K if supplement_k is None else supplement_k
    budget = max_chars or WRITER_CONTEXT_CHARS
    by_id = {c["chunk_id"]: {"id": c["chunk_id"], "text": c["text"],
                             "page": c.get("page"), "pages": c.get("pages") or []}
             for c in window_chunks}
    own_ids = set(by_id)
    used = sum(len(c["text"]) + _CHUNK_RENDER_CHARS for c in window_chunks)

    if supplement_k > 0 and retriever is not None:
        query = " ".join(c["text"][:200] for c in window_chunks[:3])
        try:
            for c in retriever.retrieve(f"{query} — {mode}", k=supplement_k) or []:
                if c["id"] in own_ids:
                    continue
                size = len(c["text"]) + _CHUNK_RENDER_CHARS
                # Supplements are a bonus; the window's own pages are the
                # contract. Stop adding rather than push the window over budget.
                if used + size > budget:
                    break
                by_id[c["id"]] = c
                used += size
        except Exception as exc:  # noqa: BLE001
            print(f"[coverage] supplemental retrieval failed ({exc}); "
                  f"window keeps its own chunks.")

    return [by_id[i] for i in sorted(by_id)]


def _format_context(chunks) -> str:
    """Render retrieved chunks as numbered passages the agent can cite."""
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


# Deterministic citation verification: [n] markers are only kept if n is a
# chunk ID that was actually retrieved and shown to the model. This turns
# "don't invent citations" from a prompt instruction into a code guarantee.
# Any number of digits: chunk ids reach retriever.MAX_CHUNKS (6000), and the
# UI renders every bracketed number as a citation (NotesOutput.jsx), so a
# marker this pattern missed would be shown to the reader yet never checked.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def enforce_citations(notes: str, valid_ids) -> str:
    """Strip any [n] citation whose n is not a real retrieved-chunk ID.

    Pure text post-processing — no model call. Leaves markdown links
    (``[text](url)``) untouched because they never match a bare ``[digits]``
    pattern followed by nothing.
    """
    valid = set()
    for i in valid_ids:
        try:
            valid.add(int(i))
        except (TypeError, ValueError):
            continue

    def _sub(match):
        return match.group(0) if int(match.group(1)) in valid else ""

    cleaned = _CITATION_RE.sub(_sub, notes or "")
    # Tidy whitespace left behind by removals (trailing spaces before newlines).
    return re.sub(r"[ \t]+(\n)", r"\1", cleaned)


# Per-claim grounding check before the notes are finalised. On by default;
# set GROUNDING_CHECK=0 to skip it (one extra helper-model call per 8 claims).
GROUNDING_CHECK = os.getenv("GROUNDING_CHECK", "1").strip().lower() not in ("0", "false", "no")

# Set CITATION_DEBUG=1 to trace claim -> chunk -> page -> citation on stdout.
CITATION_DEBUG = os.getenv("CITATION_DEBUG", "").strip().lower() in ("1", "true", "yes")


def validate_citations(notes: str, chunk_map, page_count: int = 0) -> str:
    """Drop every [n] that cannot be resolved to real, in-range evidence.

    Extends enforce_citations with the page dimension. A citation survives only
    if n was actually retrieved AND -- when the document has pages at all --
    that chunk carries a page inside 1..page_count. The model never supplies a
    page: it names a passage, and the retriever's metadata decides what page
    that passage came from. This is the code guarantee behind "never invent a
    citation".
    """
    valid, dropped = set(), {}
    for cid, chunk in (chunk_map or {}).items():
        try:
            n = int(cid)
        except (TypeError, ValueError):
            continue
        page = chunk.get("page") if isinstance(chunk, dict) else None
        if page_count and page is None:
            dropped[n] = "no page metadata on a paged document"
            continue
        if page is not None and page_count and not 1 <= int(page) <= page_count:
            dropped[n] = f"page {page} outside 1..{page_count}"
            continue
        valid.add(n)

    if CITATION_DEBUG:
        for n, why in sorted(dropped.items()):
            print(f"[citation] [{n}] DROPPED: {why}")
        for n in sorted(valid):
            chunk = chunk_map.get(n) or {}
            excerpt = " ".join((chunk.get("text") or "")[:60].split())
            print(f"[citation] [{n}] -> p.{chunk.get('page')} :: {excerpt}…")

    return enforce_citations(notes, valid)


# ---------------------------------------------------------------------------
# Grounding: does the cited evidence actually SUPPORT the claim?
# ---------------------------------------------------------------------------
#
# validate_citations answers "is this a real, in-range page?". That is a
# different question from "does this passage say this?", and only the first
# was being asked -- so a claim could carry a perfectly resolving citation to
# a page that never makes it.

# A line is checked because it ASSERTS something, not because it happens to
# carry a citation. Selecting on "[n]" was a hole: an invented claim with no
# citation was invisible to this check AND to validate_citations, so nothing
# looked at it at all.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_RULE_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
_ALL_BOLD_RE = re.compile(r"^\*\*[^*]+\*\*:?$")
_BULLET_PREFIX = " \t-*•>0123456789.)"

# Below this, a line is a label or fragment rather than an assertion.
MIN_CLAIM_WORDS = int(os.getenv("GROUNDING_MIN_CLAIM_WORDS", "5"))


def _strip_markup(line: str) -> str:
    """The claim text: bullet/number prefix, citations and emphasis removed."""
    body = _CITATION_RE.sub("", line).strip().lstrip(_BULLET_PREFIX).strip()
    return body.replace("**", "").replace("`", "").strip()


def _is_structural(line: str) -> bool:
    """Headings, rules and short labels carry no factual assertion.

    These are excluded even when they carry a citation — a section title is not
    a claim, and spending verifier budget on it (or worse, deleting it) is wrong.
    """
    raw = line.strip()
    if not raw:
        return True
    if _MD_HEADING_RE.match(raw) or _RULE_RE.match(raw):
        return True
    body = _strip_markup(line)
    if not body:
        return True
    # "**Unit 2: Data Structures**" — a fully emphasised line is a heading.
    if _ALL_BOLD_RE.match(raw.lstrip(_BULLET_PREFIX).strip()):
        return True
    # "Lists:" / "Sets & frozensets" — a short label introducing what follows.
    if len(body.split()) < MIN_CLAIM_WORDS:
        return True
    return False


def _claim_lines(notes):
    """Substantive claim lines, with any citation ids they carry.

    Returns [(line_index, text, [cited ids])]; ids may be empty — an uncited
    line is still checked, against the full source rather than a named passage.
    """
    out = []
    for i, raw in enumerate((notes or "").split("\n")):
        ids = [int(m) for m in _CITATION_RE.findall(raw)]
        if _is_structural(raw):
            continue
        out.append((i, raw, list(dict.fromkeys(ids))))
    return out


# How much of the source is shown for judging UNCITED claims. They name no
# passage, so the whole retrieved context is the evidence — the same material
# the writer saw, so anything absent from it was not in the document.
GROUNDING_SOURCE_CHARS = int(os.getenv("GROUNDING_SOURCE_CHARS", "14000"))

# When the whole context does NOT fit that window (every windowed run on a
# large source), an uncited claim is judged against the passages retrieved FOR
# IT instead. The first GROUNDING_SOURCE_CHARS of a 300k-char context is a
# partial view presented as the whole source - the same class of bug as
# IncompleteStreamError - and "unsupported" on that view deleted true claims
# about later pages.
GROUNDING_RETRIEVE_K = int(os.getenv("GROUNDING_RETRIEVE_K", "4"))


def _source_body(chunk_map):
    """Every passage in the chunk map, in id order, as the judge sees it."""
    return "\n".join(
        f"[{cid}] {(chunk or {}).get('text', '')}"
        for cid, chunk in sorted((chunk_map or {}).items(), key=lambda kv: kv[0])
    )


def _evidence_block(items, chunk_map, uncited_evidence=None):
    """Per-claim evidence: the passages it cites, or 'the whole source'.

    `uncited_evidence` maps a claim's line index to the passages retrieved for
    it; it is only supplied when the whole source is too big to show.
    """
    blocks = []
    for n, (line_i, text, ids) in enumerate(items, start=1):
        claim = _strip_markup(text)
        if ids:
            evidence = "\n".join(
                f"  [{i}] {(chunk_map.get(i) or {}).get('text', '')}" for i in ids
            )
        elif uncited_evidence and line_i in uncited_evidence:
            evidence = (
                "  (no passage cited — these are the passages from the source "
                "most relevant to this claim; judge against them)\n"
                + "\n".join(f"  [{c['id']}] {c.get('text', '')}"
                            for c in uncited_evidence[line_i])
            )
        else:
            evidence = "  (no passage cited — judge against FULL SOURCE below)"
        blocks.append(f"CLAIM {n}: {claim}\nEVIDENCE:\n{evidence}")
    return "\n\n".join(blocks)


def _full_source_block(items, chunk_map, uncited_evidence=None):
    """The whole retrieved context, included only when some claim is uncited.

    Not included when uncited claims carry their own retrieved evidence: that
    happens only when the source is too big for this block, and a truncated
    copy labelled "every passage" is exactly the partial view to avoid.
    """
    if all(ids for _, _, ids in items):
        return ""
    if uncited_evidence is not None:
        return ""
    body = _source_body(chunk_map)
    return (
        "\n\nFULL SOURCE — every passage retrieved from the document. A claim "
        "with no cited passage must be judged against this, and ONLY this:\n"
        f"\"\"\"{body[:GROUNDING_SOURCE_CHARS]}\"\"\"\n"
    )


def _claim_evidence(item, chunk_map, uncited_evidence=None) -> str:
    """The exact evidence text a claim was judged against, for quote checks."""
    line_i, _text, ids = item
    if ids:
        return " ".join((chunk_map.get(i) or {}).get("text", "") for i in ids)
    if uncited_evidence and line_i in uncited_evidence:
        return " ".join(c.get("text", "") for c in uncited_evidence[line_i])
    return _source_body(chunk_map)[:GROUNDING_SOURCE_CHARS]


def _verdict_pass(items, chunk_map, model=None, uncited_evidence=None):
    """One verdict call: {n: (status, evidence_quote)} for the batch."""
    verdict_prompt = f"""You are the GROUNDING agent. For each claim decide whether the
evidence shown with it actually states or directly entails it.

- "supported": everything the claim asserts is present in its evidence.
  Paraphrase and summary are fine — wording need not match.
- "partial": the general point is there, but the claim adds specifics the
  evidence does not state (invented role names, numbers, steps, examples).
- "unsupported": the evidence does not state this at all. Evidence merely being
  ABOUT the topic is not support.

A claim whose EVIDENCE says "no passage cited" is judged against the FULL
SOURCE. If the source does not contain it, it is "unsupported" — however true
or standard the statement is. Terminology, syntax, code examples and technique
names that do not appear in the source are NOT supported by it.

For EVERY verdict also give "evidence_quote": the one sentence from THAT
claim's own evidence (its passages, or the FULL SOURCE for an uncited claim)
that comes closest to the claim, copied word for word - whatever the verdict.

Judge only against the evidence shown. Do not use outside knowledge.
Return a verdict for EVERY claim.

Respond with ONLY JSON:
{{"verdicts": [{{"n": 1, "status": "supported", "evidence_quote": "..."}}]}}

{_evidence_block(items, chunk_map, uncited_evidence)}{_full_source_block(items, chunk_map, uncited_evidence)}"""

    data = safe_json(call_model(verdict_prompt, max_tokens=900 + 60 * len(items),
                                model=model, temperature=0.0, json_mode=True))
    out = {}
    for v in (data.get("verdicts") or []):
        try:
            n = int(v.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        out[n] = (str(v.get("status", "")).lower(), str(v.get("evidence_quote") or ""))
    return out


def _verify_batch(items, chunk_map, model=None, uncited_evidence=None):
    """Rule on a batch of (claim, its own evidence) pairs.

    Returns {n: (status, fix)} where status is supported / partial /
    unsupported / unverified. Only verdicts that survived checking come back:

    - "unsupported" deletes the line downstream, so it needs evidence: the
      judge must quote the sentence of THAT claim's evidence closest to it,
      and the quote must really be there. A verdict without one - a bare
      label, an invented quote, or a quote from another claim's passages
      (a misnumbered verdict) - becomes "unverified", and the line is kept.
    - a "partial" rewrite is judged again, against the same evidence, before
      it may replace the line. Unless that re-check says "supported" with a
      real quote, the claim is "unverified" and the original stays.

    Verdicts and rewrites are separate calls on purpose. Asking one call for
    a verdict AND a rewritten sentence per claim overruns the output budget
    on a reasoning model: the JSON truncates and most claims come back
    unjudged, which silently lets a fabrication through.
    """
    raw = _verdict_pass(items, chunk_map, model, uncited_evidence)
    out = {}
    for n, (status, quote) in raw.items():
        if not 1 <= n <= len(items):
            continue
        if status == "unsupported" and not _evidence_supported(
                quote, _claim_evidence(items[n - 1], chunk_map, uncited_evidence)):
            status = "unverified"
        out[n] = (status, "")

    # Second pass only for claims that need a rewrite.
    partial = [n for n, (status, _) in out.items() if status == "partial"]
    if not partial:
        return out
    listing = "\n\n".join(
        f"CLAIM {n}: {items[n - 1][1].strip()}\nEVIDENCE:\n"
        + "\n".join(f"  [{i}] {(chunk_map.get(i) or {}).get('text', '')}"
                    for i in items[n - 1][2])
        for n in partial
    )
    fix_prompt = f"""Rewrite each claim so it says ONLY what its evidence supports.
Remove the unsupported specifics; keep the wording natural and keep the [n]
citation markers exactly as they appear. Do not add anything new.

Respond with ONLY JSON:
{{"fixes": [{{"n": 1, "text": "the corrected line"}}]}}

{listing}"""
    fixes = safe_json(call_model(fix_prompt, max_tokens=1200, model=model,
                                 temperature=0.0, json_mode=True))
    proposed = {}
    for f in (fixes.get("fixes") or []):
        try:
            n = int(f.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        text = str(f.get("text") or "").strip()
        if n in partial and text and _keeps_provenance(items[n - 1][1], text):
            proposed[n] = text

    # Re-check each rewrite against the evidence its original was judged on.
    for n in partial:
        if n not in proposed:
            out[n] = ("unverified", "")
    if proposed:
        order = sorted(proposed)
        recheck_items = [(items[n - 1][0], proposed[n], items[n - 1][2]) for n in order]
        again = _verdict_pass(recheck_items, chunk_map, model, uncited_evidence)
        for k, n in enumerate(order, 1):
            status, quote = again.get(k, ("", ""))
            ok = status == "supported" and _evidence_supported(
                quote, _claim_evidence(recheck_items[k - 1], chunk_map, uncited_evidence))
            out[n] = ("partial", proposed[n]) if ok else ("unverified", "")
    return out


def _keeps_provenance(original: str, fix: str) -> bool:
    """A repair may narrow a claim's citations, never change them.

    Every citation in the fix must be one the original line already had: the
    verdict was reached on THOSE passages, and a new number - even a real
    one - points at evidence nobody checked. Observed: [1] came back as
    [999], after validate_citations had already run. A cited claim must also
    keep at least one citation, because losing provenance is worse than
    leaving a slightly over-reaching claim. A claim that never carried a
    citation has none to lose, but may not gain one either.
    """
    had = {int(m) for m in _CITATION_RE.findall(original)}
    has = {int(m) for m in _CITATION_RE.findall(fix)}
    if not has <= had:
        return False
    return bool(has) or not had


def verify_claim_support(notes, chunk_map, model=None, batch_size=6, retriever=None):
    """Drop or repair claims their own cited evidence does not support.

    This is the difference between "is this citation a valid page?" (which
    validate_citations already answers) and "does this evidence actually say
    this?". A citation can resolve perfectly and still be attached to a claim
    the page never makes.

    Returns (notes, stats). Fails OPEN: if the model call fails or returns
    nothing usable, the notes are returned untouched rather than gutted.

    UNCITED claims are judged against the whole source when it fits
    GROUNDING_SOURCE_CHARS. When it does not, a truncated source would be a
    partial view, so each uncited claim is judged against the passages
    `retriever` finds for it instead; with no retriever (or nothing found) the
    claim is left as written and counted in `skipped_partial_view` - deleting
    on a view known to be partial is how true claims about later pages were
    lost.
    """
    items = _claim_lines(notes)
    # `unjudged` is tracked separately on purpose: a model call that fails or
    # returns nothing must not be indistinguishable from a clean pass.
    stats = {"checked": len(items), "supported": 0, "rewritten": 0,
             "removed": 0, "unjudged": 0, "uncited_retrieved": 0,
             "skipped_partial_view": 0, "unverified": 0, "unverified_lines": []}
    if not items:
        return notes, stats

    uncited_evidence = None
    if (any(not ids for _, _, ids in items)
            and len(_source_body(chunk_map)) > GROUNDING_SOURCE_CHARS):
        uncited_evidence, kept = {}, []
        for item in items:
            line_i, text, ids = item
            if ids:
                kept.append(item)
                continue
            found = []
            if retriever is not None:
                try:
                    # Only passages in chunk_map: the writer never saw the
                    # others, so they cannot be what a claim was drawn from.
                    found = [c for c in (retriever.retrieve(_strip_markup(text),
                                                            k=GROUNDING_RETRIEVE_K) or [])
                             if c.get("id") in (chunk_map or {})]
                except Exception as exc:  # noqa: BLE001
                    print(f"[grounding] evidence retrieval failed ({exc}).")
                    found = []
            if found:
                uncited_evidence[line_i] = [
                    {"id": c["id"], "text": (chunk_map[c["id"]] or {}).get("text", "")}
                    for c in found
                ]
                stats["uncited_retrieved"] += 1
                kept.append(item)
            else:
                stats["skipped_partial_view"] += 1
        if stats["skipped_partial_view"]:
            print(f"[grounding] {stats['skipped_partial_view']} uncited claim(s) not "
                  f"judged: the source exceeds the {GROUNDING_SOURCE_CHARS}-char view "
                  f"and no evidence was retrieved for them; left as written.")
        items = kept

    batches = [items[start:start + batch_size] for start in range(0, len(items), batch_size)]

    def _judge(batch):
        if uncited_evidence is None:
            return _verify_batch(batch, chunk_map, model=model)
        return _verify_batch(batch, chunk_map, model=model,
                             uncited_evidence=uncited_evidence)

    # Batches are independent: each is judged only against its own claims'
    # evidence, and no batch reads another's verdicts. So they run
    # concurrently, and the verdicts are merged below in batch order - the
    # same merge the sequential loop did.
    verdicts = {}
    for batch, (got, exc) in zip(batches, _run_bounded(_judge, batches,
                                                       GROUNDING_CONCURRENCY, "grounding")):
        if exc is not None:
            print(f"[grounding] batch failed ({exc}); keeping those claims as written.")
            continue
        for local_n, verdict in got.items():
            if 1 <= local_n <= len(batch):
                verdicts[batch[local_n - 1][0]] = verdict

    lines = notes.split("\n")
    for line_i, (status, fix) in verdicts.items():
        original = lines[line_i]
        if status == "unsupported":
            lines[line_i] = None
            stats["removed"] += 1
            if CITATION_DEBUG:
                print(f"[grounding] REMOVED unsupported: {original.strip()[:90]}")
        elif status == "partial" and fix.strip() and _keeps_provenance(original, fix):
            # Keep the original leading markup (bullet, indent) so the rewrite
            # doesn't fall out of the surrounding list.
            prefix = original[:len(original) - len(original.lstrip(" -•\t"))]
            lines[line_i] = prefix + fix.strip().lstrip("-•").lstrip()
            stats["rewritten"] += 1
            if CITATION_DEBUG:
                print(f"[grounding] REWROTE partial: {original.strip()[:70]}")
                print(f"[grounding]           -> {fix.strip()[:70]}")
        elif status in ("unverified", "partial"):
            # Judged, but nothing checkable backs acting on it (no real quote
            # for a deletion, or no re-checked rewrite): kept as written, and
            # said so rather than counted as supported.
            stats["unverified"] += 1
            stats["unverified_lines"].append(original.strip()[:300])
            if CITATION_DEBUG:
                print(f"[grounding] KEPT unverified: {original.strip()[:90]}")
        else:
            stats["supported"] += 1

    # Claims skipped for a partial view were never sent, so they count here too.
    stats["unjudged"] = stats["checked"] - len(verdicts)
    if stats["unjudged"]:
        print(f"[grounding] {stats['unjudged']} of {stats['checked']} claims were not "
              f"judged; those are left exactly as written.")
    return "\n".join(x for x in lines if x is not None), stats

# ---------------------------------------------------------------------------
# Notes windowing: never let head-truncation hide the tail of long notes
# ---------------------------------------------------------------------------

# Hard cap on notes fed to the in-pipeline FULL-REWRITE step (revise; the
# standalone rewrite / edit no longer truncate - see rewrite_notes).
# Sized to fit the largest sectioned output (8 sections x ~500 words ~ 28k
# chars) with headroom; a rewrite prompt must NEVER see a truncated copy,
# because the model can only return what it was shown -- truncation here
# silently deletes the tail of the document.
NOTES_REWRITE_CAP = int(os.getenv("NOTES_REWRITE_CAP", "30000"))


def _notes_excerpt(notes: str, max_chars: int) -> str:
    """Return the notes whole if they fit, else an EVEN SAMPLE across the
    entire notes. Used for read-only consumers (critique coverage, quiz,
    flashcards, chat): head-truncation (`notes[:N]`) made the tail of long
    notes invisible, producing false 'missing topic' flags and quizzes that
    never covered later sections."""
    notes = notes or ""
    if len(notes) <= max_chars:
        return notes
    n_seg = 6
    seg = max(1, max_chars // n_seg)
    parts = []
    # First n_seg-1 segments evenly spaced from the start of the notes...
    stride = max(1, (len(notes) - seg) // (n_seg - 1))
    for i in range(n_seg - 1):
        start = i * stride
        parts.append(notes[start:start + seg])
    # ...and the LAST segment anchored to the very END, so the tail of the
    # notes is always represented.
    parts.append(notes[-seg:])
    return "\n[...]\n".join(p for p in parts if p)


# Document genre. The gatekeeper already asks "is this academic?"; it now also
# reports what KIND of document it is, because a question bank and a lecture
# handout are both academic and must not be written up the same way.
#
# The vocabulary is deliberately small at the point where behaviour branches:
# every task-shaped genre maps to one writing mode. The finer label is kept for
# display and for future use.
TASK_DOC_TYPES = frozenset(
    {"question_bank", "assignment", "exam", "worksheet", "syllabus"}
)
KNOWN_DOC_TYPES = frozenset(
    {"explanatory", "mixed", "other"} | TASK_DOC_TYPES
)


def is_task_document(doc_type: str) -> bool:
    return (doc_type or "").strip().lower() in TASK_DOC_TYPES


_TASK_RULE = """
DOCUMENT TYPE — this source is a list of TASKS/QUESTIONS, not a set of
explanations. It tells the learner what to DO; it does not teach the concepts.
Write accordingly:
- Organise and group what the document ASKS FOR. Preserve its unit/section
  structure and numbering where it has one.
- Name the topic each task covers ("Topic covered: single inheritance"), rather
  than explaining that topic.
- Do NOT convert a task into a statement of fact. "Write a program to implement
  single inheritance" must NOT become "Single inheritance lets one class inherit
  from another" — true or not, the document did not say it.
- Do NOT add definitions, worked examples, code snippets, syntax, or technical
  terminology the document does not itself contain. If a term does not appear in
  the CONTEXT, do not introduce it.
- The reader must be able to tell what the DOCUMENT contains from what you
  write. Never imply it explained something it only asked for.
"""

_MIXED_RULE = """
DOCUMENT TYPE — this source MIXES explanation with exercises/questions. Keep the
two apart:
- Explanatory passages become normal study notes.
- Tasks and questions are reported as tasks ("the document asks the learner
  to…"), never rewritten into statements of fact.
- Do not use an explanation from your own knowledge to fill in what a task only
  names.
"""


def _doc_type_rule(doc_type: str) -> str:
    """Writer instruction for this genre. Explanatory returns "" so the
    existing behaviour for ordinary lecture/textbook PDFs is byte-identical."""
    dt = (doc_type or "explanatory").strip().lower()
    if is_task_document(dt):
        return _TASK_RULE
    if dt == "mixed":
        return _MIXED_RULE
    return ""


_CITE_RULE = (
    "Support each point with a citation to the passage number(s) it came from, "
    "in square brackets right after the point, e.g. [1] or [2][5]. Only cite "
    "numbers that appear in the CONTEXT. Do not invent citations.\n\n"
    "GROUNDING — the CONTEXT is the only source of truth:\n"
    "- Every factual statement must come from the CONTEXT passages. If it is "
    "not there, leave it out.\n"
    "- Do NOT add facts from your own knowledge of the subject, however "
    "standard or obviously true they seem.\n"
    "- Do NOT invent examples, role names, numbers, steps, processes or "
    "terminology that the passages do not state.\n"
    "- Do NOT infer unstated detail. Naming a technique is not licence to "
    "describe how it usually works.\n"
    "- A citation means \"this passage states this\", not \"this passage is "
    "about this topic\". If a passage only mentions the topic, say only what "
    "it actually says.\n"
    "- Paraphrasing and summarising ARE wanted: write clearly in your own "
    "words. Faithful summary, not transcription, and not elaboration."
)

# ---------------------------------------------------------------------------
# Tuning tables
# ---------------------------------------------------------------------------

LENGTH_TARGETS = {
    "short": "250-350 words",
    "medium": "500-700 words",
    "long": "1000-1500 words",
}

# Output token budget per length (enough to finish without truncation).
# Roughly 1.6 tokens/word plus headroom for markdown, citations, and headers.
LENGTH_MAX_TOKENS = {
    "short": 1200,
    "medium": 2200,
    "long": 4096,
    "xl": 6000,  # used when revising long sectioned notes
}


def _max_tokens(length: str) -> int:
    return LENGTH_MAX_TOKENS.get((length or "medium").lower(), LENGTH_MAX_TOKENS["medium"])


# Per-SECTION word budgets for long documents (map-reduce path).
SECTION_WORDS = {
    "short": "120-180 words",
    "medium": "220-320 words",
    "long": "350-500 words",
}
SECTION_MAX_TOKENS = {
    "short": 700,
    "medium": 1100,
    "long": 1600,
}

MODE_GUIDANCE = {
    "exam": "Focus on exam-critical facts, definitions, formulas, and likely "
    "test questions. Be concise and high-yield.",
    "revision": "Optimise for quick last-minute revision: key points, memory "
    "hooks, and easily scannable structure.",
    "deep study": "Explain concepts thoroughly with reasoning, context, and "
    "connections between ideas. Prioritise understanding.",
    "summary": "Produce a faithful, compact summary that captures the core "
    "ideas without losing essential detail.",
}

TONE_GUIDANCE = {
    "academic": "Use precise academic language.",
    "formal": "Use clear, professional, formal language.",
    "casual": "Use a friendly, conversational tone.",
    "simple": "Use simple, plain language a beginner can follow.",
}


def _format_instructions(fmt: str) -> str:
    fmt = (fmt or "bullet").lower()
    if fmt == "numbered":
        return (
            "FORMAT: Use a numbered list. Start each point with `1.`, `2.`, etc. "
            "Group points under bold section headers written as `**Header:**` on "
            "their OWN line (never put list content on the same line as a header)."
        )
    if fmt == "paragraph":
        return (
            "FORMAT: Write in short paragraphs. Begin each section with a bold "
            "header written as `**Header:**` on its OWN line, followed by the "
            "paragraph text. Use inline `**bold**` to emphasise key terms."
        )
    return (
        "FORMAT: Use bullet points. Start each bullet with `• `. Group bullets "
        "under bold section headers written as `**Header:**` on their OWN line. "
        "Use `- ` for sub-points indented under a bullet. Use inline `**bold**` "
        "for key terms. You may use `$...$` for inline math and triple-backtick "
        "fenced blocks for code."
    )


def _instr_block(instructions: str) -> str:
    instructions = (instructions or "").strip()
    if not instructions:
        return ""
    return f"\nADDITIONAL USER INSTRUCTIONS (you MUST follow these):\n{instructions}\n"


# ---------------------------------------------------------------------------
# Agent: Digest (full-coverage topic scan for very large documents)
# ---------------------------------------------------------------------------

def digest_document(text, model=None) -> str:
    """Scan the ENTIRE document segment by segment and return a merged topic
    inventory. Runs on the cheap helper model (its own token quota), so 100%
    coverage costs nothing from the main model's budget. A failed segment is
    skipped rather than failing the run — partial inventory still beats none."""
    segments = [
        text[i:i + DIGEST_SEGMENT_CHARS]
        for i in range(0, len(text), DIGEST_SEGMENT_CHARS)
    ][:DIGEST_MAX_SEGMENTS]

    def _scan(numbered):
        i, seg = numbered
        prompt = f"""You are scanning part {i} of {len(segments)} of a document to build a
topic inventory. List the distinct topics and key concepts covered in THIS part.

Respond with ONLY a bullet list (max 10 bullets). Each bullet is a short,
specific topic phrase (3-8 words). No commentary, no numbering, no headers.

PART {i}:
\"\"\"{seg}\"\"\""""
        return call_model(prompt, max_tokens=250, model=model, temperature=0.1)

    # Segments are independent (each prompt holds only its own slice), so
    # they are scanned concurrently and joined back in document order.
    inventory = []
    for out, exc in _run_bounded(_scan, list(enumerate(segments, 1)),
                                 DIGEST_CONCURRENCY, "digest"):
        if exc is not None:
            continue
        if out and out.strip():
            inventory.append(out.strip())
    return "\n".join(inventory)


# ---------------------------------------------------------------------------
# Agent: Plan
# ---------------------------------------------------------------------------

def plan_outline(text, mode, tone, length, model=None, instructions="", doc_chars=None, topic_inventory="") -> dict:
    doc_chars = doc_chars or len(text or "")
    if doc_chars > 120000:
        outline_rule = (
            "This is a LARGE document (a book chapter or long report). Produce "
            "8-12 outline sections that together cover ALL of its major topics — "
            "do not skip parts of the document."
        )
    elif doc_chars > SECTION_DOC_THRESHOLD:
        outline_rule = (
            "This is a substantial document. Produce 5-8 outline sections that "
            "together cover ALL of its major topics — do not skip parts."
        )
    else:
        outline_rule = "Produce 3-5 outline sections covering the material."

    inventory_block = ""
    if (topic_inventory or "").strip():
        inventory_block = f"""
TOPIC INVENTORY — built by scanning the ENTIRE document part by part. This is
the complete list of topics the document contains. Your outline MUST cover all
major topics below (group closely related ones under one section):
{topic_inventory[:8000]}
"""

    prompt = f"""You are the PLANNING agent in a notes-generation pipeline.

Analyse the SOURCE material (an even sample spanning the WHOLE document) and
produce a study plan. {outline_rule}
{inventory_block}
Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone}
Target length: {LENGTH_TARGETS.get(length.lower(), '300-400 words')}
{_instr_block(instructions)}
Respond with ONLY a JSON object (no prose, no code fences) of this exact shape:
{{
  "outline": ["section heading 1", "section heading 2", "..."],
  "checklist": ["key point that MUST be covered", "..."],
  "difficulty": "beginner | intermediate | advanced",
  "suggested_format": "bullet | numbered | paragraph"
}}

SOURCE:
\"\"\"{text[:24000]}\"\"\""""

    data = safe_json(call_model(prompt, max_tokens=700, model=model, temperature=0.1, json_mode=True))

    if not data or "outline" not in data:
        return {
            "outline": ["Overview", "Key Concepts", "Important Details", "Summary"],
            "checklist": ["Define core terms", "Cover main ideas", "Highlight key takeaways"],
            "difficulty": "intermediate",
            "suggested_format": "bullet",
        }

    data.setdefault("outline", ["Overview", "Key Concepts", "Summary"])
    data.setdefault("checklist", ["Cover main ideas"])
    data.setdefault("difficulty", "intermediate")
    data.setdefault("suggested_format", "bullet")
    return data


# ---------------------------------------------------------------------------
# Agent: Write (prompt builder + streaming)
# ---------------------------------------------------------------------------

def _write_prompt(context, mode, tone, length, fmt, plan, instructions="",
                  doc_type="explanatory") -> str:
    outline = plan.get("outline", [])
    checklist = plan.get("checklist", [])
    outline_str = "\n".join(f"- {o}" for o in outline) if outline else "- (derive a sensible outline)"
    checklist_str = "\n".join(f"- {c}" for c in checklist) if checklist else ""

    return f"""You are the WRITING agent in a notes-generation pipeline.

Write high-quality study notes grounded in the CONTEXT passages below.

Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone} — {TONE_GUIDANCE.get(tone.lower(), '')}
Length target: {LENGTH_TARGETS.get(length.lower(), '500-700 words')}. Treat this as a
SOFT target and aim for the MIDDLE of the range rather than either edge. It ranks below
the things that matter more — in this order:

  1. Factual faithfulness — every claim supported by the CONTEXT.
  2. Coverage of the important concepts.
  3. Concise explanation — say a thing once, clearly.
  4. Hitting the length.

Never pad, restate or add filler to reach the lower bound. If the source genuinely does
not contain enough substantive material, finishing below the range is the right answer.
When it does contain enough, use the room rather than stopping early.

Cover every outline point, but select what earns the space rather than summarising every
passage you were given.

Prioritise, in this order: core concepts; important definitions; major workflows and
worked examples; important comparisons; key numbers and results; named design patterns.
Leave out section-divider slides, decorative material, repeated examples and restatements
of something you already made. More evidence than you need is supplied on purpose — a
page-by-page transcript is a worse answer than a well-chosen summary.

Follow this outline:
{outline_str}

Make sure you cover these points:
{checklist_str}

{_format_instructions(fmt)}
{_doc_type_rule(doc_type)}
{_CITE_RULE}
{_instr_block(instructions)}
Write ONLY the notes themselves — no preamble, no closing remarks.

CONTEXT (numbered passages — cite these):
{context[:40000]}"""


def write_notes(context, mode, tone, length, fmt, plan, model=None, instructions="") -> str:
    return call_model(
        _write_prompt(context, mode, tone, length, fmt, plan, instructions),
        max_tokens=_max_tokens(length),
        model=model,
    )


def write_notes_stream(context, mode, tone, length, fmt, plan, model=None, instructions="",
                       doc_type="explanatory", on_serve=None, cancel=None):
    yield from call_model_stream(
        _write_prompt(context, mode, tone, length, fmt, plan, instructions, doc_type=doc_type),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.5,
        on_serve=on_serve,
        **_cancel_kw(cancel),
    )


# ---------------------------------------------------------------------------
# Agent: Section writer (map-reduce path for long documents)
# ---------------------------------------------------------------------------

def write_section_stream(section, context, mode, tone, length, fmt, checklist=None, model=None,
                         instructions="", doc_type="explanatory", on_serve=None,
                         is_part=False, cancel=None):
    """Write ONE unit of a larger set of notes from its own context (streamed).

    `is_part` distinguishes the two callers. An outline section has a topical
    title the writer can be told to write about; a coverage window is named by
    the pages it spans, which is not a topic at all. Telling the model to write
    'the section titled "Pages 8-10"' made one window narrate its own confusion
    into the notes ("Now, I need to write the section..."), so a part gets a
    prompt that describes the actual job.
    """
    related = "\n".join(f"- {c}" for c in (checklist or [])[:6])
    words = SECTION_WORDS.get((length or "medium").lower(), "120-180 words")

    # Windows are BUILT to fit WRITER_CONTEXT_CHARS, so this is unreachable by
    # construction. If it ever fires, say so loudly and send the context in full
    # anyway: dropping source the window promised to cover is the bug this
    # replaces, and a silent fallback would reintroduce it.
    if len(context) > WRITER_CONTEXT_CHARS:
        print(f"[coverage] WARNING: window context is {len(context)} chars, over "
              f"the {WRITER_CONTEXT_CHARS} budget; sending in full, not truncating.")

    if is_part:
        intro = f"""You are the WRITING agent producing ONE PART of a larger set of
study notes. This part covers {section} of the source document. Write notes on
the material in the CONTEXT below and nothing else.

Give this part your own short topical headings taken from the material. Do NOT
mention page numbers, part numbers, or these instructions in the notes, and do
NOT narrate your own process — no "the passages show", no "I need to", no
commentary about what you can or cannot find. Write the notes themselves, with
no preamble."""
    else:
        intro = f"""You are the WRITING agent producing ONE SECTION of a larger set of
study notes. Write ONLY the body of the section titled "{section}" — do NOT
repeat the section title, do NOT write other sections, no preamble."""

    prompt = f"""{intro}

Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone} — {TONE_GUIDANCE.get(tone.lower(), '')}
Section length: {words}. Treat this as a minimum — cover this section's points
thoroughly with concrete facts and explanations from the context. Prefer more
detail over brevity.

Cover any of these plan points that belong to this section:
{related or '- (use your judgment)'}

{_format_instructions(fmt)}
{_doc_type_rule(doc_type)}
{_CITE_RULE}
{_instr_block(instructions)}
CONTEXT (numbered passages retrieved for THIS section — cite these):
{context}"""

    yield from call_model_stream(
        prompt,
        max_tokens=SECTION_MAX_TOKENS.get((length or "medium").lower(), 750),
        model=model,
        temperature=0.5,
        on_serve=on_serve,
        **_cancel_kw(cancel),
    )


# ---------------------------------------------------------------------------
# Agent: Critique
# ---------------------------------------------------------------------------

# Judge evidence for the critique and the reviser on large sources. The whole
# context of a windowed run (up to ~300k chars) cannot be shown, and a head
# slice of it (`source[:12000]`, `context[:40000]`) was a partial view presented
# as the ground truth - the same class of bug as IncompleteStreamError: true
# claims about later pages looked "unsupported" because their passages were
# never shown. Instead the judge is shown the passages the notes CITE, whole.
CRITIQUE_SOURCE_CHARS = 12000  # head slice kept when no citation-based view applies
# Same budget as the reviser: both judge the same notes. Notes on a large
# source routinely cite dozens of passages (40 x ~700 chars ~ 29k); a 24k
# budget showed only 0.825 of cited evidence on the synthetic long-document
# test, and these providers have ample context for 40k.
CRITIQUE_CONTEXT_CHARS = int(os.getenv("CRITIQUE_CONTEXT_CHARS", "40000"))
REVISE_CONTEXT_CHARS = int(os.getenv("REVISE_CONTEXT_CHARS", "40000"))

_PASSAGE_HEAD_RE = r"(?m)^\[{}\] "


def _cited_ids(text, chunk_map):
    """Citation ids in `text` that name a passage in `chunk_map`, in id order."""
    return sorted({int(m) for m in _CITATION_RE.findall(text or "")} & set(chunk_map or {}))


def _passages_within(ids, chunk_map, budget, priority=()):
    """Render whole passages for `ids` (id order) within `budget` chars.

    Passages are never cut: a half passage can hide exactly the sentence a
    claim rests on. What does not fit is returned as `omitted` so the prompt
    can SAY its view is partial. `priority` ids are admitted first.
    Returns (context, shown_ids, omitted_ids).
    """
    order = list(dict.fromkeys([i for i in priority if i in ids] + list(ids)))
    shown, used = set(), 0
    for i in order:
        size = len(f"[{i}] {(chunk_map.get(i) or {}).get('text', '')}") + 2
        if used + size > budget:
            continue  # a smaller passage later may still fit
        shown.add(i)
        used += size
    shown_sorted = sorted(shown)
    context = _format_context(
        [{"id": i, "text": (chunk_map.get(i) or {}).get("text", "")} for i in shown_sorted])
    return context, shown_sorted, sorted(set(ids) - shown)


def _numbered_lines(lines) -> str:
    """Notes with each non-empty line prefixed "L<n>: " (1-based)."""
    return "\n".join(f"L{i}: {ln}" if ln.strip() else ln
                     for i, ln in enumerate(lines, 1))


def _critique_flag(flag, notes_lines):
    """(claim text, cited ids) for one unsupported_claims entry.

    An entry may be a string (ids read from any [n] in it) or an object with
    "claim", "line" and/or "citations". The line's own citations are the
    claim's provenance, so a valid line number is read from the notes rather
    than trusted to the critic's quoting. Without a valid line, explicit
    citation ids and any [n] in the text are used.
    """
    if not isinstance(flag, dict):
        text = str(flag or "").strip()
        return text, {int(m) for m in _CITATION_RE.findall(text)}
    text = str(flag.get("claim") or flag.get("text") or "").strip()
    ids = {int(m) for m in _CITATION_RE.findall(text)}
    for c in flag.get("citations") or []:
        try:
            ids.add(int(c))
        except (TypeError, ValueError):
            continue
    try:
        line = int(flag.get("line"))
    except (TypeError, ValueError):
        line = 0
    if 1 <= line <= len(notes_lines):
        # The notes, not the critic, say what the line cites - including
        # that it cites nothing, which makes it grounding's to judge.
        ids = {int(m) for m in _CITATION_RE.findall(notes_lines[line - 1])}
        if not text:
            text = _strip_markup(notes_lines[line - 1])
    return text, ids


def critique_notes(notes, plan, mode, source="", model=None, doc_sample="",
                   chunk_map=None) -> dict:
    """
    Grounded critique. FAITHFULNESS is judged against `source` — the numbered
    CONTEXT passages the writer was actually given (the ground truth for what
    the notes were allowed to claim). COVERAGE is judged against `doc_sample`,
    an even breadth sample of the wider document, so topics the retrieval
    missed can still be reported as missing.

    `needs_revision` triggers on real signal only: the model's own flag, a
    score below threshold, or unsupported claims. `missing_topics` alone is
    advisory — it feeds corrective re-retrieval, not an automatic rewrite.

    With `chunk_map`, a source too big to show whole is replaced by the
    passages CITED in the notes excerpt the critique sees (whole passages, up
    to CRITIQUE_CONTEXT_CHARS). `evidence_shown` / `evidence_cited` in the
    result make a partial view observable instead of silent.
    """
    notes_excerpt = _notes_excerpt(notes, 20000)
    cited_all = sorted({int(m) for m in _CITATION_RE.findall(notes_excerpt)})
    context_text = source[:CRITIQUE_SOURCE_CHARS]
    shown_ids = None  # set only when judging against cited passages
    omitted_ids = []
    if chunk_map and len(source or "") > CRITIQUE_SOURCE_CHARS:
        ids = _cited_ids(notes_excerpt, chunk_map)
        if ids:
            context_text, shown, omitted_ids = _passages_within(
                ids, chunk_map, CRITIQUE_CONTEXT_CHARS)
            shown_ids = set(shown)
    notes_lines = (notes or "").split("\n")
    if shown_ids is not None:
        # Flags are acted on here only if they can be tied to the passages a
        # claim cites. Critics quote claims WITHOUT their [n] markers, so the
        # notes are numbered and the critic names the line instead.
        notes_excerpt = _notes_excerpt(_numbered_lines(notes_lines), 20000)
    evidence_shown = sum(
        1 for i in cited_all if re.search(_PASSAGE_HEAD_RE.format(i), context_text))

    checklist = plan.get("checklist", [])
    checklist_str = "\n".join(f"- {c}" for c in checklist) if checklist else "(none)"

    sample_block = ""
    if (doc_sample or "").strip():
        sample_block = f"""
DOCUMENT SAMPLE — an even sample of the wider document, for judging COVERAGE
only (a topic present here but absent from the notes may be a missing topic;
do NOT use this block to judge faithfulness):
\"\"\"{doc_sample[:8000]}\"\"\"
"""

    faithfulness = """1. FAITHFULNESS — does every claim in the notes actually appear in / follow from
   the CONTEXT passages below? The CONTEXT is the ONLY ground truth for this:
   list any statement that is fabricated, distorted, or unsupported by it."""
    context_header = "CONTEXT (the passages the notes were written from — the ground truth):"
    if shown_ids is not None:
        shown_str = ", ".join(str(i) for i in sorted(shown_ids))
        omitted_str = ""
        if omitted_ids:
            omitted_str = (
                "\n   Passages cited by the notes but NOT shown (over budget): "
                + ", ".join(str(i) for i in omitted_ids)
                + ". Do NOT list a claim citing only these as unsupported.")
        # Uncited claims are not judged here on purpose: their evidence could be
        # anywhere in a document too big to show, and the grounding step
        # (verify_claim_support) checks each one against passages retrieved for it.
        faithfulness = f"""1. FAITHFULNESS — the CONTEXT below holds the passages the notes CITE
   (passage ids shown: {shown_str}), not the whole document.{omitted_str}
   List a claim as unsupported ONLY if the passage it cites IS shown and does
   not support it (fabricated, distorted, or not stated there). List an
   uncited claim only if the shown CONTEXT contradicts it — uncited claims are
   verified separately.
   Each NOTES line starts with its number ("L12: "). Report every unsupported
   claim as an object naming that line and the passage ids it cites:
   {{"line": 12, "claim": "the claim, quoted", "citations": [3]}}"""
        context_header = ("CONTEXT (the passages cited by the notes, by id — the ground "
                          "truth for those claims):")

    prompt = f"""You are the CRITIQUE agent in a notes-generation pipeline. Be a
strict, fair reviewer for the "{mode}" study mode.

Judge the NOTES on THREE things:
{faithfulness}
2. COVERAGE — are any important points from the checklist or the document
   missing from the notes?
3. QUALITY — clarity, structure, and usefulness for studying.

Checklist that should be covered:
{checklist_str}

Respond with ONLY a JSON object of this exact shape:
{{
  "score": <integer 1-10>,
  "needs_revision": <true|false>,
  "unsupported_claims": ["claim in the notes NOT supported by the context", "..."],
  "missing_topics": ["important point the notes omitted", "..."],
  "issues": ["other quality problem", "..."],
  "strengths": ["what was done well", "..."]
}}

Scoring: deduct heavily for any unsupported_claims (faithfulness matters most).
A score of {REVISE_THRESHOLD} or above with NO unsupported claims means no
revision is needed.

{context_header}
\"\"\"{context_text}\"\"\"
{sample_block}
NOTES:
\"\"\"{notes_excerpt}\"\"\""""

    data = safe_json(
        call_model(prompt, max_tokens=700, model=model, temperature=0.1, json_mode=True)
    )

    # Conservative fallback: if we can't parse the critique, assume revision is
    # needed rather than silently passing.
    if not data or "score" not in data:
        return {
            "score": 5,
            "needs_revision": True,
            "unsupported_claims": [],
            "missing_topics": [],
            "issues": ["Critique could not be parsed; revising to be safe."],
            "strengths": [],
            "evidence_shown": evidence_shown,
            "evidence_cited": len(cited_all),
            "deferred_to_grounding": [],
        }

    try:
        score = int(data.get("score", 5))
    except (TypeError, ValueError):
        score = 5
    score = max(1, min(10, score))

    flags = [_critique_flag(u, notes_lines)
             for u in (data.get("unsupported_claims", []) or [])]
    flags = [f for f in flags if f[0]]
    unsupported = [text for text, _ids in flags]
    deferred = []
    if shown_ids is not None:
        # Code guarantees behind the prompt rules - a prompt rule is not relied
        # on where a deterministic guard is possible, since a flag here makes
        # the reviser delete the claim. A claim whose cited passages were ALL
        # withheld cannot have been judged, so it is not flagged. An UNCITED
        # claim's evidence could be anywhere in a source too big to show, so
        # grounding owns it (judged against passages retrieved for it); the
        # flag is deferred, and kept visible, rather than acted on here.
        deferred = [text for text, ids in flags if not ids]
        dropped = [text for text, ids in flags if ids and not (ids & shown_ids)]
        if dropped:
            print(f"[critique] ignored {len(dropped)} unsupported flag(s) on claims "
                  f"whose cited passages were not shown.")
        unsupported = [text for text, ids in flags if ids & shown_ids]
    missing = data.get("missing_topics", []) or []
    # missing_topics is deliberately NOT a trigger on its own — an LLM critic
    # almost always lists something, which previously forced a revision on
    # nearly every run. Missing topics instead drive corrective re-retrieval
    # inside the revise loop when a revision does happen.
    needs = (
        bool(data.get("needs_revision", False))
        or score < REVISE_THRESHOLD
        or len(unsupported) > 0
    )

    return {
        "score": score,
        "needs_revision": needs,
        "unsupported_claims": unsupported,
        "missing_topics": missing,
        "issues": data.get("issues", []) or [],
        "strengths": data.get("strengths", []) or [],
        "evidence_shown": evidence_shown,
        "evidence_cited": len(cited_all),
        "deferred_to_grounding": deferred,
    }


# ---------------------------------------------------------------------------
# Agent: Revise (prompt builder + streaming)
# ---------------------------------------------------------------------------

def _revise_prompt(notes, critique, mode, plan, fmt, context="", instructions="",
                   chunk_map=None, added_ids=()) -> str:
    issues = critique.get("issues", [])
    missing = critique.get("missing_topics", [])
    unsupported = critique.get("unsupported_claims", [])
    issues_str = "\n".join(f"- {i}" for i in issues) if issues else "- (general polish)"
    missing_str = "\n".join(f"- {m}" for m in missing) if missing else "- (none)"
    unsupported_str = "\n".join(f"- {u}" for u in unsupported) if unsupported else "- (none)"

    context_block = (
        f"\n\nCONTEXT (numbered passages — the ONLY source of truth; cite by number):\n{context[:REVISE_CONTEXT_CHARS]}"
        if context
        else ""
    )
    # A context too big to show whole is replaced by the passages the notes
    # cite plus those corrective re-retrieval just added - not its head slice,
    # which hid the evidence for every later page from the reviser (the
    # partial-view bug; see CRITIQUE_CONTEXT_CHARS).
    if context and chunk_map and len(context) > REVISE_CONTEXT_CHARS:
        added = [i for i in (added_ids or ()) if i in chunk_map]
        ids = sorted(set(_cited_ids(notes[:NOTES_REWRITE_CAP], chunk_map)) | set(added))
        if ids:
            body, _shown, omitted = _passages_within(
                ids, chunk_map, REVISE_CONTEXT_CHARS, priority=added)
            omitted_str = (
                "\nPassages cited but NOT shown (over budget): "
                + ", ".join(str(i) for i in omitted)
                + ". Keep claims citing them as they are.") if omitted else ""
            context_block = (
                "\n\nCONTEXT (numbered passages — the ONLY source of truth; cite by "
                "number). These are the passages the notes cite plus any retrieved "
                f"for missing topics, not the whole document.{omitted_str}\n{body}")

    return f"""You are the REVISION agent in a notes-generation pipeline.

Improve the NOTES below. Keep the same study mode ("{mode}") and formatting.

REMOVE or CORRECT these unsupported/fabricated claims (NOT in the context):
{unsupported_str}

ADD these missing topics (they ARE in the context):
{missing_str}

Also fix these quality issues:
{issues_str}

Rules: every claim must be grounded in the CONTEXT. Do not invent facts. Keep
existing correct content. {_CITE_RULE}

{_format_instructions(fmt)}
{_instr_block(instructions)}
Return ONLY the full, revised notes — no commentary.
{context_block}

CURRENT NOTES:
\"\"\"{notes[:NOTES_REWRITE_CAP]}\"\"\""""


def revise_notes(notes, critique, mode, plan, fmt, model=None, instructions="", context="", length="medium",
                 chunk_map=None, added_ids=()) -> str:
    return call_model(
        _revise_prompt(notes, critique, mode, plan, fmt, context, instructions,
                       chunk_map=chunk_map, added_ids=added_ids),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.4,
    )


def revise_notes_stream(notes, critique, mode, plan, fmt, model=None, instructions="", context="",
                        length="medium", on_serve=None, chunk_map=None, added_ids=(), cancel=None):
    yield from call_model_stream(
        _revise_prompt(notes, critique, mode, plan, fmt, context, instructions,
                       chunk_map=chunk_map, added_ids=added_ids),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.4,
        on_serve=on_serve,
        **_cancel_kw(cancel),
    )


# ---------------------------------------------------------------------------
# Agent: Rewrite (shorter / longer)
# ---------------------------------------------------------------------------

# Rewrite and inline edit REPLACE the user's notes with the model's answer, so
# a partial answer is data loss - the same bug class as IncompleteStreamError.
# Both used to send notes[:NOTES_REWRITE_CAP] and ask for the whole document
# back: anything past the cap, or past the output budget, silently vanished.
# Now neither ever truncates its input, and every call is strict (a completion
# cut at the token cap raises instead of being returned as if complete).

# Notes longer than this are rewritten in parts. Sized so a part's expected
# output fits its token budget with headroom even for "longer" (1.5x):
# 8000 chars ~ 2.7k tokens in, ~4k out, under the 4500-token "longer" cap.
REWRITE_PART_CHARS = int(os.getenv("REWRITE_PART_CHARS", "8000"))
# Bound the number of sequential calls one click can trigger (12 x 8000 =
# 96k chars); longer notes get a clear refusal instead of a very slow request.
REWRITE_MAX_PARTS = int(os.getenv("REWRITE_MAX_PARTS", "12"))

_HEADING_LINE = re.compile(r"^#{1,6}\s")


class RewriteTooLongError(ValueError):
    """The notes would need more than REWRITE_MAX_PARTS rewrite calls."""


def _pack(pieces, limit):
    """Greedily join consecutive pieces into chunks of at most `limit` chars
    (a single piece longer than `limit` stays whole as its own chunk)."""
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > limit:
            out.append(cur)
            cur = ""
        cur += p
    if cur:
        out.append(cur)
    return out


def _split_for_rewrite(notes: str, limit: int):
    """Split notes into parts of <= `limit` chars that tile the text exactly.

    Boundaries prefer markdown headings (so each part is whole sections),
    then blank-line paragraph breaks, then line breaks. Never mid-line: a
    single line longer than `limit` becomes its own (oversized) part.
    """
    lines = notes.splitlines(keepends=True)
    sections, cur = [], ""
    for ln in lines:
        if _HEADING_LINE.match(ln) and cur:
            sections.append(cur)
            cur = ""
        cur += ln
    if cur:
        sections.append(cur)

    pieces = []
    for sec in sections:
        if len(sec) <= limit:
            pieces.append(sec)
            continue
        # Paragraphs: split after each blank line, keeping the text intact.
        paras = [p for p in re.split(r"(?<=\n\n)", sec) if p]
        for para in paras:
            if len(para) <= limit:
                pieces.append(para)
            else:
                pieces.extend(_pack(para.splitlines(keepends=True), limit))
    # A section that fits is one piece, so a part only starts mid-section
    # when that section is itself longer than `limit`.
    return _pack(pieces, limit)


def _rewrite_budget(chars: int, direction: str) -> int:
    # Budget scales with input so the model can return the FULL rewritten text
    # (a fixed budget silently truncated long rewrites).
    est_tokens = max(1, chars) // 3
    if direction == "longer":
        return min(4500, max(3000, est_tokens * 2))
    return min(3500, max(1800, est_tokens))


def rewrite_notes(notes, direction, mode="exam", tone="academic", fmt="bullet", model=None) -> str:
    if direction == "shorter":
        change = "Condense these notes to roughly half the length, keeping only the most important points."
    elif direction == "longer":
        change = "Expand these notes with more detail, examples, and explanation — roughly 1.5x longer."
    else:
        change = "Rewrite these notes to improve clarity while keeping the same length."

    if len(notes) <= REWRITE_PART_CHARS:
        parts = [notes]
    else:
        parts = _split_for_rewrite(notes, REWRITE_PART_CHARS)
    if len(parts) > REWRITE_MAX_PARTS:
        raise RewriteTooLongError(len(notes))

    out = []
    for i, part in enumerate(parts, 1):
        where = ""
        if len(parts) > 1:
            where = (f"\nThis is part {i} of {len(parts)} of a longer set of notes. "
                     "Rewrite ONLY this part. Do not add an introduction or a "
                     "conclusion, and keep its headings.\n")
        prompt = f"""You are the REWRITING agent. {change}
{where}
Keep the "{mode}" study focus and a {tone} tone.

{_format_instructions(fmt)}

Return ONLY the rewritten notes — no commentary.

NOTES:
\"\"\"{part}\"\"\""""
        # strict: a part cut at the token cap raises (and so does any other
        # failure), which aborts the whole rewrite - partially rewritten notes
        # are never returned.
        out.append(call_model(prompt, max_tokens=_rewrite_budget(len(part), direction),
                              model=model, strict=True).strip())
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Inline edit: apply an instruction to a selected passage
# ---------------------------------------------------------------------------

# Surrounding notes shown with the selection, for coherence only.
EDIT_CONTEXT_CHARS = int(os.getenv("EDIT_CONTEXT_CHARS", "1500"))
# Most source lines one anchored edit may cover; beyond that the anchors are
# ignored and the text-location path is used instead.
EDIT_MAX_LINES = int(os.getenv("EDIT_MAX_LINES", "200"))


class SelectionNotFoundError(ValueError):
    """The selected text could not be found in the notes."""


class SelectionAmbiguousError(ValueError):
    """The selected text occurs more than once in the notes."""


class EmptyEditError(RuntimeError):
    """The model returned no replacement text."""


def _locate_selection(notes: str, selection: str):
    """Return the (start, end) span of `selection` in `notes`.

    Exact match first; otherwise a whitespace-normalised match (the UI's
    selection text comes from rendered markdown, where line breaks and
    indentation collapse) mapped back to the original span. Exactly one
    match is required - splicing into the wrong occurrence would silently
    edit text the user never selected.
    """
    first = notes.find(selection) if selection else -1
    if first != -1:
        if notes.find(selection, first + 1) != -1:
            raise SelectionAmbiguousError()
        return first, first + len(selection)

    target = " ".join(selection.split())
    if not target:
        raise SelectionNotFoundError()
    # Collapse each whitespace run in the notes to one space, remembering
    # where every kept character came from.
    norm, idx = [], []
    for m in re.finditer(r"\s+|\S+", notes):
        if m.group(0)[0].isspace():
            norm.append(" ")
            idx.append(m.start())
        else:
            norm.append(m.group(0))
            idx.extend(range(m.start(), m.end()))
    flat = "".join(norm)
    hits, pos = [], flat.find(target)
    while pos != -1 and len(hits) < 2:
        hits.append(pos)
        pos = flat.find(target, pos + 1)
    if not hits:
        raise SelectionNotFoundError()
    if len(hits) > 1:
        raise SelectionAmbiguousError()
    s = hits[0]
    e = s + len(target) - 1
    return idx[s], idx[e] + 1


def _clean_replacement(text: str, keep_indent: bool = False) -> str:
    text = (text or "").rstrip()
    core = text.strip()
    # Models sometimes echo the prompt's triple-quote delimiters.
    if len(core) >= 6 and core.startswith('"""') and core.endswith('"""'):
        text = core[3:-3].rstrip()
    if not keep_indent:
        return text.strip()
    # Drop leading blank lines but keep the first line's indentation (a
    # nested bullet must stay nested).
    return re.sub(r"^(?:[ \t]*\n)+", "", text)


def _anchored_span(lines, line_start, line_end):
    """Validated, blank-trimmed (start, end) line indices, or None."""
    if line_start is None or line_end is None:
        return None
    if not (0 <= line_start <= line_end < len(lines)):
        return None
    if line_end - line_start + 1 > EDIT_MAX_LINES:
        return None
    while line_start < line_end and not lines[line_start].strip():
        line_start += 1
    while line_end > line_start and not lines[line_end].strip():
        line_end -= 1
    if not lines[line_start].strip():
        return None
    return line_start, line_end


def _edit_lines(lines, start, end, selection, instruction, model):
    # Anchored edit: the UI told us which SOURCE lines the highlight covers.
    # Needed because the highlight is rendered text - bold markers, bullets
    # and [n] citations (shown as page chips) are gone from it, so it rarely
    # matches the markdown verbatim. The model edits the raw lines, markdown
    # and citations intact, and the server splices them back by line index.
    passage = "\n".join(lines[start:end + 1])
    before = "\n".join(lines[:start])[-EDIT_CONTEXT_CHARS:]
    after = "\n".join(lines[end + 1:])[:EDIT_CONTEXT_CHARS]
    prompt = f"""You are editing study notes. Apply the INSTRUCTION to the
SOURCE LINES below. The reader highlighted this rendered text within them:
"{selection[:2000]}"
Change only what the instruction asks; keep the rest of these lines, their
markdown formatting and ALL [n] citations. The text before and after is
context so your edit fits in; do not repeat or change it. Return ONLY the
replacement for the source lines - no commentary, no quotes.

INSTRUCTION: {instruction or "improve this passage"}

CONTEXT BEFORE:
\"\"\"{before}\"\"\"

SOURCE LINES:
\"\"\"{passage}\"\"\"

CONTEXT AFTER:
\"\"\"{after}\"\"\""""
    budget = min(8000, max(600, len(passage) * 2 // 3))
    replacement = _clean_replacement(
        call_model(prompt, max_tokens=budget, model=model, temperature=0.4, strict=True),
        keep_indent=True)
    if not replacement.strip():
        raise EmptyEditError()
    return "\n".join(lines[:start] + [replacement] + lines[end + 1:])


def edit_selection(notes, selection, instruction, model=None,
                   line_start=None, line_end=None) -> str:
    # Edit only the selection, then splice (the search/replace pattern code
    # editors use instead of whole-file rewrites): the model returns just the
    # new passage, so text outside the selection cannot be dropped or drift,
    # and the output budget no longer has to fit the whole document.
    # Source-line anchors from the UI are preferred; locating the rendered
    # text in the markdown is the fallback when they are missing or invalid.
    lines = notes.split("\n")
    span = _anchored_span(lines, line_start, line_end)
    if span is not None:
        return _edit_lines(lines, span[0], span[1], selection, instruction, model)

    start, end = _locate_selection(notes, selection)
    # Leave the selection's own leading/trailing whitespace in place.
    while start < end and notes[start].isspace():
        start += 1
    while end > start and notes[end - 1].isspace():
        end -= 1
    passage = notes[start:end]
    before = notes[max(0, start - EDIT_CONTEXT_CHARS):start]
    after = notes[end:end + EDIT_CONTEXT_CHARS]

    prompt = f"""You are editing study notes. Apply the INSTRUCTION to the
SELECTED PASSAGE only. The text before and after it is context so your edit
fits in; do not repeat or change it. Preserve the passage's markdown formatting
and keep any [n] citations it relies on. Return ONLY the replacement text for
the selected passage — no commentary, no quotes.

INSTRUCTION: {instruction or "improve this passage"}

CONTEXT BEFORE:
\"\"\"{before}\"\"\"

SELECTED PASSAGE:
\"\"\"{passage}\"\"\"

CONTEXT AFTER:
\"\"\"{after}\"\"\""""
    # Sized to the selection (~3 chars/token, room to roughly double it).
    budget = min(8000, max(600, len(passage) * 2 // 3))
    replacement = _clean_replacement(
        call_model(prompt, max_tokens=budget, model=model, temperature=0.4, strict=True))
    if not replacement:
        raise EmptyEditError()
    return notes[:start] + replacement + notes[end:]


# ---------------------------------------------------------------------------
# Agent: Title (auto-name a session)
# ---------------------------------------------------------------------------

def generate_title(notes, model=None) -> str:
    prompt = f"""Give a short, specific title (3-6 words, Title Case) that names the
topic of these study notes. Respond with ONLY the title — no quotes, no punctuation
at the end.

NOTES:
\"\"\"{notes[:1500]}\"\"\""""
    title = call_model(prompt, max_tokens=30, model=model, temperature=0.1).strip()
    # tidy: single line, strip surrounding quotes
    title = title.splitlines()[0].strip().strip('"').strip("'") if title else ""
    return title[:60]


# ---------------------------------------------------------------------------
# Agent: Quiz
# ---------------------------------------------------------------------------

def _flat(text) -> str:
    """Collapse a value to one clean line (the plain-text wire format is
    line-oriented, so embedded newlines would corrupt parsing)."""
    return " ".join(str(text or "").split())


def _valid_questions(data, n: int) -> list:
    """Validate model-returned quiz JSON. Returns [] if unusable, else a list
    of fully-formed questions (question text, options A-D, valid answer)."""
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list):
        return []
    out = []
    for q in qs:
        if not isinstance(q, dict):
            continue
        text = _flat(q.get("question"))
        opts = q.get("options")
        if isinstance(opts, list) and len(opts) >= 4:
            opts = {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]}
        if not isinstance(opts, dict):
            continue
        norm = {str(k).strip().upper()[:1]: _flat(v) for k, v in opts.items()}
        if not all(norm.get(letter) for letter in "ABCD"):
            continue
        answer = str(q.get("answer") or "").strip().upper()[:1]
        # `not answer` first: "" is a substring of every string, so
        # `"" not in "ABCD"` is False and an answerless question got through.
        if not answer or answer not in "ABCD" or not text:
            continue
        out.append({
            "question": text,
            "options": {letter: norm[letter] for letter in "ABCD"},
            "answer": answer,
            "explanation": _flat(q.get("explanation")),
        })
    return out[:n]


def _render_quiz(questions: list) -> str:
    """Deterministically render validated questions into the plain-text format
    the frontend/exports parse (Q#) / A)-D) / Answer: / Explanation:)."""
    blocks = []
    for i, q in enumerate(questions, 1):
        lines = [f"Q{i}) {q['question']}"]
        lines += [f"{letter}) {q['options'][letter]}" for letter in "ABCD"]
        lines.append(f"Answer: {q['answer']}")
        if q.get("explanation"):
            lines.append(f"Explanation: {q['explanation']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# The quiz writer and the answer-key checker must read the SAME notes. The
# checker used to get a 9000-char excerpt of notes the writer saw at 12000, so
# it could "correct" a right answer from a partial view (the same bug class as
# the critique/judge truncation fixes). Both now use this one constant.
QUIZ_NOTES_CHARS = 12000

_QUIZ_JSON_SCHEMA = """{"questions": [{"question": "<question text>",
"options": {"A": "<option>", "B": "<option>", "C": "<option>", "D": "<option>"},
"answer": "<A|B|C|D>", "explanation": "<one-sentence explanation>"}]}"""


def generate_quiz(notes, n=5, model=None) -> str:
    """Generate a quiz via structured JSON (validated, then rendered to the
    plain-text wire format deterministically). Models drift from free-form
    text formats constantly; JSON mode + validation makes output reliable.
    Falls back to the legacy plain-text prompt if JSON parsing fails."""
    prompt = f"""You are the QUIZ agent. Create exactly {n} multiple-choice
questions that test understanding of the NOTES.

Return ONLY a JSON object in exactly this shape (no markdown, no commentary):
{_QUIZ_JSON_SCHEMA}

Every question must have exactly four options A-D and one correct answer letter.

NOTES:
\"\"\"{_notes_excerpt(notes, QUIZ_NOTES_CHARS)}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=1600, model=model, temperature=0.3, json_mode=True))
        questions = _valid_questions(data, n)
        if questions:
            return _render_quiz(questions)
    except Exception:  # noqa: BLE001
        pass
    return _generate_quiz_text(notes, n=n, model=model)


def _generate_quiz_text(notes, n=5, model=None) -> str:
    """Legacy plain-text fallback (kept so a JSON hiccup can't break quizzes)."""
    prompt = f"""You are the QUIZ agent. Create exactly {n} multiple-choice
questions that test understanding of the NOTES.

Use EXACTLY this plain-text format for each question (no markdown, no extra text):

Q1) <question text>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <A|B|C|D>
Explanation: <one-sentence explanation>

Leave a blank line between questions. Number them Q1, Q2, ... up to Q{n}.

NOTES:
\"\"\"{_notes_excerpt(notes, QUIZ_NOTES_CHARS)}\"\"\""""

    return call_model(prompt, max_tokens=1200, model=model, temperature=0.3)


# A correction must carry a quote from the notes that Python can find there.
# The verifier's verdict alone is not enough: a wrong verifier used to silently
# overwrite a right answer. Quotes shorter than this can't pin anything down.
_EVIDENCE_MIN_WORDS = 5


def _norm_evidence(text) -> str:
    """Normalise notes/quote text for the evidence check: lowercase, drop
    `**` bold markers and `[n]` citation markers (with the space before them),
    collapse whitespace. Applied to BOTH sides, so a plain-text quote matches
    notes that render the same sentence with markdown and citations."""
    t = str(text or "").lower().replace("**", "")
    t = re.sub(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]", "", t)
    return " ".join(t.split())


def _evidence_supported(evidence, notes_view: str) -> bool:
    quote = _norm_evidence(evidence).strip(" \"'“”‘’")
    quote = quote.strip(" .…").strip()
    if len(quote.split()) < _EVIDENCE_MIN_WORDS:
        return False
    return quote in _norm_evidence(notes_view)


def _unchecked_report(questions: int = 0) -> dict:
    return {"checked": False, "questions": questions, "judged": 0,
            "corrected": 0, "rejected": 0, "disputed": []}


def verify_quiz_detailed(notes, quiz, model=None):
    """
    Check the quiz answer key against the notes and say what was checked.

    Returns (quiz_text, report). The verifier gives a verdict for EVERY
    question; a correction is applied only when its evidence quote is found
    (after normalisation) in the exact notes excerpt the generator wrote from.
    Unsupported corrections are rejected: the original answer stays and the
    question is reported as disputed.

    report = {"checked", "questions", "judged", "corrected", "rejected",
    "disputed"}. `checked` is False when the call failed or returned nothing
    usable — previously that looked identical to "no corrections needed".
    judged < questions means the key was only partly checked. Any failure
    returns the original quiz unchanged.
    """
    parsed = _parse_quiz_text(quiz) if quiz and quiz.strip() else []
    report = _unchecked_report(len(parsed))
    if not parsed:
        return quiz, report

    notes_view = _notes_excerpt(notes, QUIZ_NOTES_CHARS)
    prompt = f"""You are a QUIZ VERIFIER. For EVERY question in the QUIZ, check whether
the marked answer letter is actually correct according to the NOTES.

Return ONLY a JSON object with one verdict per question, in exactly this shape:
{{"verdicts": [{{"q": <question number>, "correct": <true|false>,
"answer": "<the correct letter A|B|C|D>",
"evidence": "<exact sentence copied from the NOTES that supports the correct answer>",
"explanation": "<one-sentence explanation>"}}]}}

The evidence must be copied word for word from the NOTES, not paraphrased.

NOTES:
\"\"\"{notes_view}\"\"\"

QUIZ:
\"\"\"{quiz}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=1400, model=model, temperature=0.0, json_mode=True))
    except Exception:  # noqa: BLE001
        return quiz, report

    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list):
        return quiz, report

    seen, changed = set(), False
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        try:
            idx = int(v.get("q")) - 1
        except (TypeError, ValueError):
            continue
        correct = v.get("correct")
        if isinstance(correct, str):
            correct = {"true": True, "false": False}.get(correct.strip().lower())
        if not (0 <= idx < len(parsed)) or idx in seen or not isinstance(correct, bool):
            continue
        seen.add(idx)
        if correct:
            continue
        answer = str(v.get("answer") or "").strip().upper()[:1]
        if answer == parsed[idx]["answer"]:
            continue  # self-contradictory verdict: nothing to change
        if answer and answer in "ABCD" and _evidence_supported(v.get("evidence"), notes_view):
            parsed[idx]["answer"] = answer
            # The old explanation argued for the wrong answer; replace it.
            expl = _flat(v.get("explanation")) or f'From your notes: "{_flat(v.get("evidence"))}"'
            parsed[idx]["explanation"] = expl
            report["corrected"] += 1
            changed = True
        else:
            report["rejected"] += 1
            report["disputed"].append(idx + 1)

    report["judged"] = len(seen)
    report["checked"] = report["judged"] > 0
    report["disputed"].sort()
    return (_render_quiz(parsed) if changed else quiz), report


def verify_quiz(notes, quiz, model=None) -> str:
    """Answer-key check returning only the quiz text (run_agent and the eval
    harness use this). See verify_quiz_detailed for the report."""
    return verify_quiz_detailed(notes, quiz, model=model)[0]


def _parse_quiz_text(quiz: str) -> list:
    """Parse the plain-text quiz format back into structured questions
    (mirror of the frontend parser, used to apply verifier corrections)."""
    questions, current = [], None
    for raw_line in (quiz or "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        q_match = re.match(r"^Q?\s*\d+[).:]\s*(.+)$", line, re.IGNORECASE)
        opt_match = re.match(r"^([A-D])[).:]\s*(.+)$", line)
        ans_match = re.match(r"^Answer\s*:?\s*([A-D])", line, re.IGNORECASE)
        exp_match = re.match(r"^Explanation\s*:?\s*(.+)$", line, re.IGNORECASE)
        if q_match and not opt_match:
            if current and current.get("question") and len(current.get("options", {})) == 4:
                questions.append(current)
            current = {"question": q_match.group(1).strip(), "options": {},
                       "answer": "", "explanation": ""}
        elif opt_match and current is not None:
            current["options"][opt_match.group(1).upper()] = opt_match.group(2).strip()
        elif ans_match and current is not None:
            current["answer"] = ans_match.group(1).upper()
        elif exp_match and current is not None:
            current["explanation"] = exp_match.group(1).strip()
    if current and current.get("question") and len(current.get("options", {})) == 4:
        questions.append(current)
    return questions


# ---------------------------------------------------------------------------
# Agent: Flashcards
# ---------------------------------------------------------------------------

def _valid_cards(data, n: int) -> list:
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, list):
        return []
    out = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        front, back = _flat(card.get("front")), _flat(card.get("back"))
        if front and back:
            out.append({"front": front, "back": back})
    return out[:n]


def _render_flashcards(cards: list) -> str:
    blocks = []
    for i, card in enumerate(cards, 1):
        blocks.append(f"CARD {i}\nFront: {card['front']}\nBack: {card['back']}")
    return "\n\n".join(blocks)


def generate_flashcards(notes, n=8, model=None) -> str:
    """Structured-JSON flashcards (validated + deterministically rendered),
    with the legacy plain-text prompt as fallback."""
    prompt = f"""You are the FLASHCARD agent. Create exactly {n} flashcards
from the NOTES.

Return ONLY a JSON object in exactly this shape (no markdown, no commentary):
{{"cards": [{{"front": "<concise question or term>", "back": "<clear, correct answer>"}}]}}

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=1400, model=model, json_mode=True))
        cards = _valid_cards(data, n)
        if cards:
            return _render_flashcards(cards)
    except Exception:  # noqa: BLE001
        pass
    return _generate_flashcards_text(notes, n=n, model=model)


def _generate_flashcards_text(notes, n=8, model=None) -> str:
    """Legacy plain-text fallback."""
    prompt = f"""You are the FLASHCARD agent. Create exactly {n} flashcards
from the NOTES.

Use EXACTLY this plain-text format for each card (no markdown, no extra text):

CARD 1
Front: <concise question or term>
Back: <clear, correct answer>

Leave a blank line between cards. Number them CARD 1 ... CARD {n}.

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    return call_model(prompt, max_tokens=1200, model=model)


# ---------------------------------------------------------------------------
# Chat: ask questions grounded in the notes
# ---------------------------------------------------------------------------

# How many prior chat turns go into the prompt (the most recent ones).
CHAT_HISTORY_TURNS = 6


def chat_about_notes_stream(notes, question, history=None, model=None, cancel=None):
    """Stream a tutor-style answer grounded in the provided notes."""
    history = history or []
    convo = ""
    for turn in history[-CHAT_HISTORY_TURNS:]:
        role = "Student" if turn.get("role") == "user" else "Tutor"
        convo += f"{role}: {(turn.get('content') or '')[:2000]}\n"

    prompt = f"""You are a helpful study TUTOR. Answer the student's question using
primarily the NOTES below as context. If the notes don't cover it, you may use
general knowledge but say so briefly. Be clear and concise. You may use `$...$`
for math and fenced code blocks.

NOTES:
\"\"\"{_notes_excerpt(notes, 10000)}\"\"\"

Conversation so far:
{convo}
Student: {question}
Tutor:"""

    yield from call_model_stream(prompt, max_tokens=900, model=model, **_cancel_kw(cancel))


# ---------------------------------------------------------------------------
# Gatekeeper: accept only academic / study material
# ---------------------------------------------------------------------------

def classify_academic(text, model=None) -> dict:
    prompt = f"""You are a strict gatekeeper for an ACADEMIC study-notes generator.
Decide whether the SOURCE is genuine academic / study material — a school,
college, or exam subject such as the sciences, mathematics, computer science,
engineering, medicine, the humanities, history, social sciences, economics, law,
or languages.

REJECT material that is primarily: celebrity or entertainment trivia, gossip,
sports results, product marketing/advertising, personal or casual content, or
anything not intended for serious study.

Also report what KIND of document it is, judged from the SOURCE itself:
- "explanatory": teaches — definitions, explanations, worked examples (lecture
  notes, textbook chapter, slides, article)
- "question_bank": a list of questions/problems to solve
- "assignment": tasks to complete and submit
- "exam": an exam or test paper
- "worksheet": practice exercises
- "syllabus": a course/topic outline with no teaching
- "mixed": substantial explanation AND substantial tasks/questions
- "other": none of the above

A document dominated by imperatives — "Write a program to…", "Define…",
"Answer the following…", numbered tasks — is task-shaped, NOT explanatory,
even when the subject matter is academic.

Respond with ONLY a JSON object:
{{"academic": true|false, "subject": "<subject or 'n/a'>",
  "doc_type": "<one of the types above>", "reason": "<one short sentence>"}}

SOURCE:
\"\"\"{text[:4000]}\"\"\""""

    data = safe_json(call_model(prompt, max_tokens=200, model=model, temperature=0.0, json_mode=True))

    # Permissive on parse failure — don't block legitimate content over a glitch.
    # "explanatory" is the safe default for doc_type for the same reason: it is
    # the behaviour that shipped before, so a classifier glitch cannot silently
    # downgrade an ordinary lecture PDF.
    if not isinstance(data, dict) or "academic" not in data:
        return {"academic": True, "subject": "n/a", "doc_type": "explanatory", "reason": ""}
    doc_type = str(data.get("doc_type", "explanatory") or "explanatory").strip().lower()
    if doc_type not in KNOWN_DOC_TYPES:
        doc_type = "explanatory"
    return {
        "academic": bool(data.get("academic", True)),
        "subject": data.get("subject", "n/a") or "n/a",
        "doc_type": doc_type,
        "reason": data.get("reason", "") or "",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _emit(type_, step, content="", data=None):
    return {"type": type_, "step": step, "content": content, "data": data}


def _page_tagger(retriever, spans):
    """Return a function that stamps each chunk with its source page(s).

    Chunks cut against page spans already CARRY their page — it is the page
    they were cut from, not a guess — so the common path just passes it
    through.

    The offset fallback below only runs for chunks with no page (spans absent
    or stale). It reports the true RANGE from both offsets rather than the
    start alone: a chunk that straddles pages used to be labelled with the
    first of them, which is exactly how citations ended up 1-3 pages early.
    """
    if not spans:
        return lambda chunks: chunks

    meta = {c["chunk_id"]: c for c in retriever.chunks_meta}

    def tag(chunks):
        out = []
        for chunk in chunks:
            if chunk.get("page"):
                out.append(chunk)
                continue
            m = meta.get(chunk["id"], {})
            first = page_for_offset(spans, m.get("start_offset", 0))
            last = page_for_offset(spans, max(m.get("start_offset", 0), m.get("end_offset", 1) - 1))
            if not first:
                out.append(chunk)
                continue
            pages = list(range(first, (last or first) + 1))
            out.append({**chunk, "page": first, "pages": pages})
        return out

    return tag


class _RunTimings:
    """Server-side stage timings for one generation, logged as JSON lines.

    Each stage logs start/end (epoch ms), duration_ms, the model calls made
    while it was open (every provider attempt, failovers included), the
    provider:model pairs that answered, and stage-specific counts such as
    concurrency. Only names and numbers - never text, titles, prompts or ids
    of the user. Doubles as the models.call_stats_scope sink: calls from any
    thread carrying this run's scope land in the stage currently open.
    """

    def __init__(self, gen_id=None):
        self.gen_id = gen_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._current = None
        self.started = time.time()
        self.model_calls = 0

    def record(self, provider, model_id, ok):
        with self._lock:
            self.model_calls += 1
            stage = self._current
            if stage is None:
                return
            stage["model_calls"] += 1
            if ok:
                stage["providers"].add(f"{provider}:{model_id}")

    def begin(self, name, **extra):
        """Open a stage (stages are sequential; an open one is closed first)."""
        self.end()
        with self._lock:
            self._current = {"stage": name, "start": time.time(), "model_calls": 0,
                             "providers": set(), **extra}

    def end(self, outcome="ok", **extra):
        with self._lock:
            stage, self._current = self._current, None
        if stage is not None:
            stage.update(extra)
            self._log(stage.pop("start"), time.time(), outcome=outcome, **stage)

    def log_stage(self, name, start, end, **extra):
        """Log a stage timed elsewhere (e.g. on a background thread)."""
        self._log(start, end, stage=name, model_calls=0, providers=(), outcome="ok", **extra)

    def _log(self, start, end, **fields):
        providers = sorted(fields.pop("providers", ()) or ())
        payload = {"gen_id": self.gen_id, "stage": fields.pop("stage"),
                   "start_ms": int(start * 1000), "end_ms": int(end * 1000),
                   "duration_ms": int((end - start) * 1000),
                   "model_calls": fields.pop("model_calls", 0),
                   "providers": providers, **fields}
        _logger.info("[timing] %s", json.dumps(payload))

    def finish(self, outcome):
        # A stage still open here was interrupted by the run's ending.
        self.end(outcome={"done": "ok", "blocked": "ok", "cancelled": "cancelled",
                          "closed": "cancelled"}.get(outcome, "error"))
        self._log(self.started, time.time(), stage="total", model_calls=self.model_calls,
                  providers=(), outcome=outcome)


def _build_index(text, page_spans, timings):
    """Build the retrieval index (runs on a background thread, see below)."""
    start = time.time()
    retriever = Retriever(text, spans=page_spans)
    timings.log_stage("index_build", start, time.time(),
                      chunks=len(getattr(retriever, "chunks_meta", []) or []))
    return retriever


def run_agent(text, mode, tone, length, fmt, model=None, instructions="",
              include_quiz=True, include_flashcards=True, page_spans=None, cancel=None,
              generation_id=None):
    """
    Drive the full pipeline, yielding event dicts as each stage progresses.

    `cancel` is an optional threading.Event. Once it is set the pipeline stops
    at the next stage boundary (before forwarding another event), the
    streaming model calls stop at their next delta, the non-streaming ones are
    not started, and queued section writers never start. A cancelled run
    yields nothing more - no `error`, no `done` - is logged at INFO, and is
    never recorded as lost coverage or a cut-off stream. Closing this
    generator closes the pipeline (and waits for its section writers when
    cancelled).

    Event types and ordering are documented on _run_agent_events.

    Every run logs "[timing]" JSON lines per stage plus a "total", all sharing
    one gen_id (`generation_id`, or a random one) so lines from concurrent
    runs can be told apart.
    """
    timings = _RunTimings(generation_id)
    inner = _run_agent_events(
        text, mode, tone, length, fmt, model=model, instructions=instructions,
        include_quiz=include_quiz, include_flashcards=include_flashcards,
        page_spans=page_spans, cancel=cancel, timings=timings,
    )
    outcome = "closed"
    try:
        while True:
            if cancel is not None and cancel.is_set():
                _logger.info("[pipeline] cancelled; stopped at a stage boundary")
                outcome = "cancelled"
                return
            try:
                # Steps may run on different executor threads, so the call-stats
                # sink is (re)bound per step, exactly like the cancel flag.
                with call_stats_scope(timings):
                    if cancel is None:
                        event = next(inner)
                    else:
                        # Ambient flag for the non-streaming call_model() calls made
                        # during this step - scoped to the step, never a yield.
                        with cancel_scope(cancel):
                            event = next(inner)
            except StopIteration:
                if outcome == "closed":
                    outcome = "ended"
                return
            except PipelineCancelled:
                _logger.info("[pipeline] cancelled mid-stage; stopped")
                outcome = "cancelled"
                return
            if cancel is not None and cancel.is_set():
                _logger.info("[pipeline] cancelled; stopped at a stage boundary")
                outcome = "cancelled"
                return
            if event.get("type") in ("done", "error", "blocked"):
                outcome = event["type"]
            yield event
    finally:
        inner.close()
        timings.finish(outcome)


def _run_agent_events(text, mode, tone, length, fmt, model=None, instructions="",
                      include_quiz=True, include_flashcards=True, page_spans=None,
                      cancel=None, timings=None):
    """
    The pipeline body behind run_agent (which adds cancellation and timing).

    Event types:
      status, plan_done, sources, notes_delta, notes_done, critique_done,
      revise_start, notes_revised, title_done, quiz_done, flashcards_done,
      done, error

    `sources` may be emitted AGAIN during the revise loop when corrective
    re-retrieval adds chunks for missing topics — the payload is always the
    full merged list (a superset of the previous one), so clients can simply
    replace their sources state.
    """
    timings = timings or _RunTimings()
    # Overlap: the retrieval index is built on a background thread WHILE the
    # topic check runs. Dependency-safe: the index reads only `text` and
    # `page_spans`, the topic check reads only `text`, and neither consumes the
    # other's output. The index is still first USED (and any build error still
    # raised) at the same point as before - right after the gate - so the event
    # order and error handling are unchanged. A blocked document just abandons
    # the build (it makes no model calls; at most one embedding batch).
    index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
    index_future = index_pool.submit(_build_index, text, page_spans, timings)
    try:
        # Mechanical packaging agents (gatekeeper, title, quiz, flashcards) run
        # on a cheaper model with its OWN free-tier daily quota, so they don't
        # spend the main model's token budget — writing & critique keep the
        # strong model where quality actually matters.
        helper = helper_model(model)

        # 0. Academic gatekeeper — this tool only handles study material.
        yield _emit("status", "gate", "Checking topic…")
        timings.begin("topic_check")
        gate = classify_academic(text, model=helper)
        timings.end()
        doc_type = gate.get("doc_type", "explanatory")

        # Which provider ACTUALLY produced the notes. Recorded from the writer
        # and revise calls, so a silent failover to Gemini is visible instead of
        # being indistinguishable from an NVIDIA success.
        served = {}

        def _record_serve(provider, served_model, fallback_used, reason):
            served.update({
                "provider": provider,
                "model": served_model,
                "fallback_used": bool(fallback_used),
            })
            if fallback_used and reason:
                served["fallback_reason"] = reason

        if not gate.get("academic", True):
            yield _emit(
                "blocked",
                "blocked",
                gate.get("reason") or "This doesn't look like academic study material.",
                gate,
            )
            return

        if doc_type != "explanatory":
            yield _emit("status", "gate", f"Detected document type: {doc_type.replace('_', ' ')}.")

        # The retrieval index over the source (RAG), started above.
        # Spans make chunking page-bounded: a chunk is cut from inside one
        # page, so its page is a fact rather than an offset lookup.
        retriever = index_future.result()

        # Trace each retrieved chunk back to the page it came from, so a
        # citation can point at the document rather than at an opaque passage
        # number. A no-op for sources with no pages (pasted text, URLs,
        # transcripts) — they keep citing exactly as before.
        _with_pages = _page_tagger(retriever, page_spans)

        # 1-2. Plan. For very large documents the even sample alone covers too
        # little (a topic on only a few pages can be invisible in it), so first
        # the helper model scans the WHOLE document and builds a topic
        # inventory the planner must cover — 100% coverage at planning time.
        topic_inventory = ""
        if len(text) > DIGEST_DOC_THRESHOLD:
            n_parts = min(
                (len(text) + DIGEST_SEGMENT_CHARS - 1) // DIGEST_SEGMENT_CHARS,
                DIGEST_MAX_SEGMENTS,
            )
            yield _emit("status", "plan", f"Scanning full document ({n_parts} parts)…")
            timings.begin("digest", concurrency=DIGEST_CONCURRENCY, segments=n_parts)
            topic_inventory = digest_document(text, model=helper)
            timings.end()

        yield _emit("status", "plan", "Planning outline...")
        timings.begin("plan")
        plan = plan_outline(
            retriever.sample(24000), mode, tone, length,
            model=model, instructions=instructions, doc_chars=len(text),
            topic_inventory=topic_inventory,
        )
        timings.end()
        yield _emit("plan_done", "plan", "", plan)

        # Coverage starts complete and is narrowed only by real failures, so
        # every path - single-pass included - carries a definite answer.
        _total_pages = len(page_spans or [])
        coverage = {"complete": True, "total_pages": _total_pages,
                    "processed_pages": _total_pages, "failed_pages": [],
                    "failed_windows": []}

        active_fmt = fmt or plan.get("suggested_format", "bullet")
        outline = plan.get("outline", []) or ["Overview", "Key Concepts", "Summary"]
        checklist = plan.get("checklist", []) or []

        # Long documents: MAP-REDUCE. Each outline section gets its OWN
        # retrieval across the whole document and is written from its own
        # context — so no part of a large source is left out.
        # Size alone decides. This used to also require len(outline) >= 3,
        # which meant a terse plan could drop a 100-page document into the
        # single-pass path and lose most of it. Windows no longer come from
        # the outline, so the outline no longer gates coverage.
        sectioned = len(text) > SECTION_DOC_THRESHOLD

        if sectioned:
            timings.begin("write", mode="sectioned", concurrency=max(1, SECTION_CONCURRENCY))
            # COVERAGE FIRST. Windows partition the document, so every chunk
            # is written about exactly once; retrieval then ADDS related
            # evidence from elsewhere. Previously each outline topic ran its own
            # retrieval and the union was the writer's entire view of the
            # source — capped at SECTION_MAX_COUNT x SECTION_RETRIEVAL_K chunks
            # however large the document was (9% of a 100-page source, 4% of a
            # 200-page one), and a passage no query ranked highly was simply
            # never written about.
            windows = _document_windows(retriever.chunks_meta)
            section_ctx = []
            chunk_map = {}
            for title, win_chunks in windows:
                ctx_chunks = _window_context(win_chunks, retriever, mode)
                # The window's OWN pages - not the supplements, which belong to
                # other windows - are what this window is responsible for, and
                # therefore what is lost if it fails.
                own_pages = sorted({c["page"] for c in win_chunks if c.get("page")})
                section_ctx.append((title, ctx_chunks, own_pages))
                for c in ctx_chunks:
                    chunk_map[c["id"]] = c
            all_chunks = [chunk_map[i] for i in sorted(chunk_map)]
            yield _emit(
                "status", "write",
                f"Covering the whole document in {len(windows)} part(s)…",
            )
            context = _format_context(all_chunks)
            yield _emit("sources", "write", "", _with_pages(all_chunks))

            # 3-4. Write sections CONCURRENTLY (each has independent context),
            # but stream them to the client strictly IN ORDER: every section's
            # deltas go into its own queue; the main generator drains queue 1
            # live while later sections are already being written in the
            # background. Event sequence is identical to the sequential path.
            parts = []

            def _push(s):
                parts.append(s)
                return _emit("notes_delta", "write", s)

            def _write_worker(sec, ctx, out_q):
                # A window that produces nothing is retried once. call_model_stream
                # already fails over across providers, so reaching here means the
                # whole chain came back empty or errored - usually transient (a 503,
                # a truncated response). Observed live: two of six windows lost that
                # way in one run, and a lost window is lost coverage, the one thing
                # this milestone exists to prevent.
                #
                # Only a window that emitted NOTHING is retried, so a retry can
                # never duplicate text the client has already been streamed.
                #
                # Cancellation (PipelineCancelled, a BaseException) is not caught
                # by `except Exception`, so it is never retried or reported as a
                # failed window; the `finally` still terminates the queue so the
                # consumer can never hang on it.
                err = None
                try:
                    for attempt in (1, 2):
                        raise_if_cancelled(cancel)
                        sent = False
                        try:
                            for delta in write_section_stream(
                                sec, ctx, mode, tone, length, active_fmt,
                                checklist=checklist, model=model, instructions=instructions,
                                doc_type=doc_type, on_serve=_record_serve, is_part=True,
                                **_cancel_kw(cancel),
                            ):
                                raise_if_cancelled(cancel)
                                sent = True
                                out_q.put(("delta", delta))
                        except Exception as exc:  # noqa: BLE001
                            err = exc
                            if sent:
                                break  # partial output already streamed - do not redo it
                            continue
                        if sent:
                            err = None
                            break
                        err = err or _WindowProducedNothing(
                            f"no provider produced output for {sec}")
                except PipelineCancelled:
                    err = None
                finally:
                    if err is not None:
                        out_q.put(("error", err))
                    out_q.put(("end", None))

            queues = [_queue.Queue() for _ in section_ctx]
            pool = ThreadPoolExecutor(max_workers=max(1, SECTION_CONCURRENCY),
                                      thread_name_prefix="section")
            # One window is no longer one run. With up to COVERAGE_MAX_WINDOWS
            # generation calls the chance that at least one hits a provider
            # error is that many times higher, and raising would discard every
            # other window's finished work. A failed window becomes a reported
            # gap instead: losing one part beats losing all of them, and
            # staying quiet about it would be worse than either.
            failed = []
            try:
                # _carry_scopes: the writers' model calls count toward this
                # run's timing record (the cancel flag is also passed explicitly).
                for (sec, sec_chunks, _pp), out_q in zip(section_ctx, queues):
                    pool.submit(_carry_scopes(_write_worker), sec,
                                _format_context(sec_chunks), out_q)

                for i, ((sec, _sec_chunks, sec_pages), out_q) in enumerate(
                        zip(section_ctx, queues), 1):
                    yield _emit(
                        "status", "write",
                        f"Writing section {i}/{len(section_ctx)}: {sec}…",
                    )
                    # No mechanical page-range heading: the writer supplies its
                    # own topical headings, the status line above already names
                    # the pages, and the citations carry page provenance.
                    mark = len(parts)
                    err = None
                    while True:
                        kind, payload = out_q.get()
                        if kind == "delta":
                            yield _push(payload)
                        elif kind == "error":
                            err = payload
                        else:  # "end"
                            break
                    # A cancelled writer ends its queue early; that is not a
                    # failed window and must never be recorded as lost coverage.
                    raise_if_cancelled(cancel)
                    if err is not None:
                        failed.append({"window": sec, "pages": sec_pages,
                                       "reason": _failure_slug(err)})
                        print(f"[coverage] section {sec!r} failed to write: "
                              f"{type(err).__name__}")
                        if len(parts) == mark:
                            # Nothing was written for this part at all.
                            continue
                    yield _push("\n\n")
            finally:
                # Sections that haven't started are always dropped. When the run
                # was CANCELLED, also wait for the running writers - they stop at
                # their next delta - so that once this generator is closed no
                # writer thread is still spending tokens (the caller releases
                # the generation slot only after that). Uncancelled early exits
                # keep the old non-blocking behaviour.
                pool.shutdown(wait=cancel is not None and cancel.is_set(),
                              cancel_futures=True)

            # Durable coverage record. Until now a lost window survived only
            # as a transient status message, so once the stream ended a run
            # missing four pages was byte-for-byte indistinguishable from a
            # complete one in history, export and share.
            failed_titles = {f["window"] for f in failed}
            done_pages, lost_pages = set(), set()
            for _t, _c, pp in section_ctx:
                (lost_pages if _t in failed_titles else done_pages).update(pp)
            # A page straddling a window boundary can belong to two windows; if
            # either wrote it, it is not lost.
            lost_pages -= done_pages
            coverage = {
                "complete": not failed,
                "total_pages": coverage["total_pages"],
                "processed_pages": len(done_pages),
                "failed_pages": sorted(lost_pages),
                "failed_windows": failed,
            }

            if failed and len(failed) == len(section_ctx):
                raise UserFacingError(
                    "Every part of the document failed to generate. That is "
                    "usually a transient provider problem rather than an issue "
                    "with your source — please try again."
                )
            if failed:
                yield _emit(
                    "status", "write",
                    f"{len(failed)} of {len(section_ctx)} part(s) could not be "
                    f"completed ({', '.join(f['window'] for f in failed)}); "
                    f"the rest of the notes are complete.",
                )

            notes = "".join(parts).strip()
            if not notes:
                raise UserFacingError(
                    "The model returned an empty draft. That is usually a transient "
                    "provider hiccup rather than a problem with your source — "
                    "please try again."
                )
            timings.end(windows=len(section_ctx), failed_windows=len(failed))
            yield _emit("notes_done", "write", notes)
        else:
            timings.begin("write", mode="single_pass")
            # Small sources: single-pass write over one retrieval (fast path).
            # Pull a length-scaled slice so "long" actually has enough source
            # material to hit its word target.
            query = " ".join(outline + checklist + [mode])
            single_k = SINGLE_PASS_K.get((length or "medium").lower(), 14)
            # Document order for the writer: retrieve() returns relevance
            # order (its ordering is a measured property), but evidence reads
            # better in the order the document tells it — and the sectioned
            # path above already sorts its union the same way.
            chunks = sorted(retriever.retrieve(query, k=single_k), key=lambda c: c["id"])
            chunk_map = {c["id"]: c for c in chunks}
            context = _format_context(chunks)
            yield _emit("sources", "write", "", _with_pages(chunks))

            yield _emit("status", "write", "Writing notes...")
            parts = []
            try:
                for delta in write_notes_stream(
                    context, mode, tone, length, active_fmt, plan, model=model,
                    instructions=instructions, doc_type=doc_type, on_serve=_record_serve,
                    **_cancel_kw(cancel),
                ):
                    parts.append(delta)
                    yield _emit("notes_delta", "write", delta)
            except IncompleteStreamError as exc:
                # The client already has the partial draft, and a partial draft
                # beats none - but it must be recorded as incomplete, exactly
                # like a failed window, so the banner and coverage record say so.
                if not "".join(parts).strip():
                    raise
                coverage["complete"] = False
                coverage["failed_windows"] = [
                    {"window": "draft", "pages": [], "reason": exc.reason}]
                yield _emit("status", "write",
                            "The draft was cut off before it finished; "
                            "keeping what was written.")
            notes = "".join(parts).strip()
            if not notes:
                raise UserFacingError(
                    "The model returned an empty draft. That is usually a transient "
                    "provider hiccup rather than a problem with your source — "
                    "please try again."
                )
            timings.end()
            yield _emit("notes_done", "write", notes)

        # Critique: FAITHFULNESS is judged against the CONTEXT the writer was
        # actually given (previously it was judged against an unrelated even
        # sample, which produced false "unsupported claim" flags). COVERAGE is
        # still judged against a breadth sample of the whole document. Revise
        # with a bigger budget when the notes are sectioned.
        doc_sample = retriever.sample(8000)
        revise_length = "xl" if sectioned else length

        def _crit_msg(c):
            sc = c.get("score", 0)
            uns = len(c.get("unsupported_claims", []))
            if c.get("needs_revision"):
                extra = f", {uns} unsupported claim(s)" if uns else ""
                return f"Quality {sc}/10{extra} — revising ✍"
            return f"Quality {sc}/10 — faithful, no revision needed ✓"

        # 5-6. Critique (grounded against the writer's CONTEXT for faithfulness,
        # plus a breadth sample for coverage)
        yield _emit("status", "critique", "Checking faithfulness & coverage...")
        timings.begin("critique", round=0)
        critique = critique_notes(
            notes, plan, mode, source=context, model=model, doc_sample=doc_sample,
            chunk_map=chunk_map,
        )
        timings.end()
        yield _emit("critique_done", "critique", _crit_msg(critique), critique)

        best_notes = notes
        best_critique = critique

        # 7. Iterative revise loop: (corrective re-retrieval) → revise →
        # re-critique, keep the best version.
        rounds = 0
        # Safety: a full-rewrite revision must see the WHOLE notes. If they
        # exceed the rewrite cap, revising would silently drop the tail —
        # keep the draft (citation cleanup below still runs).
        if len(notes) > NOTES_REWRITE_CAP and critique.get("needs_revision"):
            critique["needs_revision"] = False
            yield _emit(
                "status", "revise",
                "Notes are too long for a safe full revision — keeping the draft.",
            )
        while critique.get("needs_revision") and rounds < MAX_REVISION_ROUNDS:
            rounds += 1

            # Corrective re-retrieval (CRAG): the reviser can only add missing
            # topics if the context actually CONTAINS them. Query the retriever
            # with each missing topic and merge any new chunks into the context
            # before revising; the frontend gets the merged sources list.
            missing = critique.get("missing_topics") or []
            timings.begin("revise", round=rounds)
            # Ids added this round, so the reviser is shown them even when the
            # full context is too big to send.
            round_added = []
            if missing:
                added = False
                for topic in missing[:CORRECTIVE_TOPICS_MAX]:
                    try:
                        extra = retriever.retrieve(str(topic), k=CORRECTIVE_K_PER_TOPIC)
                    except Exception:  # noqa: BLE001
                        continue
                    for c in extra:
                        if c["id"] not in chunk_map:
                            chunk_map[c["id"]] = c
                            round_added.append(c["id"])
                            added = True
                if added:
                    merged = [chunk_map[i] for i in sorted(chunk_map)]
                    context = _format_context(merged)
                    yield _emit("sources", "revise", "", _with_pages(merged))
                    yield _emit(
                        "status", "revise",
                        f"Retrieved extra context for {min(len(missing), CORRECTIVE_TOPICS_MAX)} missing topic(s)…",
                    )

            yield _emit("status", "revise", f"Revising (round {rounds}/{MAX_REVISION_ROUNDS})...")
            yield _emit("revise_start", "revise", "")
            parts = []
            cut_off = False
            try:
                for delta in revise_notes_stream(
                    notes, critique, mode, plan, active_fmt, model=model,
                    instructions=instructions, context=context, length=revise_length,
                    on_serve=_record_serve, chunk_map=chunk_map, added_ids=round_added,
                    **_cancel_kw(cancel),
                ):
                    parts.append(delta)
                    yield _emit("notes_delta", "revise", delta)
            except Exception as exc:  # noqa: BLE001
                # Includes IncompleteStreamError: a half-finished revision is
                # discarded, never allowed to replace complete notes.
                print(f"[agent] revision round {rounds} failed ({exc}); keeping current notes.")
                cut_off = isinstance(exc, IncompleteStreamError)
                parts = []

            revised = "".join(parts).strip()
            timings.end(outcome="ok" if revised else "discarded")
            # A revision that comes back empty is a FAILED revision, never an
            # instruction to delete the notes. This used to overwrite good
            # notes with "" — the provider intermittently closes the stream
            # without emitting anything, and the user watched their notes stop
            # part-way or vanish. Re-emit what we already have so the client's
            # accumulated delta buffer is corrected, then stop revising.
            if not revised:
                yield _emit("notes_revised", "revise", notes)
                yield _emit("status", "revise",
                            "Revision was cut off — keeping the previous version."
                            if cut_off else
                            "Revision returned nothing — keeping the previous version.")
                break

            notes = revised
            yield _emit("notes_revised", "revise", notes)

            # Re-critique the revised notes (against the possibly augmented context).
            yield _emit("status", "critique", f"Re-checking (round {rounds})...")
            timings.begin("critique", round=rounds)
            critique = critique_notes(
                notes, plan, mode, source=context, model=model, doc_sample=doc_sample,
                chunk_map=chunk_map,
            )
            timings.end()
            yield _emit("critique_done", "critique", _crit_msg(critique), critique)

            # Strictly better only: a revision that merely ties has not earned
            # the right to replace the version already in hand.
            if critique.get("score", 0) > best_critique.get("score", 0):
                best_notes, best_critique = notes, critique

        if rounds == 0:
            yield _emit("status", "revise", "No revision needed ✓")

        # A later revision can score lower — fall back to the best version seen.
        if best_notes != notes:
            notes, critique = best_notes, best_critique
            yield _emit("notes_revised", "revise", notes)
            yield _emit("critique_done", "critique", _crit_msg(critique), critique)

        # Citation verification (deterministic, no model call): drop any [n]
        # citation that doesn't point at a chunk the model was actually shown,
        # or whose chunk carries no usable page on a paged document. The page
        # itself is never something the model supplied.
        cleaned = validate_citations(notes, chunk_map, page_count=len(page_spans or []))
        if cleaned != notes:
            notes = cleaned
            yield _emit("notes_revised", "revise", notes)

        # Grounding check. The step above proves each citation points at a real
        # page; this one asks the different question of whether that page
        # actually says what the claim says. Runs on the helper model — it is a
        # mechanical per-claim judgement, not the writing that decides quality.
        if GROUNDING_CHECK and chunk_map:
            yield _emit("status", "critique", "Checking every claim against its source…")
            timings.begin("grounding", concurrency=GROUNDING_CONCURRENCY)
            try:
                grounded, gstats = verify_claim_support(notes, chunk_map, model=helper,
                                                        retriever=retriever)
            except Exception as exc:  # noqa: BLE001
                print(f"[grounding] check failed ({exc}); notes left as written.")
                grounded, gstats = notes, None
            timings.end(outcome="ok" if gstats is not None else "failed_open",
                        claims=(gstats or {}).get("checked", 0))
            if gstats and (gstats["removed"] or gstats["rewritten"]):
                # Grounding rewrites lines, so the citation guarantee above
                # must be re-established on what it returns.
                notes = validate_citations(grounded, chunk_map,
                                           page_count=len(page_spans or []))
                yield _emit(
                    "status", "critique",
                    f"Grounding: {gstats['rewritten']} claim(s) tightened, "
                    f"{gstats['removed']} unsupported claim(s) removed.",
                )
                yield _emit("notes_revised", "revise", notes)

        # Auto-title (best effort). NOT overlapped with grounding: the title is
        # generated from the notes AFTER grounding has edited them, so running
        # the two together would change the title.
        timings.begin("title")
        try:
            title = generate_title(notes, model=helper)
        except Exception:  # noqa: BLE001
            title = ""
        timings.end()
        if title:
            yield _emit("title_done", "title", title)

        # 8-9. Quiz (generate, then verify the answer key against the notes).
        # Skipped by default in the app — the user generates these on demand
        # from the Learn sidebar via /api/quiz and /api/flashcards.
        if include_quiz:
            yield _emit("status", "quiz", "Generating quiz...")
            timings.begin("quiz")
            quiz = generate_quiz(notes, n=5, model=helper)
            quiz = verify_quiz(notes, quiz, model=helper)
            timings.end()
            yield _emit("quiz_done", "quiz", quiz)

        # 10-11. Flashcards
        if include_flashcards:
            yield _emit("status", "flashcards", "Creating flashcards...")
            timings.begin("flashcards")
            cards = generate_flashcards(notes, n=8, model=helper)
            timings.end()
            yield _emit("flashcards_done", "flashcards", cards)

        # 12. Done
        # Which provider actually served the notes. Only non-secret fields:
        # provider name, model id, and a coarse reason slug.
        # The artifact must not present itself as complete when it is not.
        # This is deliberately the LAST edit to `notes`: the title, quiz and
        # flashcards above are generated from the clean notes, and the banner
        # then travels with the text into history, export and share - none of
        # which carry the event stream.
        if not coverage["complete"]:
            where = (f"pages {_page_ranges(coverage['failed_pages'])}"
                     if coverage["failed_pages"]
                     else f"{len(coverage['failed_windows'])} part(s)")
            notes = (
                f"> **Incomplete coverage** \u2014 {where} could not be generated "
                f"after retries. These notes cover the rest of the document.\n\n"
                + notes
            )
            yield _emit("notes_revised", "revise", notes)

        yield _emit("done", "complete", "All done!", {
            "provider": served.get("provider"),
            "model": served.get("model"),
            "fallback_used": served.get("fallback_used", False),
            "fallback_reason": served.get("fallback_reason"),
            "doc_type": doc_type,
            "coverage": coverage,
        })

    except UserFacingError as exc:
        if isinstance(exc, ProvidersUnavailableError):
            # Keep the per-provider reasons for the operator; the user only
            # learns that the providers are unavailable.
            log_unexpected_error("pipeline", exc)
        yield _emit("error", "error", exc.user_message)
    except Exception as exc:  # noqa: BLE001
        # Raw exception text can carry provider errors, URLs or internal
        # paths: log it with an id and give the user only the id.
        error_id = log_unexpected_error("pipeline", exc)
        yield _emit("error", "error",
                    "Something went wrong while generating your notes. Please "
                    f"try again. (error id: {error_id})")
    finally:
        # Never waits: after a blocked gate, an error or a cancellation the
        # index build is simply abandoned (it holds no slot and no model call).
        index_pool.shutdown(wait=False)