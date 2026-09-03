"""Large-document coverage: every meaningful section must reach the writer.

The failure this guards against is silent. Sections used to be outline topics,
each with its own retrieval, so the writer's whole view of a large source was
capped at SECTION_MAX_COUNT x SECTION_RETRIEVAL_K chunks — 48, whatever the
document's size — and a passage no query ranked highly was never written about
at all. Measured before the change: 9% of chunks on a 100-page document, 4% on
a 200-page one.

These tests are deterministic and make no model calls: coverage is now an
arithmetic property of how windows are built, so it can be asserted directly.
"""
import threading

import agent
import retrieval.semantic as sem
from retriever import Retriever, page_spans


# Distinct, deliberately unrelated concepts. A retrieval-gated pipeline tends to
# return the semantically dominant region; these have nothing in common, so a
# section that goes missing is unambiguous.
CONCEPTS = [
    ("ALPHACONCEPT", "photosynthesis chloroplast thylakoid"),
    ("BETACONCEPT", "mortgage amortisation interest schedule"),
    ("GAMMACONCEPT", "volcanic basalt magma stratigraphy"),
    ("DELTACONCEPT", "sonnet iambic pentameter volta"),
    ("EPSILONCONCEPT", "kubernetes scheduler pod eviction"),
    ("ZETACONCEPT", "mitochondrial haplogroup migration"),
]


def _synthetic_pages(pages_per_concept=3, chars=1200):
    """One page list where each concept owns a contiguous run of pages."""
    pages = []
    for marker, words in CONCEPTS:
        for i in range(pages_per_concept):
            filler = " ".join(f"{words} detail{i}x{j}" for j in range(chars // 45))
            pages.append(f"{marker} section. {filler}")
    return pages


def _index(pages):
    sem.semantic_available = lambda: False  # BM25 only: deterministic, no quota
    joined = "\n\n".join(p.strip() for p in pages)
    return Retriever(joined, spans=page_spans(pages), mode="bm25"), joined


# ---------------------------------------------------------------------------
# Windows partition the document
# ---------------------------------------------------------------------------

def test_windows_cover_every_chunk_exactly_once():
    r, _ = _index(_synthetic_pages())
    windows = agent._document_windows(r.chunks_meta)
    seen = [c["chunk_id"] for _, chunks in windows for c in chunks]
    assert sorted(seen) == sorted(c["chunk_id"] for c in r.chunks_meta)
    assert len(seen) == len(set(seen)), "a chunk must not appear in two windows"


def test_every_concept_reaches_the_writer():
    """The synthetic coverage test: no section may disappear."""
    r, _ = _index(_synthetic_pages())
    windows = agent._document_windows(r.chunks_meta)
    covered = " ".join(c["text"] for _, chunks in windows for c in chunks)
    missing = [m for m, _ in CONCEPTS if m not in covered]
    assert not missing, f"sections lost from writer coverage: {missing}"


def test_window_count_is_bounded_but_coverage_is_not():
    """A bigger document produces BIGGER windows, not more calls — cost stays
    bounded while coverage stays complete."""
    small, _ = _index(_synthetic_pages(pages_per_concept=2))
    large, _ = _index(_synthetic_pages(pages_per_concept=12))

    w_small = agent._document_windows(small.chunks_meta)
    w_large = agent._document_windows(large.chunks_meta)

    assert len(w_large) <= agent.COVERAGE_MAX_WINDOWS
    assert len(w_small) <= agent.COVERAGE_MAX_WINDOWS
    # Complete coverage at both sizes.
    for r, w in ((small, w_small), (large, w_large)):
        assert sum(len(c) for _, c in w) == len(r.chunks_meta)
    # The large document's windows really are larger.
    def avg(w):
        return sum(len(c["text"]) for _, cs in w for c in cs) / max(len(w), 1)
    assert avg(w_large) > avg(w_small)


def test_late_sections_are_not_lost():
    """Regression for the specific failure mode: the tail of a long document
    silently never being written about."""
    r, _ = _index(_synthetic_pages(pages_per_concept=8))
    windows = agent._document_windows(r.chunks_meta)
    last_chunk = max(c["chunk_id"] for c in r.chunks_meta)
    covered = {c["chunk_id"] for _, chunks in windows for c in chunks}
    assert last_chunk in covered
    tail = CONCEPTS[-1][0]
    assert any(tail in c["text"] for _, chunks in windows for c in chunks)


# ---------------------------------------------------------------------------
# Retrieval supplements a window; it never defines one
# ---------------------------------------------------------------------------

def test_window_keeps_its_own_chunks_when_retrieval_fails():
    """A Gemini 429 (or any retrieval error) must not delete a section."""
    r, _ = _index(_synthetic_pages())
    _, win = agent._document_windows(r.chunks_meta)[0]

    class Broken:
        def retrieve(self, *a, **k):
            raise RuntimeError("429 Too Many Requests")

    ctx = agent._window_context(win, Broken(), "exam")
    assert {c["id"] for c in ctx} == {c["chunk_id"] for c in win}


def test_window_context_adds_supplements_without_displacing_its_own():
    r, _ = _index(_synthetic_pages())
    _, win = agent._document_windows(r.chunks_meta)[0]
    ctx = agent._window_context(win, r, "exam")
    own = {c["chunk_id"] for c in win}
    assert own <= {c["id"] for c in ctx}, "a window must always keep its own chunks"


def test_window_context_preserves_page_provenance():
    """Citations must stay resolvable: every context chunk keeps its page."""
    r, _ = _index(_synthetic_pages())
    _, win = agent._document_windows(r.chunks_meta)[0]
    for c in agent._window_context(win, r, "exam"):
        assert c["id"] is not None
        assert c.get("page"), "page provenance lost in window context"


def test_empty_document_produces_no_windows():
    assert agent._document_windows([]) == []


# ---------------------------------------------------------------------------
# Coverage vs the old retrieval-gated selection
# ---------------------------------------------------------------------------

def test_coverage_beats_the_old_retrieval_ceiling():
    """The old path could show the writer at most SECTION_MAX_COUNT x
    SECTION_RETRIEVAL_K chunks however large the document was."""
    r, _ = _index(_synthetic_pages(pages_per_concept=10))
    old_ceiling = agent.SECTION_MAX_COUNT * agent.SECTION_RETRIEVAL_K
    new_covered = sum(len(c) for _, c in agent._document_windows(r.chunks_meta))
    assert len(r.chunks_meta) > old_ceiling, "fixture must exceed the old ceiling"
    assert new_covered == len(r.chunks_meta)
    assert new_covered > old_ceiling


def test_pipeline_writes_every_window(monkeypatch):
    """End to end with fake models: one generation call per window, and the
    document's own chunks all reach the sources event."""
    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
              on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if "PLANNING agent" in prompt:
            return ('{"outline":["A","B","C"],"checklist":["x"],'
                    '"difficulty":"easy","suggested_format":"bullet"}')
        if "CRITIQUE agent" in prompt:
            return '{"score":9,"needs_revision":false,"unsupported_claims":[],"missing_topics":[]}'
        if "GROUNDING agent" in prompt:
            return '{"verdicts":[]}'
        if "gatekeeper" in prompt:
            return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
        return "ok"

    calls = {"n": 0}
    lock = threading.Lock()

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        with lock:
            calls["n"] += 1
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "body "

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", stream)
    sem.semantic_available = lambda: False

    pages = _synthetic_pages(pages_per_concept=2)
    text = "\n\n".join(pages)
    events = list(agent.run_agent(text, "exam", "academic", "medium", "bullet",
                                  include_quiz=False, include_flashcards=False,
                                  page_spans=page_spans(pages)))
    assert "error" not in [e["type"] for e in events]

    statuses = [e["content"] for e in events if e["type"] == "status"]
    covering = [s for s in statuses if "Covering the whole document" in s]
    assert covering, "pipeline must announce document-wide coverage"

    sources = next(e["data"] for e in events if e["type"] == "sources")
    pages_seen = {s["page"] for s in sources if s.get("page")}
    assert pages_seen == set(range(1, len(pages) + 1)), (
        f"every page must reach the writer; missing {set(range(1, len(pages)+1)) - pages_seen}")
    assert calls["n"] >= 1
