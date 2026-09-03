"""How document-wide coverage behaves when things break.

Windowing multiplied the number of generation calls a large document makes (one
per window, up to COVERAGE_MAX_WINDOWS). That raises the odds that at least one
call fails, so the failure path matters more than it used to: a single transient
error must not discard eleven finished windows, and it must not be swallowed
either.

A window is retried once when it produces nothing, so a test that needs a window
to stay broken must fail it on CONTENT, not on call order - keying off "the
first call" only exercises the retry.
"""
import threading

import agent
import retrieval.semantic as sem
from retriever import page_spans


def _pages(n=14, chars=1400):
    return [f"Section {i}. " + " ".join(f"topic{i} detail{j}" for j in range(chars // 20))
            for i in range(1, n + 1)]


# Page 1's text lands in the first window and nowhere else, so this identifies
# one specific window across retries and across concurrent workers.
DOOMED = "Section 1."


def _plan(prompt):
    if "PLANNING agent" in prompt:
        return ('{"outline":["A","B","C"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}')
    if "CRITIQUE agent" in prompt:
        return '{"score":9,"needs_revision":false,"unsupported_claims":[],"missing_topics":[]}'
    if "GROUNDING agent" in prompt:
        return '{"verdicts":[]}'
    if "gatekeeper" in prompt:
        return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
    return "Title"


def _run(monkeypatch, stream, model=None):
    def _model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
               on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        return _plan(prompt)

    monkeypatch.setattr(agent, "call_model", model or _model)
    monkeypatch.setattr(agent, "call_model_stream", stream)
    sem.semantic_available = lambda: False
    pages = _pages()
    return list(agent.run_agent("\n\n".join(pages), "exam", "academic", "medium",
                                "bullet", include_quiz=False, include_flashcards=False,
                                page_spans=page_spans(pages)))


def _types(events):
    return [e["type"] for e in events]


def _statuses(events):
    return [e["content"] for e in events if e["type"] == "status"]


def _notes(events):
    return next(e["content"] for e in events if e["type"] == "notes_done")


def _gaps(events):
    return [s for s in _statuses(events) if "could not be completed" in s]


def _windows(events):
    return [s for s in _statuses(events) if s.startswith("Writing section")]


# ---------------------------------------------------------------------------
# One broken window must not take the others down
# ---------------------------------------------------------------------------

def test_one_permanently_failing_window_does_not_lose_the_others(monkeypatch):
    """The regression this design would introduce if left unhandled: 1 of 12
    calls fails and the user gets nothing."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if DOOMED in prompt:
            raise RuntimeError("nvidia 500 upstream error")
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "surviving window content. "

    events = _run(monkeypatch, stream)
    assert "error" not in _types(events)
    notes = _notes(events)
    assert "surviving window content" in notes
    assert len(_windows(events)) > 1
    assert notes.count("surviving window content") == len(_windows(events)) - 1


def test_a_permanently_failed_window_is_reported_not_hidden(monkeypatch):
    """Retry must not paper over a window that genuinely cannot be written."""
    attempts = {"n": 0}
    lock = threading.Lock()

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if DOOMED in prompt:
            with lock:
                attempts["n"] += 1
            raise RuntimeError("provider exhausted")
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "body text for this window. "

    events = _run(monkeypatch, stream)
    assert _gaps(events), f"a lost section must be surfaced; got {_statuses(events)}"
    assert attempts["n"] == 2, f"the window should be tried twice, was {attempts['n']}"


def test_all_windows_failing_is_an_error_not_empty_notes(monkeypatch):
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        raise RuntimeError("every provider down")
        yield  # pragma: no cover

    events = _run(monkeypatch, stream)
    assert "error" in _types(events), "total failure must not present as success"
    assert "notes_done" not in _types(events)


def test_failed_window_leaves_no_empty_heading(monkeypatch):
    """A heading with nothing under it would read as 'the model had nothing to
    say about these pages' rather than as a failure."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if DOOMED in prompt:
            raise RuntimeError("boom")
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "**Topic:**\nreal content here. "

    notes = _notes(_run(monkeypatch, stream))
    lines = notes.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("**") and line.strip().endswith(":**"):
            assert "".join(lines[i + 1:]).strip(), f"heading {line!r} has no content"


def test_persistently_empty_window_is_a_gap_not_a_crash(monkeypatch):
    """NVIDIA returning a 200 with no content used to overwrite good notes with
    nothing; here it must simply not contribute a section."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if DOOMED in prompt:
            return
        yield "substantive content for this window. "

    events = _run(monkeypatch, stream)
    assert "error" not in _types(events)
    assert "substantive content" in _notes(events)
    assert _gaps(events), "an empty window must be reported, not silently dropped"


# ---------------------------------------------------------------------------
# Retry of a window that produced nothing
# ---------------------------------------------------------------------------

def test_transient_empty_window_is_retried_and_recovered(monkeypatch):
    """Observed live: both providers returned an empty stream for a window and
    two of six parts were lost. One retry recovers a transient failure."""
    lock = threading.Lock()
    seen = {}

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        with lock:
            key = DOOMED in prompt
            seen[key] = seen.get(key, 0) + 1
            nth = seen[key]
        if key and nth == 1:
            return  # empty stream, exactly as the outage produced
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "recovered content for this window. "

    events = _run(monkeypatch, stream)
    assert "error" not in _types(events)
    assert not _gaps(events), "a window recovered by retry must not be a reported gap"
    assert _notes(events).count("recovered content") == len(_windows(events))


def test_retry_does_not_duplicate_partial_output(monkeypatch):
    """A window that already streamed text must not be rewritten from scratch."""
    lock = threading.Lock()
    seen = {"n": 0}

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "UNIQUEMARKER partial text "
        if DOOMED in prompt:
            with lock:
                seen["n"] += 1
            raise RuntimeError("connection dropped mid-stream")

    events = _run(monkeypatch, stream)
    notes = _notes(events)
    assert notes.count("UNIQUEMARKER") == len(_windows(events)), (
        f"expected one fragment per window, got {notes.count('UNIQUEMARKER')} "
        f"for {len(_windows(events))} windows")
    assert seen["n"] == 1, "a partially-written window must not be retried"


# ---------------------------------------------------------------------------
# Neighbouring subsystems
# ---------------------------------------------------------------------------

def test_retrieval_failure_still_covers_the_whole_document(monkeypatch):
    """A Gemini 429 degrades supplements, never coverage: windows are built
    from the chunk list, not from retrieval."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if on_serve:
            on_serve("gemini", "test-model", True, "nvidia_rate_limited")
        yield "content. "

    class FlakyRetriever(agent.Retriever):
        def retrieve(self, *a, **k):
            raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(agent, "Retriever", FlakyRetriever)
    events = _run(monkeypatch, stream)
    assert "error" not in _types(events)
    sources = next(e["data"] for e in events if e["type"] == "sources")
    pages_seen = {s["page"] for s in sources if s.get("page")}
    assert pages_seen == set(range(1, 15)), (
        f"coverage must not depend on retrieval; missing {set(range(1,15)) - pages_seen}")


def test_grounding_failure_does_not_gut_the_notes(monkeypatch):
    """Fail open: a broken verifier must leave the written notes intact."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "- a substantive claim drawn from the source document here\n"

    def boom(*a, **k):
        raise RuntimeError("grounding provider down")

    monkeypatch.setattr(agent, "_verify_batch", boom)
    events = _run(monkeypatch, stream)
    assert "error" not in _types(events)
    assert "substantive claim" in _notes(events)


def test_done_event_still_reports_provider_after_a_partial_failure(monkeypatch):
    """Observability must survive the failure path."""
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if DOOMED in prompt:
            raise RuntimeError("nvidia down")
        if on_serve:
            on_serve("gemini", "gemini-3.6-flash", True, "nvidia_empty_response")
        yield "content. "

    events = _run(monkeypatch, stream)
    done = next((e for e in events if e["type"] == "done"), None)
    assert done is not None
    data = done.get("data") or {}
    assert data.get("provider") == "gemini"
    assert data.get("fallback_used") is True
