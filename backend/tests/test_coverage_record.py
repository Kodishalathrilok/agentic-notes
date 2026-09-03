"""B2: an incomplete run must say so, durably.

A window that fails after retry and provider fallback used to survive only as a
transient status message. Once the stream ended, notes missing four pages were
byte-for-byte indistinguishable from complete notes in history, export and
share. The record now travels two ways: structured, on the `done` event, and in
the notes text itself, which is what history and export actually persist.
"""
import json

import agent
import retrieval.semantic as sem
from retriever import page_spans

DOOMED = "Section 1."  # page 1's text: identifies window 1 across retries


def _pages(n=14, chars=1400):
    return [f"Section {i}. " + " ".join(f"topic{i} detail{j}" for j in range(chars // 20))
            for i in range(1, n + 1)]


def _model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
           on_serve=None):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    if "PLANNING agent" in prompt:
        return ('{"outline":["A","B"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}')
    if "CRITIQUE agent" in prompt:
        return '{"score":9,"needs_revision":false,"unsupported_claims":[],"missing_topics":[]}'
    if "GROUNDING agent" in prompt:
        return '{"verdicts":[]}'
    if "gatekeeper" in prompt:
        return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
    return "Title"


def _run(monkeypatch, stream):
    monkeypatch.setattr(agent, "call_model", _model)
    monkeypatch.setattr(agent, "call_model_stream", stream)
    sem.semantic_available = lambda: False
    pg = _pages()
    return list(agent.run_agent("\n\n".join(pg), "exam", "academic", "medium",
                                "bullet", include_quiz=False, include_flashcards=False,
                                page_spans=page_spans(pg)))


def _coverage(events):
    done = next((e.get("data") or {} for e in events if e["type"] == "done"), {})
    return done.get("coverage")


def _notes(events):
    return next((e["content"] for e in reversed(events)
                 if e["type"] in ("notes_revised", "notes_done")), "")


def _ok(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "content. "


def _fail_window_one(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    if DOOMED in prompt:
        raise RuntimeError(
            "500 from https://integrate.api.nvidia.com/v1/chat?api_key=SECRET_VALUE")
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "content. "


# ---------------------------------------------------------------------------
# a complete run says so
# ---------------------------------------------------------------------------

def test_complete_run_is_marked_complete(monkeypatch):
    events = _run(monkeypatch, _ok)
    cov = _coverage(events)
    assert cov is not None, "every run must carry a coverage record"
    assert cov["complete"] is True
    assert cov["failed_pages"] == [] and cov["failed_windows"] == []
    assert cov["processed_pages"] == cov["total_pages"] == 14


def test_complete_run_notes_carry_no_warning(monkeypatch):
    assert "Incomplete coverage" not in _notes(_run(monkeypatch, _ok))


def test_small_document_single_pass_also_reports_coverage(monkeypatch):
    """The fast path must answer the question too, not omit it."""
    monkeypatch.setattr(agent, "call_model", _model)
    monkeypatch.setattr(agent, "call_model_stream", _ok)
    sem.semantic_available = lambda: False
    pg = [f"Page {p}. " + " ".join(f"w{j}" for j in range(60)) for p in range(1, 4)]
    events = list(agent.run_agent("\n\n".join(pg), "exam", "academic", "medium",
                                  "bullet", include_quiz=False,
                                  include_flashcards=False, page_spans=page_spans(pg)))
    cov = _coverage(events)
    assert cov["complete"] is True and cov["total_pages"] == 3


# ---------------------------------------------------------------------------
# an incomplete run cannot look complete
# ---------------------------------------------------------------------------

def test_failed_window_marks_the_run_incomplete(monkeypatch):
    cov = _coverage(_run(monkeypatch, _fail_window_one))
    assert cov["complete"] is False
    assert cov["failed_pages"], "the missing pages must be identified"
    assert cov["failed_windows"], "the failed window must be identified"


def test_failed_window_is_named_with_its_pages_and_reason(monkeypatch):
    cov = _coverage(_run(monkeypatch, _fail_window_one))
    win = cov["failed_windows"][0]
    assert set(win) == {"window", "pages", "reason"}
    assert win["pages"], "a failed window must say which pages it owned"
    assert win["reason"] == "runtimeerror"


def test_empty_output_is_reported_as_its_own_category(monkeypatch):
    def empty(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if DOOMED in prompt:
            return
        yield "content. "

    cov = _coverage(_run(monkeypatch, empty))
    assert cov["failed_windows"][0]["reason"] == "empty_output"


def test_the_notes_themselves_carry_the_warning(monkeypatch):
    """History, export and share persist `notes`, not the event stream."""
    notes = _notes(_run(monkeypatch, _fail_window_one))
    assert "Incomplete coverage" in notes
    assert notes.lstrip().startswith(">"), "the warning must lead the document"


def test_coverage_arithmetic_adds_up(monkeypatch):
    cov = _coverage(_run(monkeypatch, _fail_window_one))
    assert cov["processed_pages"] + len(cov["failed_pages"]) == cov["total_pages"]


def test_a_boundary_page_written_by_another_window_is_not_reported_lost(monkeypatch):
    """A page can straddle two windows; if either wrote it, it is not missing."""
    events = _run(monkeypatch, _fail_window_one)
    cov = _coverage(events)
    failed_win_pages = set(cov["failed_windows"][0]["pages"])
    assert set(cov["failed_pages"]) <= failed_win_pages
    assert set(cov["failed_pages"]) != failed_win_pages or len(failed_win_pages) == 1, (
        "a shared boundary page should be excluded from failed_pages")


def test_recovered_window_leaves_coverage_complete(monkeypatch):
    """Retry still works: a transient failure must not mark the run incomplete."""
    seen = {}

    def transient(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        key = DOOMED in prompt
        seen[key] = seen.get(key, 0) + 1
        if key and seen[key] == 1:
            raise RuntimeError("transient blip")
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        yield "content. "

    events = _run(monkeypatch, transient)
    cov = _coverage(events)
    assert cov["complete"] is True, "a recovered window is not a coverage gap"
    assert "Incomplete coverage" not in _notes(events)


# ---------------------------------------------------------------------------
# the record must never carry a secret
# ---------------------------------------------------------------------------

def test_coverage_record_never_leaks_provider_detail(monkeypatch):
    """Provider messages carry the request URL, which carries the API key."""
    events = _run(monkeypatch, _fail_window_one)
    blob = json.dumps(_coverage(events)) + _notes(events)
    for secret in ("SECRET_VALUE", "api_key", "https://", "integrate.api.nvidia.com"):
        assert secret not in blob, f"{secret!r} leaked into the coverage record"
