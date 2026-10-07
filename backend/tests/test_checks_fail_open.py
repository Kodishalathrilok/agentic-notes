"""A check that cannot run must never cost the user their notes, or their time.

Measured on a 55-page document: the critique reply could not be parsed, the
fallback answered "needs revision", and a 267-second full rewrite ran with no
finding behind it. The re-check after that rewrite then hit a provider outage
and the whole 17-minute run ended in an error with finished notes on screen.
"""
import agent
import retrieval.semantic as sem
from models import ProvidersUnavailableError
from retriever import page_spans

GOOD = ('{"score":9,"needs_revision":false,"unsupported_claims":[],'
        '"missing_topics":[],"issues":[],"strengths":[]}')
LOW = ('{"score":5,"needs_revision":true,"unsupported_claims":[],'
       '"missing_topics":[],"issues":[],"strengths":[]}')


def _model(critiques):
    """call_model fake. `critiques` is the list of replies the critic gives, in
    order; an Exception instance is raised instead of returned."""
    replies = list(critiques)

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
              on_serve=None):
        if "CRITIQUE agent" in prompt:
            reply = replies.pop(0) if replies else GOOD
            if isinstance(reply, Exception):
                raise reply
            return reply
        if "PLANNING agent" in prompt:
            return ('{"outline":["A","B"],"checklist":["x"],'
                    '"difficulty":"easy","suggested_format":"bullet"}')
        if "GROUNDING agent" in prompt:
            return '{"verdicts":[]}'
        if "gatekeeper" in prompt:
            return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
        return "Title"

    return model


def _stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "Revised notes." if "REVISION agent" in prompt else "Draft notes."


def _run(monkeypatch, critiques, pages=None):
    monkeypatch.setattr(agent, "call_model", _model(critiques))
    monkeypatch.setattr(agent, "call_model_stream", _stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    if pages is None:
        return list(agent.run_agent("source text", "exam", "academic", "medium", "bullet",
                                    include_quiz=False, include_flashcards=False))
    return list(agent.run_agent("\n\n".join(pages), "exam", "academic", "medium", "bullet",
                                include_quiz=False, include_flashcards=False,
                                page_spans=page_spans(pages)))


def _types(events):
    return [e["type"] for e in events]


def _notes(events):
    return next((e["content"] for e in reversed(events)
                 if e["type"] in ("notes_revised", "notes_done")), "")


def _long_pages(n=14, chars=1400):
    return [f"Section {i}. " + " ".join(f"topic{i} detail{j}" for j in range(chars // 20))
            for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# 1) a critique reply that cannot be read is "not judged", not "revise"
# ---------------------------------------------------------------------------

def test_an_unreadable_critique_is_not_a_verdict(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: "not json at all")
    result = agent.critique_notes("A claim [1].", {"checklist": []}, "exam",
                                  source="[1] some text")
    assert result["needs_revision"] is False, "an unreadable reply ordered a rewrite"
    assert result["score"] is None, "a made-up score reads as a verdict"
    assert result["issues"], "the reader must be told the draft was not scored"


def test_a_cut_off_critique_reply_is_not_a_verdict_either(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: '{"score": 4, "needs_rev')
    result = agent.critique_notes("A claim [1].", {"checklist": []}, "exam",
                                  source="[1] some text")
    assert result["needs_revision"] is False and result["score"] is None


def test_an_unreadable_critique_does_not_trigger_a_rewrite(monkeypatch):
    events = _run(monkeypatch, ["not json at all"])
    types = _types(events)
    assert "revise_start" not in types
    assert types[-1] == "done" and "error" not in types
    assert _notes(events) == "Draft notes."
    data = next(e["data"] for e in events if e["type"] == "critique_done")
    assert data["score"] is None and data["needs_revision"] is False


def test_a_readable_low_score_still_triggers_a_rewrite(monkeypatch):
    """The fix must not switch the revise loop off."""
    events = _run(monkeypatch, [LOW, GOOD])
    assert "revise_start" in _types(events)
    assert _notes(events) == "Revised notes."


# ---------------------------------------------------------------------------
# 2) a check that fails never discards the notes
# ---------------------------------------------------------------------------

def test_a_failed_critique_keeps_the_notes_and_finishes_the_run(monkeypatch):
    events = _run(monkeypatch, [ProvidersUnavailableError("nvidia: 503; gemini: 429")])
    types = _types(events)
    assert "error" not in types, "a failed CHECK ended a run that had written its notes"
    assert types[-1] == "done"
    assert _notes(events) == "Draft notes."
    verdict = next(e for e in events if e["type"] == "critique_done")
    assert verdict["data"]["score"] is None and verdict["data"]["needs_revision"] is False
    assert "could not" in verdict["content"], "the reader is not told the check did not run"


def test_a_failed_recheck_after_a_rewrite_keeps_the_scored_draft(monkeypatch):
    """Exactly what ended the measured run. A rewrite nobody could score has not
    earned the right to replace the draft that was scored."""
    events = _run(monkeypatch, [LOW, RuntimeError("connection reset")])
    types = _types(events)
    assert "revise_start" in types
    assert "error" not in types and types[-1] == "done"
    assert _notes(events) == "Draft notes."


def test_a_failed_critique_on_a_long_document_still_delivers_everything(monkeypatch):
    events = _run(monkeypatch, [ProvidersUnavailableError("all down")], pages=_long_pages())
    types = _types(events)
    assert "error" not in types and types[-1] == "done"
    done = next(e["data"] for e in events if e["type"] == "done")
    assert done["coverage"]["complete"] is True, "the coverage record never arrived"
    assert "title_done" in types, "the steps after the failed check did not run"


def test_a_failed_check_never_leaks_provider_detail(monkeypatch):
    secret = "https://integrate.api.nvidia.com/v1/chat?api_key=SECRET_VALUE"
    events = _run(monkeypatch, [RuntimeError(f"500 from {secret}")])
    assert "error" not in _types(events)
    blob = " ".join(str(e.get("content")) + str(e.get("data")) for e in events)
    for leak in ("SECRET_VALUE", "api_key", "integrate.api.nvidia.com"):
        assert leak not in blob


