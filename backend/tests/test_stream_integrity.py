"""A partial stream must never be mistaken for a complete one.

call_model_stream used to swallow a provider error once any delta had been
yielded, and none of the parsers looked at the provider's finish reason, so a
response cut off mid-way (connection drop) or at the token cap looked exactly
like a finished answer. Downstream that meant a truncated window counted as
full coverage, a truncated revision replaced good notes, and a chat answer
simply stopped.

The pipeline tests below fake the PROVIDER layer (models._dispatch_stream) and
keep the real call_model_stream, so they exercise the whole path.
"""
import json

import pytest
from fastapi.testclient import TestClient

import agent
import main
import models
import retrieval.semantic as sem
from retriever import page_spans


# ---------------------------------------------------------------------------
# call_model_stream
# ---------------------------------------------------------------------------

def _drain(gen):
    """Collect deltas and whatever the stream raised at the end (if anything)."""
    out, err = [], None
    try:
        for d in gen:
            out.append(d)
    except Exception as exc:  # noqa: BLE001
        err = exc
    return out, err


def test_interrupted_stream_raises_after_partial_output_and_does_not_fail_over(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: True)
    called = []

    def stream(prov, prompt, max_tokens, model, temperature):
        called.append(prov)
        yield "Hello "
        yield "world"
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    out, err = _drain(models.call_model_stream("x", model="nvidia/some-model"))

    assert out == ["Hello ", "world"], "the partial text is still delivered"
    assert err is not None, "stream ended normally - partial output looks complete"
    assert isinstance(err, models.IncompleteStreamError)
    assert err.reason == "interrupted"
    assert err.provider == "nvidia"
    assert err.chars == len("Hello world")
    assert called == ["nvidia"], "failing over after output would duplicate text"


def test_failure_before_any_output_still_fails_over(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ("nvidia", "gemini"))
    called = []

    def stream(prov, prompt, max_tokens, model, temperature):
        called.append(prov)
        if prov == "nvidia":
            raise ConnectionError("refused")
        yield "from gemini"

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    served = []
    out = "".join(models.call_model_stream(
        "x", model="nvidia/some-model",
        on_serve=lambda p, m, fb, r: served.append((p, fb))))
    assert out == "from gemini"
    assert called == ["nvidia", "gemini"]
    assert served == [("gemini", True)]


# ---------------------------------------------------------------------------
# Parsers: explicit length / MAX_TOKENS finish reasons
# ---------------------------------------------------------------------------

class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = [ln.encode("utf-8") for ln in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def _nvidia_lines(finish):
    last = {"choices": [{"delta": {}}]}
    if finish is not None:
        last["choices"][0]["finish_reason"] = finish
    return [
        "data: " + json.dumps({"choices": [{"delta": {"content": "Part "}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "one"}}]}),
        "data: " + json.dumps(last),
        "data: [DONE]",
    ]


def _gemini_lines(finish):
    last = {"candidates": [{"content": {"parts": [{"text": "one"}]}}]}
    if finish is not None:
        last["candidates"][0]["finishReason"] = finish
    return [
        "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Part "}]}}]}),
        "data: " + json.dumps(last),
    ]


def _ollama_lines(finish):
    last = {"response": "", "done": True}
    if finish is not None:
        last["done_reason"] = finish
    return [
        json.dumps({"response": "Part ", "done": False}),
        json.dumps({"response": "one", "done": False}),
        json.dumps(last),
    ]


def _install(monkeypatch, prov, lines):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == prov)
    resp = _FakeStreamResponse(lines)
    if prov == "nvidia":
        monkeypatch.setattr(models, "_nvidia_post", lambda body, stream=False: resp)
        return "nvidia/some-model"
    if prov == "gemini":
        monkeypatch.setattr(models, "_gemini_post", lambda *a, **k: resp)
        return "gemini-3.6-flash"
    monkeypatch.setattr(models.requests, "post", lambda *a, **k: resp)
    return models.OLLAMA_MODEL


@pytest.mark.parametrize("prov,lines,cut", [
    ("nvidia", _nvidia_lines("length"), True),
    ("nvidia", _nvidia_lines("stop"), False),
    ("nvidia", _nvidia_lines(None), False),
    ("gemini", _gemini_lines("MAX_TOKENS"), True),
    ("gemini", _gemini_lines("STOP"), False),
    ("gemini", _gemini_lines(None), False),
    ("ollama", _ollama_lines("length"), True),
    ("ollama", _ollama_lines("stop"), False),
    ("ollama", _ollama_lines(None), False),
])
def test_parser_finish_reason(monkeypatch, prov, lines, cut):
    model = _install(monkeypatch, prov, lines)
    out, err = _drain(models.call_model_stream("x", model=model))
    assert "".join(out) == "Part one", "all text is yielded before any error"
    if cut:
        assert isinstance(err, models.IncompleteStreamError), err
        assert err.reason == "max_tokens"
        assert err.provider == prov
        assert err.chars == len("Part one")
    else:
        # No finish reason at all is NOT treated as truncation: some catalog
        # models omit it, and flagging every such answer would be worse.
        assert err is None, err


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DOOMED = "Section 1."


def _pages(n=14, chars=1400):
    return [f"Section {i}. " + " ".join(f"topic{i} detail{j}" for j in range(chars // 20))
            for i in range(1, n + 1)]


def _critique(score, needs):
    return (f'{{"score":{score},"needs_revision":{"true" if needs else "false"},'
            '"unsupported_claims":[],"missing_topics":[],"issues":[],"strengths":[]}')


def _model_factory(critiques=None):
    seq = list(critiques or [])

    def _model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
               on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if "PLANNING agent" in prompt:
            return ('{"outline":["A","B","C"],"checklist":["x"],'
                    '"difficulty":"easy","suggested_format":"bullet"}')
        if "CRITIQUE agent" in prompt:
            return seq.pop(0) if seq else _critique(9, False)
        if "GROUNDING agent" in prompt:
            return '{"verdicts":[]}'
        if "gatekeeper" in prompt:
            return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
        return "Title"
    return _model


def _run(monkeypatch, stream, text, critiques=None, spans=None):
    monkeypatch.setattr(agent, "call_model", _model_factory(critiques))
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    monkeypatch.setattr(models, "_dispatch_stream", stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    return list(agent.run_agent(text, "exam", "academic", "medium", "bullet",
                                model="nvidia/some-model", include_quiz=False,
                                include_flashcards=False, page_spans=spans))


def _final_notes(events):
    revised = [e["content"] for e in events if e["type"] == "notes_revised"]
    return revised[-1] if revised else next(
        e["content"] for e in events if e["type"] == "notes_done")


def _coverage(events):
    done = [e for e in events if e["type"] == "done"]
    assert done, [e for e in events if e["type"] == "error"]
    return done[-1]["data"]["coverage"]


def test_sectioned_window_interrupted_is_recorded_as_failed(monkeypatch):
    def stream(prov, prompt, max_tokens, model, temperature):
        if "WRITING agent" in prompt and DOOMED in prompt:
            yield "partial window text "
            raise ConnectionError("reset")
        yield "complete window text. "

    pages = _pages()
    events = _run(monkeypatch, stream, "\n\n".join(pages), spans=page_spans(pages))
    cov = _coverage(events)
    assert cov["complete"] is False, "an interrupted window must not count as covered"
    assert len(cov["failed_windows"]) == 1
    assert cov["failed_windows"][0]["reason"] == "interrupted"
    assert _final_notes(events).startswith("> **Incomplete coverage**")


def test_interrupted_revision_never_becomes_the_notes(monkeypatch):
    def stream(prov, prompt, max_tokens, model, temperature):
        if "REVISION agent" in prompt:
            yield "Half a revi"
            raise ConnectionError("reset")
        yield "Draft notes."

    events = _run(monkeypatch, stream, "source text",
                  critiques=[_critique(5, True), _critique(9, False)])
    assert "revise_start" in [e["type"] for e in events]
    assert _final_notes(events) == "Draft notes."
    assert any("Revision was cut off" in e["content"]
               for e in events if e["type"] == "status")
    assert "error" not in [e["type"] for e in events]
    assert _coverage(events)["complete"] is True


def test_interrupted_single_pass_draft_is_marked_incomplete(monkeypatch):
    def stream(prov, prompt, max_tokens, model, temperature):
        yield "Draft that stops "
        raise ConnectionError("reset")

    events = _run(monkeypatch, stream, "source text")
    cov = _coverage(events)
    assert cov["complete"] is False
    assert cov["failed_windows"] == [{"window": "draft", "pages": [], "reason": "interrupted"}]
    final = _final_notes(events)
    assert final.startswith("> **Incomplete coverage**")
    assert "Draft that stops" in final


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------

def test_chat_interrupted_answer_ends_with_cut_off_notice(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    # The endpoint only accepts ids /api/models lists: offer this one.
    monkeypatch.setattr(models, "_nvidia_available", lambda: True)
    monkeypatch.setenv("NVIDIA_MODELS", "nvidia/some-model")

    def stream(prov, prompt, max_tokens, model, temperature):
        yield "The answer is "
        raise ConnectionError("reset")

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    client = TestClient(main.app)
    r = client.post("/api/chat", json={"notes": "some notes", "question": "why?",
                                       "model": "nvidia/some-model"})
    assert r.status_code == 200
    assert r.text.startswith("The answer is ")
    assert r.text.endswith("_[The answer was cut off — please ask again.]_")
