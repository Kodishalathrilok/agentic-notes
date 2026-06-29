"""
Pipeline orchestration test with a MOCKED model — no API key or network.

Verifies run_agent emits the right event sequence and that streamed note
deltas reassemble into the final notes.
"""

import agent


def _fake_call_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
    if "PLANNING agent" in prompt:
        return '{"outline":["A","B"],"checklist":["x"],"difficulty":"easy","suggested_format":"bullet"}'
    if "CRITIQUE agent" in prompt:
        # high score, no revision needed -> deterministic (no loop)
        return (
            '{"score":9,"needs_revision":false,"unsupported_claims":[],'
            '"missing_topics":[],"issues":[],"strengths":["clear"]}'
        )
    if "QUIZ VERIFIER" in prompt or "QUIZ agent" in prompt:
        return "Q1) Test?\nA) a\nB) b\nC) c\nD) d\nAnswer: A\nExplanation: because reasons."
    if "FLASHCARD agent" in prompt:
        return "CARD 1\nFront: f\nBack: b"
    if "title" in prompt.lower():
        return "Test Title"
    return "stub"


def _fake_stream(prompt, max_tokens=1400, model=None, temperature=0.4):
    yield "Some "
    yield "notes."


def test_pipeline_event_sequence(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("source text", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]

    for expected in ("plan_done", "notes_done", "critique_done", "quiz_done", "flashcards_done"):
        assert expected in types, f"missing {expected}"
    assert types[-1] == "done"
    assert "error" not in types

    deltas = [e["content"] for e in events if e["type"] == "notes_delta"]
    assert "".join(deltas) == "Some notes."


def test_pipeline_triggers_revision_when_needed(monkeypatch):
    calls = {"critique": 0}

    def model_low_then_high(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "CRITIQUE agent" in prompt:
            calls["critique"] += 1
            needs = "true" if calls["critique"] == 1 else "false"
            score = 5 if calls["critique"] == 1 else 9
            return (
                f'{{"score":{score},"needs_revision":{needs},'
                f'"unsupported_claims":[],"missing_topics":[],"issues":[],"strengths":[]}}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    monkeypatch.setattr(agent, "call_model", model_low_then_high)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("source text", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]

    # A low first critique should trigger a revise pass and a re-critique.
    assert "revise_start" in types
    assert types.count("critique_done") >= 2
    assert "error" not in types
