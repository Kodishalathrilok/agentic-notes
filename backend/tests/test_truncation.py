"""
Regression tests for the long-notes truncation bug.

Previously every prompt used head-truncation (`notes[:N]`), so on long
documents the TAIL of the notes was invisible to the critique (false
"missing topic" flags), the quiz/flashcards/chat (never covered later
sections), and — worst — the reviser, which then returned notes with the
tail silently deleted. These tests pin the fix.
"""

import agent

END_MARKER = "ZZZ_END_OF_NOTES_MARKER_ZZZ"


def _long_notes(total_chars: int) -> str:
    body = ("Section point about photosynthesis and cell biology. " * 400)
    notes = (body * (total_chars // len(body) + 1))[: total_chars - len(END_MARKER) - 1]
    return notes + " " + END_MARKER


# ---------------------------------------------------------------------------
# _notes_excerpt: even sampling that always includes the tail
# ---------------------------------------------------------------------------

def test_excerpt_returns_short_notes_unchanged():
    notes = "short notes " + END_MARKER
    assert agent._notes_excerpt(notes, 10000) == notes


def test_excerpt_always_includes_the_tail():
    notes = _long_notes(100_000)
    excerpt = agent._notes_excerpt(notes, 12000)
    assert END_MARKER in excerpt, "tail of long notes must be visible to read-only consumers"


def test_excerpt_respects_size_budget():
    notes = _long_notes(200_000)
    excerpt = agent._notes_excerpt(notes, 12000)
    # allow a little slack for the [...] separators
    assert len(excerpt) <= 12000 + 200


def test_excerpt_samples_beginning_too():
    notes = "AAA_START " + _long_notes(80_000)
    excerpt = agent._notes_excerpt(notes, 12000)
    assert "AAA_START" in excerpt


# ---------------------------------------------------------------------------
# Revise prompt: must contain the ENTIRE notes (up to the rewrite cap)
# ---------------------------------------------------------------------------

def test_revise_prompt_contains_tail_of_sectioned_notes():
    # Worst-case realistic sectioned output (~24k chars) fits under the cap,
    # so the reviser must see all of it, including the very end.
    notes = _long_notes(24_000)
    critique = {"issues": [], "missing_topics": [], "unsupported_claims": []}
    prompt = agent._revise_prompt(notes, critique, "exam", {}, "bullet", context="ctx")
    assert END_MARKER in prompt, "reviser must see the tail or it will delete it"


def test_rewrite_cap_covers_max_sectioned_output():
    # 8 sections x ~500 words (~3000 chars) = ~24k chars must fit the cap.
    assert agent.NOTES_REWRITE_CAP >= 24_000


# ---------------------------------------------------------------------------
# Quiz / flashcards / chat prompts see the tail
# ---------------------------------------------------------------------------

def _capture_prompt(monkeypatch):
    captured = {}

    def fake_call(prompt, **kwargs):
        captured["prompt"] = prompt
        return "stub"

    def fake_stream(prompt, **kwargs):
        captured["prompt"] = prompt
        yield "stub"

    monkeypatch.setattr(agent, "call_model", fake_call)
    monkeypatch.setattr(agent, "call_model_stream", fake_stream)
    return captured


def test_quiz_prompt_sees_tail(monkeypatch):
    captured = _capture_prompt(monkeypatch)
    agent.generate_quiz(_long_notes(40_000))
    assert END_MARKER in captured["prompt"]


def test_flashcards_prompt_sees_tail(monkeypatch):
    captured = _capture_prompt(monkeypatch)
    agent.generate_flashcards(_long_notes(40_000))
    assert END_MARKER in captured["prompt"]


def test_chat_prompt_sees_tail(monkeypatch):
    captured = _capture_prompt(monkeypatch)
    list(agent.chat_about_notes_stream(_long_notes(40_000), "what is at the end?"))
    assert END_MARKER in captured["prompt"]


def test_chat_history_items_are_capped(monkeypatch):
    captured = _capture_prompt(monkeypatch)
    huge_turn = "x" * 10_000
    list(agent.chat_about_notes_stream("notes", "q", history=[{"role": "user", "content": huge_turn}]))
    assert "x" * 2001 not in captured["prompt"], "history items must be truncated"


# ---------------------------------------------------------------------------
# Orchestrator safety valve: never run a lossy full revision
# ---------------------------------------------------------------------------

def test_oversized_notes_skip_revision_instead_of_truncating(monkeypatch):
    """If the draft exceeds the rewrite cap, the pipeline must keep the draft
    (with a status message), NOT feed a truncated copy to the reviser."""

    big = "word " * 8000  # ~40k chars > NOTES_REWRITE_CAP

    def fake_call(prompt, **kwargs):
        if "PLANNING agent" in prompt:
            return '{"outline":["A","B","C"],"checklist":["x"],"difficulty":"easy","suggested_format":"bullet"}'
        if "CRITIQUE agent" in prompt:
            # Critique demands a revision — the guard must refuse it.
            return '{"score":4,"needs_revision":true,"unsupported_claims":["bad claim"],"missing_topics":[],"issues":[],"strengths":[]}'
        if "gatekeeper" in prompt.lower():
            return '{"academic": true, "subject": "biology", "reason": ""}'
        return "stub"

    def fake_stream(prompt, **kwargs):
        yield big

    monkeypatch.setattr(agent, "call_model", fake_call)
    monkeypatch.setattr(agent, "call_model_stream", fake_stream)

    events = list(agent.run_agent("small source text", "exam", "academic", "medium", "bullet",
                                  include_quiz=False, include_flashcards=False))
    types = [e["type"] for e in events]

    assert "revise_start" not in types, "revision must be skipped for oversized notes"
    assert any(
        e["type"] == "status" and "too long" in (e.get("content") or "")
        for e in events
    ), "user should be told why revision was skipped"
    assert types[-1] == "done"


def test_classify_academic_survives_non_dict_json(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: "[1, 2, 3]")
    result = agent.classify_academic("some text")
    assert result["academic"] is True  # permissive fallback, no crash