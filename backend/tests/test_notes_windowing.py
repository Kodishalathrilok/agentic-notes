"""
Tests for the long-notes truncation fixes.

Previously, critique/quiz/flashcards/chat saw only the HEAD of long notes
(`notes[:N]`), and the revise step could rewrite from a truncated copy —
silently deleting the tail of long documents. These tests pin the fix.
"""

import agent


# ---------------------------------------------------------------------------
# _notes_excerpt: even sampling across the WHOLE notes
# ---------------------------------------------------------------------------

def test_excerpt_returns_short_notes_unchanged():
    notes = "short notes body"
    assert agent._notes_excerpt(notes, 1000) == notes


def test_excerpt_includes_tail_of_long_notes():
    # 60k chars: head marker at the start, tail marker at the end.
    notes = "HEADMARKER " + ("x" * 60000) + " TAILMARKER"
    out = agent._notes_excerpt(notes, 12000)
    assert len(out) <= 13000  # respects the cap (plus small separator slack)
    assert "HEADMARKER" in out
    assert "TAILMARKER" in out  # the old notes[:N] could never include this


def test_excerpt_handles_empty():
    assert agent._notes_excerpt("", 100) == ""
    assert agent._notes_excerpt(None, 100) == ""


# ---------------------------------------------------------------------------
# Revise prompt must not be head-truncated below the rewrite cap
# ---------------------------------------------------------------------------

def test_revise_prompt_contains_full_notes_up_to_cap():
    notes = "START " + ("y" * 25000) + " END"
    assert len(notes) <= agent.NOTES_REWRITE_CAP
    prompt = agent._revise_prompt(notes, {"issues": [], "missing_topics": [], "unsupported_claims": []},
                                  "exam", {}, "bullet")
    assert "START" in prompt
    assert " END" in prompt  # tail present — old cap (12000) dropped it


# ---------------------------------------------------------------------------
# Pipeline: oversized notes skip full-rewrite revision instead of truncating
# ---------------------------------------------------------------------------

def test_oversized_notes_skip_revision(monkeypatch):
    big_chunk = "word " * 80  # streamed repeatedly to build oversized notes

    def fake_call(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "PLANNING agent" in prompt:
            return '{"outline":["A","B","C"],"checklist":["x"],"difficulty":"easy","suggested_format":"bullet"}'
        if "CRITIQUE agent" in prompt:
            # LOW score => would normally trigger a revision round.
            return ('{"score":3,"needs_revision":true,"unsupported_claims":[],'
                    '"missing_topics":[],"issues":["weak"],"strengths":[]}')
        if "gatekeeper" in prompt.lower() or "GATEKEEPER" in prompt:
            return '{"academic": true, "subject": "test", "reason": ""}'
        return '{"academic": true, "subject": "test", "reason": ""}'

    def fake_stream(prompt, max_tokens=1400, model=None, temperature=0.4):
        # Emit enough to exceed NOTES_REWRITE_CAP overall across sections.
        for _ in range(120):
            yield big_chunk

    monkeypatch.setattr(agent, "call_model", fake_call)
    monkeypatch.setattr(agent, "call_model_stream", fake_stream)

    # Long source -> sectioned path; huge streamed sections -> oversized notes.
    source = "topic sentence. " * 2000  # > SECTION_DOC_THRESHOLD
    events = list(agent.run_agent(source, "exam", "academic", "medium", "bullet",
                                  include_quiz=False, include_flashcards=False))
    types = [e["type"] for e in events]

    assert "error" not in types
    # No revision pass happened (no revise_start), and the safety status fired.
    assert "revise_start" not in types
    statuses = [e["content"] for e in events if e["type"] == "status"]
    assert any("too long for a safe full revision" in s for s in statuses)
    assert types[-1] == "done"


# ---------------------------------------------------------------------------
# Gatekeeper survives a non-dict safe_json result
# ---------------------------------------------------------------------------

def test_classify_academic_handles_null_model_output(monkeypatch):
    monkeypatch.setattr(agent, "call_model",
                        lambda *a, **k: "null")  # safe_json -> None
    result = agent.classify_academic("some study text")
    assert result["academic"] is True  # permissive fallback, no crash