"""Grounding and critique fixes, each pinned by the test that first showed it.

a) A grounding rewrite could add a citation the line never had ([1] -> [999]),
   and nothing checked it: citations are validated BEFORE grounding runs.
"""

import json

import agent
import retrieval.semantic as sem

CHUNKS = {
    1: {"id": 1, "text": "Binary search runs in O(log n) time on a sorted array."},
    2: {"id": 2, "text": "Linear search inspects every element in turn."},
}


def _grounder(verdict_status, fix_text):
    def model(prompt, **kw):
        if "GROUNDING agent" in prompt:
            return json.dumps({"verdicts": [{"n": 1, "status": verdict_status}]})
        return json.dumps({"fixes": [{"n": 1, "text": fix_text}]})
    return model


# ---------------------------------------------------------------------------
# a) a rewrite may only keep citations the line already had
# ---------------------------------------------------------------------------

def test_partial_rewrite_cannot_introduce_a_citation_that_does_not_exist(monkeypatch):
    line = "• Binary search runs in O(log n) time and needs no sorting [1]"
    monkeypatch.setattr(agent, "call_model",
                        _grounder("partial", "Binary search runs in O(log n) time [999]"))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert "[999]" not in notes
    assert stats["rewritten"] == 0


def test_partial_rewrite_cannot_swap_in_a_different_real_citation(monkeypatch):
    line = "• Binary search runs in O(log n) time and needs no sorting [1]"
    monkeypatch.setattr(agent, "call_model",
                        _grounder("partial", "Binary search runs in O(log n) time [2]"))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert "[2]" not in notes
    assert stats["rewritten"] == 0


def test_a_rewrite_keeping_a_subset_of_the_citations_is_still_allowed(monkeypatch):
    line = "• Binary search runs in O(log n) time and inspects every element [1][2]"
    monkeypatch.setattr(agent, "call_model",
                        _grounder("partial", "Binary search runs in O(log n) time [1]"))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert notes.strip() == "• Binary search runs in O(log n) time [1]"
    assert stats["rewritten"] == 1


def _pipeline_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
                    on_serve=None, **kw):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    if "PLANNING agent" in prompt:
        return ('{"outline":["A"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}')
    if "CRITIQUE agent" in prompt:
        return '{"score":9,"needs_revision":false,"unsupported_claims":[],"missing_topics":[]}'
    if "gatekeeper" in prompt:
        return '{"academic":true,"subject":"cs","doc_type":"explanatory"}'
    return "Title"


def _stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None, **kw):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "• Binary search runs in O(log n) time on a sorted array [1]\n"


def _final_notes(events):
    return next(e["content"] for e in reversed(events)
                if e["type"] in ("notes_revised", "notes_done"))


def test_citations_are_validated_again_after_grounding(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _pipeline_model)
    monkeypatch.setattr(agent, "call_model_stream", _stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)

    def grounding_that_adds_a_bad_citation(notes, chunk_map, **kw):
        bad = notes.replace("[1]", "[1][999]")
        return bad, {"checked": 1, "supported": 0, "rewritten": 1, "removed": 0,
                     "unjudged": 0, "uncited_retrieved": 0, "skipped_partial_view": 0}

    monkeypatch.setattr(agent, "verify_claim_support", grounding_that_adds_a_bad_citation)
    events = list(agent.run_agent(
        "Binary search runs in O(log n) time on a sorted array. " * 20,
        "exam", "academic", "short", "bullet",
        include_quiz=False, include_flashcards=False))
    final = _final_notes(events)
    assert "[999]" not in final
    assert "[1]" in final
