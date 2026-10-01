"""Grounding and critique fixes, each pinned by the test that first showed it.

a) A grounding rewrite could add a citation the line never had ([1] -> [999]),
   and nothing checked it: citations are validated BEFORE grounding runs.
b) A bare "unsupported" verdict deleted the line - no evidence asked for, so a
   misnumbered or careless verdict silently removed true claims - and a
   "partial" rewrite was accepted without anyone checking the new wording.
"""

import json

import agent
import retrieval.semantic as sem

CHUNKS = {
    1: {"id": 1, "text": "Binary search runs in O(log n) time on a sorted array."},
    2: {"id": 2, "text": "Linear search inspects every element in turn."},
}


def _shown(text):
    """How a claim appears in a grounding prompt: citations stripped."""
    return f"CLAIM 1: {agent._strip_markup(text)}\nEVIDENCE"


def _grounder(verdict_status, fix_text):
    """Verdict pass -> verdict_status; fix pass -> fix_text; a re-check of the
    rewrite (the rewritten claim shown for judging) -> supported, with a quote
    from the cited passage."""
    def model(prompt, **kw):
        if "GROUNDING agent" in prompt and _shown(fix_text) in prompt:
            return json.dumps({"verdicts": [{"n": 1, "status": "supported",
                                             "evidence_quote": CHUNKS[1]["text"]}]})
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


# ---------------------------------------------------------------------------
# b) deleting a line needs a quote Python can find; rewrites are re-checked
# ---------------------------------------------------------------------------

BINARY = "Binary search runs in O(log n) time on a sorted array"
LINEAR = "Linear search inspects every element in turn"


def _verdicts(*verdicts):
    """A grounding model whose verdict pass returns `verdicts` (by n)."""
    def model(prompt, **kw):
        if "GROUNDING agent" in prompt:
            return json.dumps({"verdicts": list(verdicts)})
        return json.dumps({"fixes": []})
    return model


def test_bare_unsupported_verdict_keeps_the_line_and_marks_it_unverified(monkeypatch):
    line = "• Binary search needs the array to be stored in a linked list [1]"
    monkeypatch.setattr(agent, "call_model", _verdicts({"n": 1, "status": "unsupported"}))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert notes == line
    assert stats["removed"] == 0 and stats["unverified"] == 1
    assert stats["unverified_lines"] == [line.strip()]


def test_unsupported_with_an_invented_quote_is_not_deleted(monkeypatch):
    line = "• Binary search needs the array to be stored in a linked list [1]"
    monkeypatch.setattr(agent, "call_model", _verdicts(
        {"n": 1, "status": "unsupported",
         "evidence_quote": "Binary search requires a balanced tree to be built first"}))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert notes == line and stats["unverified"] == 1


def test_unsupported_with_a_quote_from_its_own_evidence_is_deleted(monkeypatch):
    line = "• Binary search needs the array to be stored in a linked list [1]"
    monkeypatch.setattr(agent, "call_model", _verdicts(
        {"n": 1, "status": "unsupported", "evidence_quote": BINARY}))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert notes == "" and stats["removed"] == 1 and stats["unverified"] == 0


def test_a_quote_from_another_claims_evidence_does_not_count(monkeypatch):
    """A verdict numbered for the wrong claim quotes the wrong passage."""
    notes_in = ("• Binary search needs the array to be stored in a linked list [1]\n"
                "• Linear search inspects every element in turn [2]")
    monkeypatch.setattr(agent, "call_model", _verdicts(
        {"n": 1, "status": "unsupported", "evidence_quote": LINEAR},
        {"n": 2, "status": "supported", "evidence_quote": LINEAR}))
    notes, stats = agent.verify_claim_support(notes_in, CHUNKS)
    assert notes == notes_in
    assert stats["removed"] == 0 and stats["unverified"] == 1


def test_uncited_claim_is_deleted_only_with_a_quote_from_the_source_shown(monkeypatch):
    line = "• Binary search was invented by a committee in a single afternoon"
    monkeypatch.setattr(agent, "call_model", _verdicts(
        {"n": 1, "status": "unsupported", "evidence_quote": LINEAR}))
    notes, stats = agent.verify_claim_support(line, CHUNKS)
    assert notes == "" and stats["removed"] == 1


def _partial_then_recheck(recheck):
    """Verdict pass says partial; fix pass rewrites; re-check pass answers `recheck`."""
    fixed = "Binary search runs in O(log n) time [1]"

    def model(prompt, **kw):
        if "GROUNDING agent" in prompt and _shown(fixed) in prompt:
            return json.dumps({"verdicts": [recheck]})
        if "GROUNDING agent" in prompt:
            return json.dumps({"verdicts": [{"n": 1, "status": "partial",
                                             "evidence_quote": BINARY}]})
        return json.dumps({"fixes": [{"n": 1, "text": fixed}]})
    return model


ORIGINAL = "• Binary search runs in O(log n) time and was invented in 1946 [1]"


def test_partial_rewrite_is_kept_only_if_the_recheck_supports_it(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _partial_then_recheck(
        {"n": 1, "status": "supported", "evidence_quote": BINARY}))
    notes, stats = agent.verify_claim_support(ORIGINAL, CHUNKS)
    assert notes == "• Binary search runs in O(log n) time [1]"
    assert stats["rewritten"] == 1 and stats["unverified"] == 0


def test_partial_rewrite_that_fails_the_recheck_keeps_the_original(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _partial_then_recheck(
        {"n": 1, "status": "partial", "evidence_quote": BINARY}))
    notes, stats = agent.verify_claim_support(ORIGINAL, CHUNKS)
    assert notes == ORIGINAL
    assert stats["rewritten"] == 0 and stats["unverified"] == 1


def test_partial_rewrite_whose_recheck_has_no_real_quote_keeps_the_original(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _partial_then_recheck(
        {"n": 1, "status": "supported", "evidence_quote": "not a sentence from the source"}))
    notes, stats = agent.verify_claim_support(ORIGINAL, CHUNKS)
    assert notes == ORIGINAL and stats["unverified"] == 1
