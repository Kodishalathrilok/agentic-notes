"""The judges must see the evidence for the claims they judge.

Same class of bug as IncompleteStreamError (models.py): a PARTIAL view treated
as if it were COMPLETE. On a large source the writer's context covers the whole
document, but the critique used to judge faithfulness against `source[:12000]`,
the reviser saw `context[:40000]`, and the grounding step judged UNCITED claims
against the first GROUNDING_SOURCE_CHARS of the chunk map - and deleted
whatever it could not find there. A true claim about page 150 looked
"unsupported" only because the judge was never shown page 150.

The fake judge below is deliberately realistic: it answers "supported" ONLY if
the supporting source sentence is actually present in the evidence it was
shown. Every synthetic fact carries a unique token (FACT_k) that appears both
in its source sentence and in its claim, so these tests measure what the judge
can SEE rather than mocking the answer.
"""
import inspect
import random
import re

import pytest

import agent
import retrieval.semantic as sem
from retriever import Retriever

N_FACTS = 40
SEG_CHARS = 5000  # 40 x 5000 = ~200k chars, facts spread uniformly


def _fact_sentence(k):
    return f"FACT_{k} records that the quillon{k} gauge reads {7000 + k} units."


def _support_key(k):
    # Present in the SOURCE sentence only - a claim never contains it.
    return f"FACT_{k} records that"


def _claim(k):
    return f"The quillon{k} gauge for FACT_{k} reads {7000 + k} units"


def _build_source():
    rng = random.Random(1234)
    vocab = [a + b for a in ("mor", "tal", "ven", "sil", "bra", "dek", "fu", "ga",
                             "ho", "ji", "ka", "lu")
             for b in ("ta", "ne", "ri", "so", "pu", "le", "mi", "xo", "ca", "de")]
    parts = []
    for k in range(N_FACTS):
        words, used = [], 0
        while used < SEG_CHARS - 100:
            w = rng.choice(vocab) + ("." if rng.random() < 0.08 else "")
            words.append(w)
            used += len(w) + 1
        mid = len(words) // 2
        parts.append(" ".join(words[:mid]) + " " + _fact_sentence(k) + " "
                     + " ".join(words[mid:]))
    return "\n\n".join(parts)


_SOURCE = None


def _source():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = _build_source()
    return _SOURCE


@pytest.fixture
def doc(monkeypatch):
    """Retriever + chunk_map + context built the way run_agent's windowed path
    builds them."""
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    text = _source()
    assert len(text) > agent.SECTION_DOC_THRESHOLD
    retriever = Retriever(text)
    chunk_map = {}
    for _title, win in agent._document_windows(retriever.chunks_meta):
        for c in agent._window_context(win, retriever, "exam"):
            chunk_map[c["id"]] = c
    context = agent._format_context([chunk_map[i] for i in sorted(chunk_map)])
    home = {}
    for k in range(N_FACTS):
        home[k] = next(cid for cid in sorted(chunk_map)
                       if _fact_sentence(k) in chunk_map[cid]["text"])
    return {"text": text, "retriever": retriever, "chunk_map": chunk_map,
            "context": context, "home": home}


def _notes(doc, facts=range(N_FACTS), cited=True):
    lines = ["# Gauges"]
    for k in facts:
        lines.append(f"- {_claim(k)}" + (f" [{doc['home'][k]}]" if cited else ""))
    return "\n".join(lines)


_FACT_RE = re.compile(r"FACT_(\d+)\b")


def _facts_in(text):
    return {int(m) for m in _FACT_RE.findall(text or "")}


# ---------------------------------------------------------------------------
# The realistic fake judge
# ---------------------------------------------------------------------------

class Judge:
    """Rules only on what is in front of it."""

    def __init__(self):
        self.critique_prompts = []
        self.critique_visible = []   # per critique call: set of facts whose support was shown
        self.grounding_visible = {}  # fact -> bool (support shown with that claim)

    # CRITIQUE: evidence = everything before NOTES, minus the coverage sample.
    def critique(self, prompt):
        self.critique_prompts.append(prompt)
        cut = prompt.rfind('NOTES:\n"""')
        evidence, notes = prompt[:cut], prompt[cut:]
        if "DOCUMENT SAMPLE" in evidence:
            evidence = evidence[:evidence.find("DOCUMENT SAMPLE")]
        visible = {k for k in range(1000) if _support_key(k) in evidence}
        self.critique_visible.append(visible)
        # Adversarial: flags ANY claim, cited or not, whose support it cannot
        # see - whatever the prompt tells it about uncited claims.
        unsupported = []
        for line in notes.split("\n"):
            for k in _facts_in(line):
                if k not in visible:
                    # A strict reviewer quotes the claim it rejects.
                    unsupported.append(line.strip().lstrip("- ").strip())
        score = 9 if not unsupported else 5
        return ('{"score": %d, "needs_revision": %s, "unsupported_claims": %s, '
                '"missing_topics": [], "issues": [], "strengths": []}'
                % (score, "false" if not unsupported else "true",
                   __import__("json").dumps(unsupported)))

    # GROUNDING: evidence for claim n = its own EVIDENCE block + any FULL SOURCE.
    def grounding(self, prompt):
        full = ""
        body = prompt
        # The shared-source block header, not the words in the instructions.
        if "\n\nFULL SOURCE" in prompt:
            i = prompt.find("\n\nFULL SOURCE")
            body, full = prompt[:i], prompt[i:]
        blocks = re.split(r"(?m)^CLAIM (\d+): ", body)
        verdicts = []
        for n_str, block in zip(blocks[1::2], blocks[2::2]):
            claim_line = block.split("\n", 1)[0]
            evidence = block.split("\n", 1)[1] if "\n" in block else ""
            ks = _facts_in(claim_line)
            ok = bool(ks) and all(_support_key(k) in evidence + full for k in ks)
            for k in ks:
                self.grounding_visible[k] = self.grounding_visible.get(k, False) or ok
            verdicts.append({"n": int(n_str), "status": "supported" if ok else "unsupported"})
        return __import__("json").dumps({"verdicts": verdicts})

    def call_model(self, prompt, max_tokens=1024, model=None, temperature=0.4,
                   json_mode=False, on_serve=None):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if "CRITIQUE agent" in prompt:
            return self.critique(prompt)
        if "GROUNDING agent" in prompt:
            return self.grounding(prompt)
        if "PLANNING agent" in prompt:
            return ('{"outline":["A","B","C"],"checklist":["x"],'
                    '"difficulty":"easy","suggested_format":"bullet"}')
        if "gatekeeper" in prompt:
            return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
        return "Title"


def _has(fn, param):
    return param in inspect.signature(fn).parameters


def _critique(doc, notes, judge, monkeypatch, **extra):
    monkeypatch.setattr(agent, "call_model", judge.call_model)
    kw = {"chunk_map": doc["chunk_map"]} if _has(agent.critique_notes, "chunk_map") else {}
    kw.update(extra)
    return agent.critique_notes(notes, {"checklist": []}, "exam",
                                source=doc["context"], **kw)


def _ground(doc, notes, judge, monkeypatch, with_retriever=True):
    monkeypatch.setattr(agent, "call_model", judge.call_model)
    kw = {}
    if with_retriever and _has(agent.verify_claim_support, "retriever"):
        kw["retriever"] = doc["retriever"]
    return agent.verify_claim_support(notes, doc["chunk_map"], **kw)


# ---------------------------------------------------------------------------
# 1. Measurement: evidence visibility rate
# ---------------------------------------------------------------------------

def _critique_rates(doc, judge, monkeypatch):
    out = _critique(doc, _notes(doc), judge, monkeypatch)
    prompt = judge.critique_prompts[-1]
    shown = judge.critique_visible[-1] & set(range(N_FACTS))
    named = prompt[prompt.find("NOT shown"):] if "NOT shown" in prompt else ""
    omitted_named = {k for k in range(N_FACTS) if k not in shown
                     and re.search(rf"\b{doc['home'][k]}\b", named)}
    flagged = set().union(*(_facts_in(c) for c in out["unsupported_claims"])) \
        if out["unsupported_claims"] else set()
    return len(shown) / N_FACTS, (len(shown) + len(omitted_named)) / N_FACTS, flagged


def test_evidence_visibility_rate(doc, monkeypatch):
    judge = Judge()
    crit_rate, crit_accounted, flagged = _critique_rates(doc, judge, monkeypatch)

    judge2 = Judge()
    _ground(doc, _notes(doc, cited=False), judge2, monkeypatch)
    ground_rate = sum(judge2.grounding_visible.get(k, False)
                      for k in range(N_FACTS)) / N_FACTS

    print(f"\n[visibility] source={len(doc['text'])} chars, "
          f"context={len(doc['context'])} chars, chunks={len(doc['chunk_map'])}, "
          f"facts={N_FACTS}")
    print(f"[visibility] critique evidence visibility (default budget): {crit_rate:.3f}")
    print(f"[visibility] grounding (uncited) evidence visibility:       {ground_rate:.3f}")
    assert crit_rate >= 0.95
    assert crit_accounted == 1.0
    assert not flagged
    assert ground_rate >= 0.95


# ---------------------------------------------------------------------------
# 2. Critique on a long document
# ---------------------------------------------------------------------------

def test_critique_has_no_false_unsupported_for_late_claims(doc, monkeypatch):
    judge = Judge()
    late = range(N_FACTS - 10, N_FACTS)
    out = _critique(doc, _notes(doc, late), judge, monkeypatch)
    assert out["unsupported_claims"] == []
    assert out["needs_revision"] is False
    assert out["evidence_cited"] == 10
    assert out["evidence_shown"] == 10


def test_critique_over_budget_names_omitted_and_never_flags_unseen(doc, monkeypatch):
    monkeypatch.setattr(agent, "CRITIQUE_CONTEXT_CHARS", 3000)
    judge = Judge()
    out = _critique(doc, _notes(doc), judge, monkeypatch)
    prompt = judge.critique_prompts[-1]
    shown = judge.critique_visible[-1]
    assert 0 < out["evidence_shown"] < out["evidence_cited"] == N_FACTS
    omitted = [doc["home"][k] for k in range(N_FACTS) if k not in shown]
    assert omitted
    # The omitted passages are named, so the judge knows its view is partial.
    for cid in omitted:
        assert re.search(rf"\b{cid}\b", prompt[prompt.find("NOT shown"):]), cid
    # And nothing whose passage was not shown is reported as unsupported.
    flagged = set().union(*(_facts_in(c) for c in out["unsupported_claims"])) \
        if out["unsupported_claims"] else set()
    assert not (flagged - shown)


def test_uncited_flag_is_deferred_to_grounding_in_cited_mode(doc, monkeypatch):
    judge = Judge()
    notes = _notes(doc, [N_FACTS - 1]) + f"\n- {_claim(N_FACTS - 2)}"
    out = _critique(doc, notes, judge, monkeypatch)
    assert out["unsupported_claims"] == []
    assert any(f"FACT_{N_FACTS - 2}" in d for d in out["deferred_to_grounding"])


def test_uncited_flag_kept_in_fallback_mode(doc, monkeypatch):
    judge = Judge()
    monkeypatch.setattr(agent, "call_model", judge.call_model)
    notes = f"- {_claim(N_FACTS - 2)}"
    out = agent.critique_notes(notes, {"checklist": []}, "exam", source=doc["context"],
                               chunk_map=doc["chunk_map"])  # cites nothing -> fallback
    assert any(f"FACT_{N_FACTS - 2}" in u for u in out["unsupported_claims"])
    assert out["deferred_to_grounding"] == []
    out2 = agent.critique_notes(notes, {"checklist": []}, "exam", source=doc["context"])
    assert any(f"FACT_{N_FACTS - 2}" in u for u in out2["unsupported_claims"])


def test_critique_without_chunk_map_keeps_old_behaviour(doc, monkeypatch):
    judge = Judge()
    monkeypatch.setattr(agent, "call_model", judge.call_model)
    agent.critique_notes(_notes(doc), {"checklist": []}, "exam", source=doc["context"])
    prompt = judge.critique_prompts[-1]
    assert doc["context"][:12000] in prompt
    assert doc["context"][:12001] not in prompt


# ---------------------------------------------------------------------------
# 3. Grounding of UNCITED claims on a long document
# ---------------------------------------------------------------------------

def test_true_uncited_late_claim_is_kept_with_retriever(doc, monkeypatch):
    judge = Judge()
    notes = f"- {_claim(N_FACTS - 1)}"
    out, stats = _ground(doc, notes, judge, monkeypatch)
    assert _claim(N_FACTS - 1) in out
    assert stats["removed"] == 0 and stats["supported"] == 1
    assert stats["uncited_retrieved"] == 1


def test_fabricated_uncited_claim_is_still_removed(doc, monkeypatch):
    judge = Judge()
    notes = f"- {_claim(N_FACTS - 1)}\n- The quillon999 gauge for FACT_999 reads 7999 units"
    out, stats = _ground(doc, notes, judge, monkeypatch)
    assert "FACT_999" not in out and _claim(N_FACTS - 1) in out
    assert stats["removed"] == 1


def test_no_retriever_over_budget_leaves_uncited_claims_untouched(doc, monkeypatch):
    judge = Judge()
    notes = _notes(doc, range(N_FACTS - 3, N_FACTS), cited=False)
    out, stats = _ground(doc, notes, judge, monkeypatch, with_retriever=False)
    assert out == notes
    assert stats["removed"] == 0
    assert stats["skipped_partial_view"] == 3
    assert stats["unjudged"] == 3


def test_cited_claims_unaffected_without_retriever(doc, monkeypatch):
    judge = Judge()
    notes = _notes(doc, range(N_FACTS - 3, N_FACTS), cited=True)
    out, stats = _ground(doc, notes, judge, monkeypatch, with_retriever=False)
    assert out == notes and stats["supported"] == 3


# ---------------------------------------------------------------------------
# 4. The reviser sees the passages the notes cite
# ---------------------------------------------------------------------------

def test_revise_prompt_contains_late_cited_passage(doc):
    k = N_FACTS - 1
    notes = _notes(doc, [0, k])
    critique = {"issues": [], "missing_topics": [], "unsupported_claims": []}
    prompt = agent._revise_prompt(notes, critique, "exam", {}, "bullet",
                                  context=doc["context"], chunk_map=doc["chunk_map"])
    assert _support_key(k) in prompt and _support_key(0) in prompt


def test_revise_prompt_includes_corrective_chunks_and_names_omitted(doc, monkeypatch):
    monkeypatch.setattr(agent, "REVISE_CONTEXT_CHARS", 2500)
    notes = _notes(doc)
    critique = {"issues": [], "missing_topics": [], "unsupported_claims": []}
    added = [max(doc["chunk_map"])]
    prompt = agent._revise_prompt(notes, critique, "exam", {}, "bullet",
                                  context=doc["context"], chunk_map=doc["chunk_map"],
                                  added_ids=added)
    assert doc["chunk_map"][added[0]]["text"] in prompt
    assert "NOT shown" in prompt


# ---------------------------------------------------------------------------
# 5. End to end on the windowed path
# ---------------------------------------------------------------------------

_PASSAGE_RE = re.compile(r"(?m)^\[(\d+)\] (.*)$")


def test_run_agent_keeps_true_late_claims(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    judge = Judge()
    late_uncited = N_FACTS - 2

    def stream(prompt, max_tokens=1024, model=None, temperature=0.4, on_serve=None,
               **kw):
        if on_serve:
            on_serve("nvidia", "test-model", False, "")
        if "REVISION agent" in prompt:
            # A realistic reviser does what it is told: it removes the claims
            # the critique listed as unsupported.
            listed = prompt.split("REMOVE or CORRECT", 1)[1].split("ADD these", 1)[0]
            bad = _facts_in(listed)
            cur = prompt.split('CURRENT NOTES:\n"""', 1)[1].rstrip('"')
            yield "\n".join(l for l in cur.split("\n") if not (_facts_in(l) & bad))
            return
        seen = set()
        for cid, text in _PASSAGE_RE.findall(prompt):
            for k in _facts_in(text):
                if k in seen or _support_key(k) not in text:
                    continue
                seen.add(k)
                if k == late_uncited:
                    yield f"- {_claim(k)}\n"  # true but uncited
                else:
                    yield f"- {_claim(k)} [{cid}]\n"

    monkeypatch.setattr(agent, "call_model", judge.call_model)
    monkeypatch.setattr(agent, "call_model_stream", stream)
    events = list(agent.run_agent(_source(), "exam", "academic", "medium", "bullet",
                                  include_quiz=False, include_flashcards=False))
    assert not [e for e in events if e["type"] == "error"], events[-1]
    final = [e["content"] for e in events if e["type"] in ("notes_done", "notes_revised")][-1]
    for k in (N_FACTS - 1, late_uncited, N_FACTS - 5):
        assert _claim(k) in final, k
