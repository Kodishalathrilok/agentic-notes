"""Long-document notes end to end, on faked model output (no keys).

Each test fakes what a window's writer returns and checks what reaches the
reader. The defects were observed together on one 55-page document: a heading
with nothing under it, a line about the passages, repeated topics, sections out
of page order, and bullets cut off mid-word.
"""
import re

import agent
import retrieval.semantic as sem
from retriever import page_spans

_PASSAGE = re.compile(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", re.S)


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


def _run(monkeypatch, stream, pages=None):
    monkeypatch.setattr(agent, "call_model", _model)
    monkeypatch.setattr(agent, "call_model_stream", stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    pg = pages or _pages()
    return list(agent.run_agent("\n\n".join(pg), "exam", "academic", "medium",
                                "bullet", include_quiz=False, include_flashcards=False,
                                page_spans=page_spans(pg)))


def _notes(events):
    return next((e["content"] for e in reversed(events)
                 if e["type"] in ("notes_revised", "notes_done")), "")


def _draft(events):
    return next(e["content"] for e in events if e["type"] == "notes_done")


def _ids(prompt):
    """Passage ids in the writer's CONTEXT block, in the order shown."""
    return [int(c) for c, _t in _PASSAGE.findall(prompt.split("CONTEXT (numbered passages")[-1])]


def _serve(on_serve):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")


# ---------------------------------------------------------------------------
# d) an empty heading and a line about the passages never reach the reader
# ---------------------------------------------------------------------------

def _junk(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    _serve(on_serve)
    cid = _ids(prompt)[0]
    yield (f"## Topic {cid}\n- Passage {cid} makes a concrete point here [{cid}].\n\n"
           f"## OWL Purpose {cid}\n(Passages do not detail the purpose of OWL.)\n")


def test_long_document_draft_has_no_empty_heading_or_passage_comment(monkeypatch):
    events = _run(monkeypatch, _junk)
    for notes in (_draft(events), _notes(events)):
        assert "## Topic" in notes
        assert "OWL Purpose" not in notes, "a heading with nothing under it survived"
        assert "Passages do not" not in notes, "a line about the passages survived"


def test_short_document_draft_is_cleaned_too(monkeypatch):
    pages = [f"Page {p}. " + " ".join(f"w{j}" for j in range(60)) for p in range(1, 4)]
    events = _run(monkeypatch, _junk, pages=pages)
    for notes in (_draft(events), _notes(events)):
        assert "## Topic" in notes
        assert "OWL Purpose" not in notes and "Passages do not" not in notes
