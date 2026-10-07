"""Long-document notes end to end, on faked model output (no keys).

Each test fakes what a window's writer returns and checks what reaches the
reader. The defects were observed together on one 55-page document: a heading
with nothing under it, a line about the passages, repeated topics, sections out
of page order, and bullets cut off mid-word.
"""
import itertools
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


# ---------------------------------------------------------------------------
# c) windows are merged: one heading per topic, no repeated bullet, page order
# ---------------------------------------------------------------------------

_CALLS = itertools.count(1)  # a number no two windows share


def _repeats_itself(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    """Every window writes its LATER passage first, restarts a shared topic,
    and repeats one sentence word for word."""
    _serve(on_serve)
    ids = _ids(prompt)
    lo, hi, n = ids[0], ids[-1], next(_CALLS)
    yield (f"## Late part {hi}\n- Passage {hi} closes this part of the source [{hi}].\n\n"
           f"## Early part {lo}\n- Passage {lo} opens this part of the source [{lo}].\n\n"
           f"## Recurring Theme\n- Detail number {n} belongs to the recurring theme [{lo}].\n"
           f"- The same sentence is written again by every single part [{lo}].\n")


def _section_starts(notes):
    """The first passage each '##' section cites, top to bottom."""
    return [int(m.group(1)) for m in
            (agent._CITATION_RE.search(s) for s in notes.split("\n## ")) if m]


def test_windows_are_merged_into_one_heading_per_topic(monkeypatch):
    events = _run(monkeypatch, _repeats_itself)
    windows = sum(1 for e in events if e["type"] == "status"
                  and e["content"].startswith("Writing section"))
    assert windows > 2
    for notes in (_draft(events), _notes(events)):
        assert notes.count("## Recurring Theme") == 1
        assert notes.count("belongs to the recurring theme") == windows, (
            "merging must keep every window's own bullet")


def test_a_bullet_repeated_by_every_window_appears_once(monkeypatch):
    events = _run(monkeypatch, _repeats_itself)
    for notes in (_draft(events), _notes(events)):
        assert notes.count("The same sentence is written again") == 1


def test_sections_come_out_in_page_order(monkeypatch):
    events = _run(monkeypatch, _repeats_itself)
    for notes in (_draft(events), _notes(events)):
        starts = _section_starts(notes)
        assert len(starts) > 4
        assert starts == sorted(starts), f"sections out of page order: {starts}"


# ---------------------------------------------------------------------------
# b) a cut-off window is never stitched in: it is written again as two halves
# ---------------------------------------------------------------------------

DOOMED = "Section 1."  # page 1's text: in window 1, and in its first half
FRAGMENT = "is cut off mid-wor"


def _bullets(ids):
    return "".join(f"- Passage {c} makes point number {c} about its subject [{c}].\n"
                   for c in ids)


def _coverage(events):
    done = next((e.get("data") or {} for e in events if e["type"] == "done"), {})
    return done.get("coverage")


def _windows(events):
    return sum(1 for e in events if e["type"] == "status"
               and e["content"].startswith("Writing section"))


def _cut_off(how, times=1):
    """A writer whose first `times` calls for window 1 stop mid-bullet.

    how="cap": the provider reports the token cap. how="quiet": the stream just
    ends. how="drop": the connection drops. Every other call writes one whole
    bullet per passage it was shown.
    """
    seen = {"doomed": 0, "calls": [], "ids": []}

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        _serve(on_serve)
        ids = _ids(prompt)
        seen["calls"].append(1)  # windows run two at a time; append is atomic
        if DOOMED in prompt and seen["doomed"] < times:
            seen["doomed"] += 1
            seen["ids"] = seen["ids"] or ids
            yield _bullets(ids[:-1])
            yield f"- Passage {ids[-1]} {FRAGMENT}"
            if how == "cap":
                raise agent.IncompleteStreamError("nvidia", "max_tokens", 300)
            if how == "drop":
                raise RuntimeError("connection dropped mid-stream")
            return
        yield _bullets(ids)

    return stream, seen


def _cited(notes):
    return {int(n) for n in agent._CITATION_RE.findall(notes)}


def test_a_window_cut_at_the_token_cap_is_written_again_as_two_halves(monkeypatch):
    stream, seen = _cut_off("cap")
    events = _run(monkeypatch, stream)
    assert len(seen["calls"]) == _windows(events) + 2, "one cut-off window -> two more calls"
    for notes in (_draft(events), _notes(events)):
        assert FRAGMENT not in notes, "the cut-off bullet was stitched into the notes"
        assert set(seen["ids"]) <= _cited(notes), "the rewrite lost part of the window"
    cov = _coverage(events)
    assert cov["complete"] is True and cov["failed_windows"] == []


def test_the_reader_is_told_to_discard_the_cut_off_attempt(monkeypatch):
    stream, _seen = _cut_off("cap")
    events = _run(monkeypatch, stream)
    resets = [e for e in events if e["type"] == "notes_reset"]
    assert len(resets) == 1
    assert FRAGMENT not in resets[0]["content"]
    # everything streamed after the reset, plus what the reset kept, is the draft
    at = events.index(resets[0])
    live = resets[0]["content"] + "".join(
        e["content"] for e in events[at:] if e["type"] == "notes_delta")
    assert FRAGMENT not in live


def test_an_unfinished_last_bullet_is_caught_without_any_error(monkeypatch):
    """The provider sent no finish reason, so the stream just ended."""
    stream, seen = _cut_off("quiet")
    events = _run(monkeypatch, stream)
    assert len(seen["calls"]) == _windows(events) + 2
    assert FRAGMENT not in _notes(events)
    assert set(seen["ids"]) <= _cited(_notes(events))
    assert _coverage(events)["complete"] is True


def test_a_half_cut_off_again_loses_only_its_unfinished_bullet(monkeypatch):
    stream, seen = _cut_off("cap", times=2)  # the window, then its first half
    events = _run(monkeypatch, stream)
    notes, cov = _notes(events), _coverage(events)
    assert FRAGMENT not in notes
    assert set(seen["ids"][:2]) <= _cited(notes), "the half's whole bullets were dropped"
    assert len(cov["failed_windows"]) == 1
    half = cov["failed_windows"][0]
    assert set(half) == {"window", "pages", "reason"} and half["reason"] == "max_tokens"
    first = next(e["content"] for e in events if e["type"] == "status"
                 and e["content"].startswith("Writing section 1/"))
    assert half["window"] not in first, "the whole window was charged, not the half"
    assert set(cov["failed_pages"]) <= set(half["pages"])
    assert cov["processed_pages"] + len(cov["failed_pages"]) == cov["total_pages"]


def test_a_dropped_connection_keeps_whole_bullets_and_is_not_retried(monkeypatch):
    stream, seen = _cut_off("drop")
    events = _run(monkeypatch, stream)
    notes, cov = _notes(events), _coverage(events)
    assert len(seen["calls"]) == _windows(events), "a window that streamed text is not re-run"
    assert FRAGMENT not in notes
    assert set(seen["ids"][:-1]) <= _cited(notes), "whole bullets were dropped too"
    assert [w["reason"] for w in cov["failed_windows"]] == ["runtimeerror"]


def test_unpunctuated_single_line_output_is_not_mistaken_for_a_cut_off(monkeypatch):
    calls = {"n": 0}

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        _serve(on_serve)
        calls["n"] += 1
        yield "surviving window content"

    events = _run(monkeypatch, stream)
    assert calls["n"] == _windows(events)
    assert not [e for e in events if e["type"] == "notes_reset"]


# ---------------------------------------------------------------------------
# e) the writer is told what not to write, before the passes above tidy up
# ---------------------------------------------------------------------------

def _part_prompt(monkeypatch, context="[1] passage text", **kw):
    """The prompt the real writer builds for one part of a long document."""
    captured = {}

    def fake(prompt, **_kw):
        captured["p"] = prompt
        yield "x"

    monkeypatch.setattr(agent, "call_model_stream", fake)
    args = {"checklist": ["c" * 80] * 6, "instructions": "x" * 400, "doc_type": "mixed"}
    list(agent.write_section_stream("Pages 100–120", context, "deep study", "academic",
                                    "long", "bullet", is_part=True, **{**args, **kw}))
    return captured["p"]


def _head(prompt):
    """The instructions: everything before the passages."""
    return prompt.split("CONTEXT (numbered passages")[0]


def test_the_writer_is_not_told_to_overrun_its_own_budget(monkeypatch):
    """'Treat this as a minimum ... prefer more detail over brevity' sat next to
    a hard token cap, which is how bullets came to end mid-word."""
    head = _head(_part_prompt(monkeypatch))
    assert "Treat this as a minimum" not in head
    assert "Prefer more" not in head and "more\ndetail over brevity" not in head
    assert "never stop in the middle of a bullet" in head


def test_the_writer_is_told_to_leave_gaps_out_silently(monkeypatch):
    head = " ".join(_head(_part_prompt(monkeypatch)).split())
    assert "the passages or these instructions" in head  # in the "do NOT mention" list
    assert "Leave out silently whatever the passages do not cover" in head
    assert "never leave a heading empty" in head
    assert "End every bullet with a full stop" in head


def test_the_writer_is_told_which_passages_are_background(monkeypatch):
    head = _head(_part_prompt(monkeypatch, background_ids=[40, 41]))
    assert "[40], [41] are background from other parts" in head
    assert "no heading of their own" in head
    assert "background" not in _head(_part_prompt(monkeypatch))


def test_the_writer_is_told_which_headings_already_exist(monkeypatch):
    head = _head(_part_prompt(monkeypatch, covered=["RDF Triples", "OWL Classes"]))
    assert "do not restart them: RDF Triples; OWL Classes" in head
    assert "do not restart" not in _head(_part_prompt(monkeypatch))


def test_the_passages_still_arrive_whole_after_the_new_instructions(monkeypatch):
    context = "[7] first passage\n\n[8] second passage"
    prompt = _part_prompt(monkeypatch, context=context, background_ids=[8],
                          covered=["Earlier Topic"])
    assert prompt.endswith(context)


def test_the_instructions_stay_inside_the_prompt_size_ceiling(monkeypatch):
    """test_writer_context_budget allows 4,000 chars around the passages. The
    longest list of earlier headings and background passages must fit in it."""
    context = "[1] passage text"
    prompt = _part_prompt(
        monkeypatch, context=context,
        background_ids=list(range(5990, 6000)),
        covered=[f"A fairly long heading about topic number {i}" for i in range(80)])
    assert len(prompt) - len(context) <= 4000


# ---------------------------------------------------------------------------
# f) a part writes from its own passages, not from the document-wide plan
# ---------------------------------------------------------------------------
# Measured on a real 55-page run: every window was handed the same first six
# plan points, and the points themselves state facts ("Define Semantic Web per
# Berners-Lee: machine-processible web of smart data ..."). Three windows wrote
# those six points instead of their own pages, citing whatever passage they had
# - the cover page 15 times, page 52 sixteen times. Pages 46-51 got no notes at
# all, and the grounding check then deleted 34 of those lines as unsupported.

POINT = "Define the Semantic Web per Berners-Lee as a web of smart data"


def test_a_part_is_not_handed_the_document_wide_plan_points(monkeypatch):
    prompt = _part_prompt(monkeypatch, checklist=[POINT, "Contrast Web 1.0 with Web 2.0"])
    assert "Berners-Lee" not in prompt and "Web 2.0" not in prompt
    assert "plan point" not in prompt.lower()


def test_an_outline_section_still_gets_its_plan_points(monkeypatch):
    captured = {}

    def fake(prompt, **_kw):
        captured["p"] = prompt
        yield "x"

    monkeypatch.setattr(agent, "call_model_stream", fake)
    list(agent.write_section_stream("Overview", "[1] passage text", "exam", "academic",
                                    "medium", "bullet", checklist=[POINT]))
    assert "Cover any of these plan points that belong to this section:" in captured["p"]
    assert POINT in captured["p"]


def test_no_window_of_a_long_document_is_shown_the_plan_points(monkeypatch):
    prompts = []
    monkeypatch.setattr(agent, "plan_outline", lambda *a, **k: {
        "outline": ["A", "B"], "checklist": ["ZEBRAPOINT covers the whole document"],
        "difficulty": "easy", "suggested_format": "bullet"})
    events = _run(monkeypatch, _records_prompts(prompts))
    assert len(prompts) > 2 and events[-1]["type"] == "done"
    assert not [p for p in prompts if "ZEBRAPOINT" in p]


# ---------------------------------------------------------------------------
# g) the passages next door are the neighbours', not "related evidence"
# ---------------------------------------------------------------------------
# Consecutive passages overlap by 120 characters, so the passage just before a
# window always looks related to it. Measured on a real 55-page run: for 9 of
# 12 windows the one extra passage retrieved was exactly that one - the last
# passage of the previous window - and the window then wrote it up again under
# a heading of its own ("Problems Resolved by ..." / "Problems Addressed by ...").
# Excluding only that one passage just promoted the one before it, so a whole
# window's length either side is out.

class _Fixed:
    """A retriever that always answers with the given passage ids."""

    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, k=None):
        return [{"id": i, "text": f"passage {i}", "page": 1, "pages": [1]} for i in self.ids]


def _window(first, last):
    return [{"chunk_id": i, "text": f"own passage {i}", "page": 1} for i in range(first, last + 1)]


def test_the_passages_either_side_of_a_window_are_not_added_to_it():
    ctx = agent._window_context(_window(20, 29), _Fixed([19, 30, 18, 12, 38, 25, 7]), "exam")
    ids = [c["id"] for c in ctx]
    assert not {19, 30, 18, 12, 38} & set(ids), "a neighbouring window's passage was added"
    assert 7 in ids, "a passage from elsewhere in the document is still welcome"
    assert ids == sorted(ids) and set(range(20, 30)) <= set(ids)


def test_no_window_of_a_real_index_is_given_its_neighbours_edge(monkeypatch):
    from retriever import Retriever
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    pages = _pages(20, 2000)
    r = Retriever("\n\n".join(pages), spans=page_spans(pages), mode="bm25")
    windows = agent._document_windows(r.chunks_meta, window_chars=6000)
    assert len(windows) > 3
    for _title, win in windows:
        own = {c["chunk_id"] for c in win}
        extra = {c["id"] for c in agent._window_context(win, r, "exam")} - own
        near = set(range(min(own) - len(own), max(own) + len(own) + 1))
        assert not extra & near, f"{sorted(extra & near)} belong to a neighbouring window"


def _records_prompts(prompts):
    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        _serve(on_serve)
        prompts.append(prompt)
        cid = _ids(prompt)[0]
        yield (f"## Heading from call {len(prompts)}\n"
               f"- Passage {cid} makes a concrete point here [{cid}].\n")
    return stream


def test_a_later_window_is_told_the_headings_earlier_windows_wrote(monkeypatch):
    monkeypatch.setattr(agent, "SECTION_CONCURRENCY", 1)  # one at a time: deterministic
    prompts = []
    _run(monkeypatch, _records_prompts(prompts))
    assert len(prompts) > 2
    assert "do not restart" not in _head(prompts[0]), "nothing was written before part 1"
    assert "Heading from call 1" in _head(prompts[-1])


def test_passages_named_as_background_are_ones_the_window_was_shown(monkeypatch):
    prompts = []
    _run(monkeypatch, _records_prompts(prompts))
    named = 0
    for prompt in prompts:
        line = next((ln for ln in _head(prompt).split("\n")
                     if "are background from other parts" in ln), "")
        ids = [int(n) for n in agent._CITATION_RE.findall(line)]
        named += len(ids)
        assert set(ids) <= set(_ids(prompt)), "a passage the window never saw was named"
    assert named, "no window was told which of its passages come from elsewhere"
