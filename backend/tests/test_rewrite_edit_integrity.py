"""Inline edit and rewrite must never hand back partial notes as complete.

Same bug class as IncompleteStreamError and the judge-evidence fix: partial
input or output treated as the whole thing.

- edit_selection used to send notes[:30000] and ask for the COMPLETE notes
  back, so anything past 30k (and anything the output budget couldn't fit)
  vanished when the UI replaced the notes with the answer. It now asks only
  for the replacement of the selected passage and splices it in itself.
- rewrite_notes used the same truncated copy and a ~4k-token output cap. Long
  notes are now rewritten part by part, and any failed/truncated part fails
  the whole rewrite.
- the non-streaming call_model never looked at the finish reason; strict
  callers now get IncompleteStreamError when the provider hit the token cap.

Tests fake the PROVIDER layer (models._dispatch / the HTTP helpers) so the
real call_model logic runs.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

import agent
import main
import models


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _long_notes(sections=30, body_chars=1900, tail="TAIL_MARKER_END"):
    """~60k chars of sectioned markdown with a unique sentence per section."""
    parts = []
    for i in range(sections):
        filler = (f"Point {i} explains topic {i} in some depth. " * 60)[:body_chars]
        parts.append(f"## Section {i}\n\n- Unique fact number {i} [{i % 5 + 1}].\n- {filler}\n")
    return "\n".join(parts) + f"\n{tail}\n"


def _between(prompt, label):
    """Text inside the triple quotes that follow `label` in a prompt."""
    m = re.search(re.escape(label) + r'\s*"""(.*?)"""', prompt, re.S)
    return m.group(1) if m else None


def _fake_dispatch(monkeypatch, behaviour):
    """Only NVIDIA is 'ready'; behaviour(prompt, max_tokens, strict) -> text."""
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    calls = []

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode, strict=False):
        calls.append({"prompt": prompt, "max_tokens": max_tokens, "strict": strict})
        out = behaviour(prompt, max_tokens, strict)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(models, "_dispatch", dispatch)
    return calls


# ---------------------------------------------------------------------------
# 1. edit on long notes: only the selection changes, tail survives
# ---------------------------------------------------------------------------

SEL = "Unique fact number 28 [4]."
NEW = "Rewritten fact twenty-eight, now clearer [4]."


def _edit_model(prompt, max_tokens, strict):
    # Old prompt shape (asks for the whole document): behave like a PERFECT
    # whole-document model - return exactly what it was shown with the
    # selection replaced. It still loses the tail, because it was never shown.
    if "FULL NOTES:" in prompt:
        return _between(prompt, "FULL NOTES:").replace(SEL, NEW)
    return NEW


def test_edit_near_end_of_long_notes_keeps_everything_else(monkeypatch):
    notes = _long_notes()
    assert len(notes) > 55000 and notes.count(SEL) == 1
    assert notes.index(SEL) > len(notes) - 5000, "selection must be near the END"
    calls = _fake_dispatch(monkeypatch, _edit_model)

    out = agent.edit_selection(notes, SEL, "make it clearer", model="nvidia/some-model")

    assert out == notes.replace(SEL, NEW), "only the selection may change"
    assert len(out) == len(notes) - len(SEL) + len(NEW)
    assert out.rstrip().endswith("TAIL_MARKER_END")
    assert len(calls) == 1
    prompt = calls[0]["prompt"]
    assert "## Section 0\n" not in prompt, "the whole notes must not be sent"
    assert len(prompt) < 8000
    assert SEL in prompt
    assert calls[0]["strict"] is True
    # Output budget follows the selection, not the 60k document.
    assert calls[0]["max_tokens"] < 2000


# ---------------------------------------------------------------------------
# 2. locating the selection + endpoint error mapping
# ---------------------------------------------------------------------------

def test_edit_whitespace_normalised_match(monkeypatch):
    notes = "# T\n\n- alpha beta\n  gamma   delta [2].\n- keep this line\n"
    _fake_dispatch(monkeypatch, lambda p, m, s: "ALPHA GAMMA [2].")
    out = agent.edit_selection(notes, "alpha beta gamma delta [2].", "x",
                               model="nvidia/some-model")
    assert out == "# T\n\n- ALPHA GAMMA [2].\n- keep this line\n"


def test_locate_selection_errors():
    with pytest.raises(agent.SelectionNotFoundError):
        agent._locate_selection("one two three", "four")
    with pytest.raises(agent.SelectionAmbiguousError):
        agent._locate_selection("same line\nother\nsame line\n", "same line")
    with pytest.raises(agent.SelectionAmbiguousError):
        agent._locate_selection("a  b\nx\na b\n", "a\nb")


def _post(path, payload):
    return TestClient(main.app).post(path, json=payload)


def test_edit_endpoint_not_found_is_422(monkeypatch):
    calls = _fake_dispatch(monkeypatch, lambda p, m, s: "unused")
    r = _post("/api/edit-selection", {"notes": "hello world", "selection": "absent",
                                      "instruction": "x"})
    assert r.status_code == 422
    assert "select" in r.json()["detail"].lower()
    assert calls == [], "no model call when the selection can't be located"


def test_edit_endpoint_ambiguous_is_422(monkeypatch):
    _fake_dispatch(monkeypatch, lambda p, m, s: "unused")
    r = _post("/api/edit-selection", {"notes": "dup\nmid\ndup\n", "selection": "dup",
                                      "instruction": "x"})
    assert r.status_code == 422
    assert "longer" in r.json()["detail"].lower()


@pytest.mark.parametrize("reply", ["   ", '"""\n"""'])
def test_edit_endpoint_empty_replacement_is_502(monkeypatch, reply):
    _fake_dispatch(monkeypatch, lambda p, m, s: reply)
    r = _post("/api/edit-selection", {"notes": "keep\ntarget line\nkeep\n",
                                      "selection": "target line", "instruction": "x"})
    assert r.status_code == 502
    body = r.json()
    assert "notes" not in body
    assert "nothing was changed" in body["detail"].lower()


def test_edit_endpoint_success_shape(monkeypatch):
    _fake_dispatch(monkeypatch, lambda p, m, s: "new line")
    r = _post("/api/edit-selection", {"notes": "keep\ntarget line\nkeep",
                                      "selection": "target line", "instruction": "x"})
    assert r.status_code == 200
    # The whole response, exactly: the edited notes, and the flag saying this
    # text has not been through citation validation or grounding.
    assert r.json() == {"notes": "keep\nnew line\nkeep", "unverified": True}


# ---------------------------------------------------------------------------
# 3. rewrite of long notes goes part by part
# ---------------------------------------------------------------------------

def _upper_model(prompt, max_tokens, strict):
    return _between(prompt, "NOTES:").strip().upper()


def test_rewrite_long_notes_sends_every_part_in_order(monkeypatch):
    notes = _long_notes()
    calls = _fake_dispatch(monkeypatch, _upper_model)

    out = agent.rewrite_notes(notes, "clarity", model="nvidia/some-model")

    assert "TAIL_MARKER_END" in out
    # Every part was sent and the answers joined in order: the uppercased
    # output equals the uppercased input, modulo whitespace at the joins.
    assert " ".join(out.split()) == " ".join(notes.upper().split())
    sent = [_between(c["prompt"], "NOTES:") for c in calls]
    assert len(sent) > 1
    assert "".join(sent) == notes, "parts must tile the notes exactly"
    limit = agent.REWRITE_PART_CHARS
    for i, part in enumerate(sent):
        assert part.startswith("## Section"), "parts start on a heading"
        assert len(part) <= limit
        assert f"part {i + 1} of {len(sent)}" in calls[i]["prompt"]
    assert all(c["strict"] for c in calls)


def test_rewrite_short_notes_is_a_single_call(monkeypatch):
    calls = _fake_dispatch(monkeypatch, _upper_model)
    out = agent.rewrite_notes("## A\n\n- short notes\n", "shorter", model="nvidia/some-model")
    assert out == "## A\n\n- SHORT NOTES".upper()
    assert len(calls) == 1 and calls[0]["strict"] is True
    assert "part 1 of" not in calls[0]["prompt"]


def test_split_falls_back_to_paragraphs_and_never_splits_a_line():
    para = ("word " * 300).strip()  # 1499 chars, one line
    text = "\n\n".join(f"{para} {i}" for i in range(20))  # no headings
    parts = agent._split_for_rewrite(text, 4000)
    assert "".join(parts) == text
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= 4000
        assert p.lstrip().startswith("word")


def test_too_many_parts_is_refused(monkeypatch):
    monkeypatch.setattr(agent, "REWRITE_MAX_PARTS", 3)
    calls = _fake_dispatch(monkeypatch, _upper_model)
    with pytest.raises(agent.RewriteTooLongError):
        agent.rewrite_notes(_long_notes(), "clarity", model="nvidia/some-model")
    assert calls == []


# ---------------------------------------------------------------------------
# 4. a truncated part fails the whole rewrite
# ---------------------------------------------------------------------------

class _JsonResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_truncated_part_fails_whole_rewrite_with_502(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    # The endpoint only accepts ids /api/models lists: offer this one.
    monkeypatch.setattr(models, "_nvidia_available", lambda: True)
    monkeypatch.setenv("NVIDIA_MODELS", "nvidia/some-model")
    seen = []

    def post(body, stream=False):
        prompt = body["messages"][-1]["content"]
        seen.append(prompt)
        text = _between(prompt, "NOTES:").upper()
        finish = "length" if len(seen) == 3 else "stop"
        return _JsonResp({"choices": [{"message": {"content": text[:500] if finish == "length" else text},
                                       "finish_reason": finish}]})

    monkeypatch.setattr(models, "_nvidia_post", post)
    r = _post("/api/rewrite", {"notes": _long_notes(), "direction": "clarity",
                               "model": "nvidia/some-model"})
    assert r.status_code == 502
    body = r.json()
    assert "notes" not in body
    assert "nothing was changed" in body["detail"].lower()
    assert "nvidia" not in body["detail"].lower(), "no provider/exception text leaks"
    assert len(seen) == 3, "stop at the first failed part"


def test_rewrite_too_long_endpoint_is_422(monkeypatch):
    monkeypatch.setattr(agent, "REWRITE_MAX_PARTS", 2)
    _fake_dispatch(monkeypatch, _upper_model)
    r = _post("/api/rewrite", {"notes": _long_notes(), "direction": "shorter"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 5. call_model(strict=...) honours the token-cap finish reason
# ---------------------------------------------------------------------------

def _install(monkeypatch, prov, finish):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == prov)
    if prov == "nvidia":
        choice = {"message": {"content": "Part one"}}
        if finish is not None:
            choice["finish_reason"] = finish
        resp = _JsonResp({"choices": [choice]})
        monkeypatch.setattr(models, "_nvidia_post", lambda body, stream=False: resp)
        return "nvidia/some-model"
    if prov == "gemini":
        cand = {"content": {"parts": [{"text": "Part one"}]}}
        if finish is not None:
            cand["finishReason"] = finish
        resp = _JsonResp({"candidates": [cand]})
        monkeypatch.setattr(models, "_gemini_post", lambda *a, **k: resp)
        return "gemini-3.6-flash"
    data = {"response": "Part one", "done": True}
    if finish is not None:
        data["done_reason"] = finish
    resp = _JsonResp(data)
    monkeypatch.setattr(models.requests, "post", lambda *a, **k: resp)
    return models.OLLAMA_MODEL


CUT = {"nvidia": "length", "gemini": "MAX_TOKENS", "ollama": "length"}
STOP = {"nvidia": "stop", "gemini": "STOP", "ollama": "stop"}


@pytest.mark.parametrize("prov", ["nvidia", "gemini", "ollama"])
def test_strict_raises_on_token_cap(monkeypatch, prov):
    model = _install(monkeypatch, prov, CUT[prov])
    with pytest.raises(models.IncompleteStreamError) as ei:
        models.call_model("x", model=model, strict=True)
    assert ei.value.provider == prov
    assert ei.value.reason == "max_tokens"
    assert ei.value.chars == len("Part one")


@pytest.mark.parametrize("prov", ["nvidia", "gemini", "ollama"])
def test_non_strict_returns_truncated_text_unchanged(monkeypatch, prov):
    model = _install(monkeypatch, prov, CUT[prov])
    assert models.call_model("x", model=model) == "Part one"


@pytest.mark.parametrize("prov", ["nvidia", "gemini", "ollama"])
@pytest.mark.parametrize("finish_kind", ["stop", "none"])
def test_normal_finish_unaffected_by_strict(monkeypatch, prov, finish_kind):
    finish = STOP[prov] if finish_kind == "stop" else None
    model = _install(monkeypatch, prov, finish)
    assert models.call_model("x", model=model, strict=True) == "Part one"
    assert models.call_model("x", model=model) == "Part one"


def test_strict_truncation_fails_over_to_next_provider(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ("nvidia", "gemini"))
    monkeypatch.setattr(models, "_nvidia_post", lambda body, stream=False: _JsonResp(
        {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}))
    monkeypatch.setattr(models, "_gemini_post", lambda *a, **k: _JsonResp(
        {"candidates": [{"content": {"parts": [{"text": "whole"}]}, "finishReason": "STOP"}]}))
    served = []
    out = models.call_model("x", model="nvidia/some-model", strict=True,
                            on_serve=lambda p, m, fb, r: served.append((p, fb)))
    assert out == "whole"
    assert served == [("gemini", True)]


# ---------------------------------------------------------------------------
# 6. source-line anchors: the highlight is RENDERED text
# ---------------------------------------------------------------------------
# NotesOutput renders **bold** as <strong>, drops bullet markers and shows
# [n] as "p. N" chips, so the highlighted text rarely matches the markdown.
# The UI sends the source-line range (data-line) and the server edits those
# raw lines instead.

SRC_LINE = "- **Photosynthesis** converts light [3]"
RENDERED = "Photosynthesis converts light p. 3"
EDITED_LINE = "- **Photosynthesis** converts light energy into chemical energy [3]"


def _anchored_notes():
    notes = _long_notes()
    lines = notes.split("\n")
    idx = len(lines) - 6  # near the end, before the tail marker
    lines.insert(idx, SRC_LINE)
    return "\n".join(lines), idx


def test_anchored_edit_changes_exactly_that_line(monkeypatch):
    notes, idx = _anchored_notes()
    assert RENDERED not in notes
    calls = _fake_dispatch(monkeypatch, lambda p, m, s: EDITED_LINE)

    out = agent.edit_selection(notes, RENDERED, "say what it converts light into",
                               model="nvidia/some-model", line_start=idx, line_end=idx)

    before, after = notes.split("\n"), out.split("\n")
    assert len(after) == len(before)
    assert after[idx] == EDITED_LINE
    assert [ln for i, ln in enumerate(after) if i != idx] == \
        [ln for i, ln in enumerate(before) if i != idx], "other lines byte-identical"
    assert out.rstrip().endswith("TAIL_MARKER_END")
    prompt = calls[0]["prompt"]
    assert f'SOURCE LINES:\n"""{SRC_LINE}"""' in prompt, "raw markdown line is edited"
    assert RENDERED in prompt, "the highlight is passed as a hint"
    assert "## Section 0\n" not in prompt
    assert calls[0]["strict"] is True


def test_anchored_multiline_span_trims_blank_edges(monkeypatch):
    notes = "# T\n\n- a [1]\n  - b [2]\n\n- c\n"
    calls = _fake_dispatch(monkeypatch, lambda p, m, s: "- A [1]\n  - B [2]")
    out = agent.edit_selection(notes, "a p. 1 b p. 2", "caps", model="nvidia/some-model",
                               line_start=1, line_end=4)
    assert out == "# T\n\n- A [1]\n  - B [2]\n\n- c\n"
    assert '"""- a [1]\n  - b [2]"""' in calls[0]["prompt"]


def test_out_of_range_anchors_fall_back_to_text_matching(monkeypatch):
    notes = "keep\ntarget line\nkeep"
    _fake_dispatch(monkeypatch, lambda p, m, s: "new line")
    n = len(notes.split("\n"))
    for ls, le in [(n, n), (2, 1), (0, n + 3)]:
        assert agent.edit_selection(notes, "target line", "x", model="nvidia/some-model",
                                    line_start=ls, line_end=le) == "keep\nnew line\nkeep"


def test_endpoint_anchors_edit_rendered_selection(monkeypatch):
    notes, idx = _anchored_notes()
    _fake_dispatch(monkeypatch, lambda p, m, s: EDITED_LINE)
    r = _post("/api/edit-selection", {"notes": notes, "selection": RENDERED,
                                      "instruction": "x", "line_start": idx, "line_end": idx})
    assert r.status_code == 200
    assert r.json()["notes"] == notes.strip().replace(SRC_LINE, EDITED_LINE)


def test_endpoint_anchors_account_for_stripped_leading_lines(monkeypatch):
    notes = "\n\nA\n- **B** c [1]\nD"
    _fake_dispatch(monkeypatch, lambda p, m, s: "- **B** changed [1]")
    r = _post("/api/edit-selection", {"notes": notes, "selection": "B c p. 1",
                                      "instruction": "x", "line_start": 3, "line_end": 3})
    assert r.status_code == 200
    assert r.json()["notes"] == "A\n- **B** changed [1]\nD"


def test_rendered_selection_without_anchors_is_422(monkeypatch):
    # Why the anchors exist: the rendered text alone can't be found.
    notes, idx = _anchored_notes()
    calls = _fake_dispatch(monkeypatch, lambda p, m, s: EDITED_LINE)
    r = _post("/api/edit-selection", {"notes": notes, "selection": RENDERED, "instruction": "x"})
    assert r.status_code == 422
    r = _post("/api/edit-selection", {"notes": notes, "selection": RENDERED, "instruction": "x",
                                      "line_start": 10 ** 5, "line_end": 10 ** 5})
    assert r.status_code == 422, "out-of-range anchors fall back to text matching"
    assert calls == []


# ---------------------------------------------------------------------------
# 7. rewrite / edit output is never grounded, so it may not ADD a citation
# ---------------------------------------------------------------------------
#
# Neither endpoint is sent the sources - only the notes - so nothing here can
# check that a claim is supported. What CAN be checked is that the model did
# not attach a citation the notes never carried: such a marker points at a
# passage nobody verified for this text, and the UI would render it as a
# clickable source. The response is also marked "unverified" so the UI can say
# the edited text has not been through citation validation or grounding.

def test_strip_new_citations_keeps_known_and_drops_new():
    before = "Fact one [1]. Fact two [2][12]."
    after = "Fact one, restated [1]. A new claim [7]. Fact two [12][1234]."
    out = agent.strip_new_citations(before, after)
    assert "[7]" not in out and "[1234]" not in out
    assert "[1]" in out and "[12]" in out


def test_strip_new_citations_leaves_text_alone_when_nothing_is_new():
    # Two trailing spaces are a markdown line break. With no new citation
    # there is nothing to remove, so nothing at all may change.
    before = "Line one [1].  \nLine two [2]."
    after = "Line one, shorter [1].  \nLine two [2]."
    assert agent.strip_new_citations(before, after) == after


def test_rewrite_endpoint_strips_a_citation_the_notes_never_had(monkeypatch):
    _fake_dispatch(monkeypatch, lambda p, m, s: "Short fact [1]. Invented support [9].")
    r = _post("/api/rewrite", {"notes": "A long fact [1]. Another fact [2].",
                               "direction": "shorter"})
    assert r.status_code == 200
    body = r.json()
    assert "[9]" not in body["notes"]
    assert "[1]" in body["notes"]
    assert body["unverified"] is True


def test_edit_endpoint_strips_a_citation_the_notes_never_had(monkeypatch):
    _fake_dispatch(monkeypatch, lambda p, m, s: "new line [9]")
    r = _post("/api/edit-selection", {"notes": "keep [1]\ntarget line\nkeep",
                                      "selection": "target line", "instruction": "x"})
    assert r.status_code == 200
    assert r.json() == {"notes": "keep [1]\nnew line\nkeep", "unverified": True}


def test_edit_endpoint_keeps_a_citation_the_notes_already_had(monkeypatch):
    _fake_dispatch(monkeypatch, lambda p, m, s: "new line [1]")
    r = _post("/api/edit-selection", {"notes": "keep [1]\ntarget line\nkeep",
                                      "selection": "target line", "instruction": "x"})
    assert r.status_code == 200
    assert r.json() == {"notes": "keep [1]\nnew line [1]\nkeep", "unverified": True}
