"""B1: content assigned to a window must actually reach the writer.

The pre-commit audit found that `_document_windows` / `_window_context` selected
100% of the document while the writer prompt ended with `context[:16000]`. A
window assigned pages 1-17 forwarded pages 1-5. Selection coverage was 100%;
writer coverage was 62%, and every existing test passed because they all stopped
at selection.

These tests therefore assert on the prompt the REAL writer builds, not on the
output of a helper.
"""
import re

import agent
import retrieval.semantic as sem
from retriever import Retriever, page_spans

MAX_TEXT_CHARS = 300000  # main.py's production extraction ceiling


def _doc(npages, chars_per_page):
    return [f"Page {p}. " + " ".join(f"topic{p} detail{j}"
                                     for j in range(chars_per_page // 19))
            for p in range(1, npages + 1)]


def _index(pages, clamp=True):
    sem.semantic_available = lambda: False
    text = "\n\n".join(p.strip() for p in pages)
    if clamp and len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        pages = text.split("\n\n")
        text = "\n\n".join(pages)
    return Retriever(text, spans=page_spans(pages), mode="bm25"), pages


def _real_prompt(title, ctx_block):
    """The prompt agent.write_section_stream itself builds. Worst-case
    scaffolding: longest doc-type rule, full checklist, custom instructions."""
    captured = {}

    def fake(prompt, **kw):
        captured["p"] = prompt
        yield "x"

    original = agent.call_model_stream
    agent.call_model_stream = fake
    try:
        list(agent.write_section_stream(
            title, ctx_block, "deep study", "academic", "long", "bullet",
            checklist=["c" * 80] * 6, instructions="x" * 400,
            doc_type="mixed", is_part=True))
    finally:
        agent.call_model_stream = original
    return captured["p"]


def _delivery(retriever):
    """(selected_ids, delivered_ids, selected_pages, delivered_pages, max_prompt).

    A passage counts as delivered only if its FULL text is present in the
    prompt - a truncated tail would otherwise read as delivered.
    """
    sel_ids, got_ids, sel_pg, got_pg, biggest = set(), set(), set(), set(), 0
    for title, win in agent._document_windows(retriever.chunks_meta):
        ctx = agent._window_context(win, retriever, "exam")
        prompt = _real_prompt(title, agent._format_context(ctx))
        biggest = max(biggest, len(prompt))
        body = prompt.split("CONTEXT (numbered passages")[-1]
        for c in ctx:
            sel_ids.add(c["id"])
            if c.get("page"):
                sel_pg.add(c["page"])
            if f"[{c['id']}] {c['text']}" in body:
                got_ids.add(c["id"])
                if c.get("page"):
                    got_pg.add(c["page"])
    return sel_ids, got_ids, sel_pg, got_pg, biggest


# ---------------------------------------------------------------------------
# selected == delivered, at every size
# ---------------------------------------------------------------------------

def test_small_window_delivers_everything_selected():
    r, _ = _index(_doc(3, 1200))
    sel, got, spg, gpg, _ = _delivery(r)
    assert sel == got and spg == gpg


def test_medium_window_delivers_everything_selected():
    r, _ = _index(_doc(20, 2000))
    sel, got, spg, gpg, _ = _delivery(r)
    assert sel == got, f"{len(sel - got)} passages never reached the writer"
    assert spg == gpg


def test_oversized_document_is_subdivided_not_truncated():
    """The exact regression: a document big enough that even-sized windows
    would exceed the writer budget."""
    r, pages = _index(_doc(120, 2500))
    sel, got, spg, gpg, _ = _delivery(r)
    assert sel == got, f"{len(sel - got)} passages silently truncated"
    assert gpg == {c["page"] for c in r.chunks_meta if c.get("page")}


def test_no_page_is_lost_at_the_production_ceiling():
    """MAX_TEXT_CHARS = 300,000, the most the API will ever hand the pipeline."""
    r, pages = _index(_doc(400, 2500))
    all_pages = {c["page"] for c in r.chunks_meta if c.get("page")}
    sel, got, spg, gpg, _ = _delivery(r)
    assert spg == all_pages, "selection lost a page"
    assert gpg == all_pages, f"writer never saw pages {sorted(all_pages - gpg)}"
    assert sel == got


def test_many_short_pages_all_reach_the_writer():
    r, _ = _index(_doc(1000, 300))
    sel, got, spg, gpg, _ = _delivery(r)
    assert sel == got
    assert len(gpg) == 1000


# ---------------------------------------------------------------------------
# the budget is respected, and it is a real budget
# ---------------------------------------------------------------------------

def test_every_window_context_fits_the_writer_budget():
    r, _ = _index(_doc(200, 2500))
    for title, win in agent._document_windows(r.chunks_meta):
        block = agent._format_context(agent._window_context(win, r, "exam"))
        assert len(block) <= agent.WRITER_CONTEXT_CHARS, (
            f"{title}: {len(block)} chars exceeds the "
            f"{agent.WRITER_CONTEXT_CHARS} budget")


def test_prompt_stays_within_the_justified_input_budget():
    """Context budget + worst-case scaffolding, well inside provider limits."""
    r, _ = _index(_doc(200, 2500))
    _, _, _, _, biggest = _delivery(r)
    ceiling = agent.WRITER_CONTEXT_CHARS + 4000  # 3600 measured scaffolding
    assert biggest <= ceiling, f"largest prompt {biggest} > {ceiling}"


def test_budget_forces_more_windows_rather_than_a_bigger_one():
    """max_windows is a soft target; the context budget is the hard ceiling."""
    r, _ = _index(_doc(60, 2500))
    windows = agent._document_windows(r.chunks_meta, max_windows=1)
    assert len(windows) > 1, "an oversized single window must be subdivided"
    covered = sum(len(w) for _t, w in windows)
    assert covered == len(r.chunks_meta), "subdivision must not drop chunks"
    for _t, win in windows:
        block = agent._format_context(agent._window_context(win, r, "exam"))
        assert len(block) <= agent.WRITER_CONTEXT_CHARS


def test_subdivision_happens_on_chunk_boundaries():
    """Splits land between chunks, so no passage is ever cut in half."""
    r, _ = _index(_doc(120, 2500))
    windows = agent._document_windows(r.chunks_meta)
    seen = []
    for _t, win in windows:
        cids = [c["chunk_id"] for c in win]
        assert cids == list(range(cids[0], cids[-1] + 1)), "window not contiguous"
        seen += cids
    assert sorted(seen) == sorted(c["chunk_id"] for c in r.chunks_meta)
    assert len(seen) == len(set(seen))


def test_supplements_never_push_a_window_over_budget():
    r, _ = _index(_doc(120, 2500))
    for _t, win in agent._document_windows(r.chunks_meta):
        ctx = agent._window_context(win, r, "exam", supplement_k=50)
        block = agent._format_context(ctx)
        assert len(block) <= agent.WRITER_CONTEXT_CHARS
        own = {c["chunk_id"] for c in win}
        assert own <= {c["id"] for c in ctx}, "own chunks displaced by supplements"


def test_writer_prompt_contains_no_truncation_slice():
    """Guard against the slice being reintroduced."""
    import inspect
    src = inspect.getsource(agent.write_section_stream)
    assert "context[:" not in src, "window writer must not slice its context"
