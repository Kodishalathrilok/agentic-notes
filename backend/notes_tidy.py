"""
Deterministic passes over finished notes.

Pure text: no model calls, and nothing imported from agent.py (which imports
this). Every pass returns its input untouched when there is nothing to do, so
it is safe to run on any draft, any number of times.
"""

import re
from difflib import SequenceMatcher

# Any number of digits: chunk ids reach retriever.MAX_CHUNKS (6000), and the
# UI renders every bracketed number as a citation (NotesOutput.jsx), so a
# marker this pattern missed would be shown to the reader yet never checked.
CITATION_RE = re.compile(r"\[(\d+)\]")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HASH_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
# A bold-only line is a heading when it reads like a title. One that reads like
# a statement or a formula ("**E = mc^2**", "**Always cite the page.**") is
# content, and must not be removed as an empty heading.
# ponytail: a shape test, not a parse. A short bold title that IS the content
# under a heading would be dropped; tighten here if that is ever observed.
_BOLD_HEADING = re.compile(r"^\s*\*\*([A-Za-z0-9 &,:/()'’-]+?)\*\*:?\s*$")
_BOLD_LEVEL = 7  # below every '#' level

# --- lines the model wrote about its own sources ---------------------------
# Anchored on phrases a writer only uses when talking ABOUT what it was shown.
# Bare "context", "text", "sources" and "the passage of ..." are ordinary
# subject matter and are deliberately not matched.
_QUAL = r"(?:provided|given|retrieved|supplied|above)"
_ABOUT = (rf"(?:{_QUAL}\s+(?:passages?|excerpts?|context|text|material|sources?)"
          r"|source\s+(?:text|passages?|excerpts?)|passages?|excerpts?)")
_LACKS = (r"(?:do(?:es)?\s+not|don['’]t|doesn['’]t|did\s+not|didn['’]t|lacks?"
          r"|(?:contains?|provides?|offers?|gives?|includes?|has|have)\s+no)\b")
_META = (
    # "Passages do not detail ...", "The provided context does not cover ..."
    re.compile(rf"^(?:note:\s*)?(?:the\s+|these\s+|those\s+)?{_ABOUT}\s+{_LACKS}", re.I),
    # "... is not covered in the provided text", "not mentioned in the passages"
    re.compile(r"\b(?:not|never|no|\w+n['’]t)\b.{0,80}?\b(?:covered|mentioned|discussed"
               r"|detailed|described|provided|specified|addressed|included|stated|given"
               rf"|explained|found|available)\s+(?:in|by)\s+(?:the|these|those)\s+{_ABOUT}\b",
               re.I),
    # "Note: ..." about the passages
    re.compile(rf"^note:.*\b(?:the|these|those)\s+{_ABOUT}\b", re.I),
)
# "•" is the bullet the writer is TOLD to use (agent._format_instructions).
_LIST_MARK = re.compile(r"^(?:>\s*)*(?:[-*+•]\s+|\d+[.)]\s+)?")
_TAIL_PAREN = re.compile(r"\s*\(([^()]*)\)\s*([.;]?)\s*$")


def _is_meta(text: str) -> bool:
    core = _LIST_MARK.sub("", text.strip()).strip(" *_()")
    return any(p.search(core) for p in _META)


def _cut_tail_comment(line: str) -> str:
    """'- OWL adds classes [3] (the passages do not say which).' keeps the claim."""
    m = _TAIL_PAREN.search(line)
    if not m or not _is_meta(m.group(1)):
        return line
    head = line[:m.start()]
    if not _LIST_MARK.sub("", head.strip()).strip(" *_"):
        return line  # the comment is the whole line: dropped as a line instead
    return head if head.rstrip().endswith((".", "!", "?")) else head + m.group(2)


def _level(line: str) -> int:
    m = _HASH_HEADING.match(line)
    if m:
        return len(m.group(1))
    m = _BOLD_HEADING.match(line)
    if m and len(m.group(1).split()) <= 8 and not CITATION_RE.search(line):
        return _BOLD_LEVEL
    return 0


def _levels(lines):
    """Heading level of every line (0 = not a heading); fenced code is content."""
    out, fenced = [], False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
            out.append(0)
        else:
            out.append(0 if fenced else _level(line))
    return out


def headings(text: str):
    """The heading titles in `text`, markup stripped, top to bottom."""
    lines = (text or "").split("\n")
    return [line.strip().strip("#*_: ") for line, lvl in zip(lines, _levels(lines)) if lvl]


def _blank(line: str) -> bool:
    return not line.strip() or bool(_RULE.match(line))


def _drop_empty_headings(lines):
    """Remove headings with nothing under them, repeating until stable: removing
    an empty subsection can leave its parent empty too."""
    while True:
        levels = _levels(lines)
        drop = set()
        for i, lvl in enumerate(levels):
            if not lvl:
                continue
            j = i + 1
            while j < len(lines) and _blank(lines[j]):
                j += 1
            if j == len(lines) or 0 < levels[j] <= lvl:
                drop.update(range(i, j))
        if not drop:
            return lines
        lines = [ln for k, ln in enumerate(lines) if k not in drop]


def clean_notes(notes: str) -> str:
    """Remove lines about the passages, then headings left with nothing under them.

    A line carrying a citation is never dropped: it is a claim about the
    source, and whether it is supported is the grounding check's question.
    """
    lines = (notes or "").split("\n")
    kept, fenced = [], False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            line = _cut_tail_comment(line)
            if _is_meta(line) and not CITATION_RE.search(line):
                continue
        kept.append(line)
    kept = _drop_empty_headings(kept)
    return "\n".join(kept) if kept != lines else notes


# ---------------------------------------------------------------------------
# Merge pass: a long document is written a window at a time and joined, so the
# same topic can appear under several headings and in the wrong page order.
# ---------------------------------------------------------------------------

_BULLET = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+(\S.*)$")
_NEGATIONS = frozenset({"not", "no", "never", "without", "cannot", "neither", "nor"})
_HEADING_FILLER = frozenset({"the", "a", "an", "of", "to", "and", "in", "for", "on"})
_HEADING_NUMBER = re.compile(r"^[\s#*_]*\d+[.)]\s+")


def _words(text: str):
    return re.findall(r"[a-z0-9]+", CITATION_RE.sub(" ", text).lower())


def _same_bullet(later, earlier) -> bool:
    """`later` repeats `earlier`: it adds no word, negates nothing differently,
    and is worded almost identically.

    Deliberately strict. Similarity alone would merge "Process A uses 2 ATP"
    with "Process B uses 2 ATP"; dropping a distinct fact is worse than keeping
    a repeat. A paraphrase in different words is therefore NOT caught here.
    """
    if not set(later) <= set(earlier):
        return False
    if [w for w in later if w in _NEGATIONS] != [w for w in earlier if w in _NEGATIONS]:
        return False
    return SequenceMatcher(None, later, earlier, autojunk=False).ratio() >= 0.9


def _add_citations(line: str, ids) -> str:
    marks = "".join(f"[{i}]" for i in ids)
    last = None
    for last in CITATION_RE.finditer(line):
        pass
    if last:
        return line[:last.end()] + marks + line[last.end():]
    end = re.search(r"[.!?;:]?\s*$", line).start()
    return f"{line[:end]} {marks}{line[end:]}"


def _drop_repeated_bullets(lines):
    """Drop a bullet that repeats an earlier one, moving its citations onto the
    one kept so no page loses the citation that covered it."""
    # ponytail: every bullet is compared with every earlier one - fine for a
    # few hundred bullets; index by word set if notes ever get much longer.
    out, seen, fenced = [], [], False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _BULLET.match(line)
        words = _words(m.group(2)) if m else []
        if len(words) >= 4:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            # A bullet with nested lines under it is a parent, never dropped.
            has_children = bool(nxt.strip()) and (
                len(nxt) - len(nxt.lstrip()) > len(m.group(1)))
            dup = None if has_children else next(
                (k for k, earlier in seen if _same_bullet(words, earlier)), None)
            if dup is not None:
                extra = [c for c in dict.fromkeys(CITATION_RE.findall(line))
                         if f"[{c}]" not in out[dup]]
                if extra:
                    out[dup] = _add_citations(out[dup], extra)
                continue
            seen.append((len(out), words))
        out.append(line)
    return out


def _heading_key(line: str):
    """Same key = same topic: case, markup, numbering and word order ignored."""
    return frozenset(_words(_HEADING_NUMBER.sub("", line))) - _HEADING_FILLER


def _rstrip(items):
    end = len(items)
    while end and not items[end - 1][0].strip():
        end -= 1
    return items[:end]


def _join_bodies(section, body):
    """Append a later section's body to the earlier section with its heading."""
    section, body = _rstrip(section), _rstrip(body)
    while body and not body[0][0].strip():
        body = body[1:]
    if not body:
        return section
    both_bullets = _BULLET.match(section[-1][0]) and _BULLET.match(body[0][0])
    return section + ([] if both_bullets else [("", 0)]) + body


def _merge_block(items):
    """Merge and order the sections of one block, at its shallowest heading
    level, recursing into each section for the levels below.

    `items` is [(line, heading_level)]. Subsections travel with their parent.
    ponytail: windows that used different heading levels nest instead of
    merging (a '###' window lands under the previous '##'); order is kept,
    nothing is lost. Normalise levels per window if that is ever observed.
    """
    tops = [lvl for _line, lvl in items if lvl]
    if not tops:
        return items
    top = min(tops)
    starts = [i for i, (_line, lvl) in enumerate(items) if lvl == top]
    pre = items[:starts[0]]

    merged, where = [], {}
    for a, b in zip(starts, starts[1:] + [len(items)]):
        key = _heading_key(items[a][0])
        if key and key in where:
            merged[where[key]] = _join_bodies(merged[where[key]], items[a + 1:b])
        else:
            if key:
                where[key] = len(merged)
            merged.append(items[a:b])
    merged = [[sec[0]] + _merge_block(sec[1:]) for sec in merged]

    # Page order: chunk ids are document-ordered, so the first passage a
    # section cites places it. A section citing nothing stays behind the one
    # before it.
    keys, last = [], -1
    for sec in merged:
        m = CITATION_RE.search("\n".join(line for line, _lvl in sec))
        last = int(m.group(1)) if m else last
        keys.append(last)
    blocks = [pre] if any(line.strip() for line, _lvl in pre) else []
    blocks += [sec for _key, sec in sorted(zip(keys, merged), key=lambda p: p[0])]

    out = []
    for block in blocks:
        if out:
            out.append(("", 0))
        out.extend(_rstrip(block))
    return out


def merge_sections(notes: str) -> str:
    """One heading per topic, no repeated bullet, sections in page order."""
    lines = (notes or "").split("\n")
    kept = _drop_repeated_bullets(lines)
    out = [line for line, _lvl in _merge_block(list(zip(kept, _levels(kept))))]
    unchanged = [ln for ln in out if ln.strip()] == [ln for ln in lines if ln.strip()]
    return notes if unchanged else "\n".join(out)


def tidy(notes: str, merge: bool = False) -> str:
    """clean_notes, plus merge_sections for notes joined from several windows.

    The second clean removes a heading the merge left empty (every bullet under
    it was a repeat).
    """
    out = clean_notes(notes)
    if merge:
        out = clean_notes(merge_sections(out))
    return notes if out == notes else out.rstrip()


# ---------------------------------------------------------------------------
# Cut-off text: a writer that stopped in the middle of its last line.
# ---------------------------------------------------------------------------

_SENTENCE_END = (".", "!", "?", ":", ";", "|")


def _ends_sentence(line: str) -> bool:
    core = CITATION_RE.sub("", line).rstrip().rstrip("*_)\"'”’` ")
    return core.endswith(_SENTENCE_END)


def trailing_fragment(text: str, cut: bool = False) -> bool:
    """Does `text` stop in the middle of its last line?

    `cut` means the stream is KNOWN to have stopped short (the provider
    reported the token cap): then any last line that neither ends a sentence
    nor is followed by a line break is a fragment.

    Without it only the text can tell, and the test is strict so that a writer
    who simply does not use full stops is never re-run for nothing: unbalanced
    bold or code markers, or a last line that does not end a sentence when
    nearly all of the others (three at least) do.
    """
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return False
    if sum(1 for ln in lines if _FENCE.match(ln)) % 2:
        return True  # stopped inside a code block
    last = lines[-1]
    if _FENCE.match(last) or _level(last):
        return False  # a bare heading is clean_notes' job
    if last.count("**") % 2 or last.count("`") % 2:
        return True
    if _ends_sentence(last):
        return False
    if cut:
        return not text.endswith("\n")
    others = [ln for ln in lines[:-1] if not _level(ln) and not _FENCE.match(ln)]
    return len(others) >= 3 and sum(map(_ends_sentence, others)) >= 0.8 * len(others)


def drop_fragment(text: str, cut: bool = False) -> str:
    """`text` without its unfinished last line (or its unclosed code block)."""
    if not trailing_fragment(text, cut):
        return text
    lines = text.rstrip().split("\n")
    fences = [i for i, ln in enumerate(lines) if _FENCE.match(ln)]
    keep = fences[-1] if len(fences) % 2 else len(lines) - 1
    return "\n".join(lines[:keep]) + ("\n" if keep else "")
