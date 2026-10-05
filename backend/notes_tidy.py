"""
Deterministic passes over finished notes.

Pure text: no model calls, and nothing imported from agent.py (which imports
this). Every pass returns its input untouched when there is nothing to do, so
it is safe to run on any draft, any number of times.
"""

import re

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
_LIST_MARK = re.compile(r"^(?:>\s*)*(?:[-*+]\s+|\d+[.)]\s+)?")
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
