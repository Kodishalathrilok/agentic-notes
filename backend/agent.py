"""
Multi-agent notes pipeline.

run_agent(...) drives the whole pipeline as a synchronous generator that
yields event dicts of the shape: { type, step, content, data }.

Standalone helpers (used by the regenerate / chat / title endpoints):
    generate_quiz, generate_flashcards, rewrite_notes,
    generate_title, chat_about_notes_stream
"""

import os
import re
import queue as _queue
from concurrent.futures import ThreadPoolExecutor

from models import call_model, call_model_stream, safe_json, helper_model
from retriever import Retriever

# Quality thresholds for the self-improvement loop
REVISE_THRESHOLD = 8  # revise until score reaches this (1-10)
MAX_REVISION_ROUNDS = 2  # cap revision passes to bound latency

# Long-document handling: above this size, notes are written SECTION BY SECTION
# (map-reduce) — each outline section gets its own retrieval over the whole
# document, so no part of a large source is left out.
SECTION_DOC_THRESHOLD = 12000  # chars
SECTION_RETRIEVAL_K = 6  # chunks retrieved per section

# Single-pass (small-doc) path: how many chunks the writer sees, scaled by the
# requested length. The old fixed k=8 (~5.6k chars) starved "long" notes — the
# writer can't produce 1000-1500 words from < 1000 words of context. These pull
# a much larger slice of the document (candidate pool is 30) so the length
# targets are actually reachable.
SINGLE_PASS_K = {
    "short": 8,
    "medium": 14,
    "long": 24,
}
# Cap sections so a very large doc can't fire an unbounded burst of model calls
# (protects free-tier rate limits). Override with SECTION_MAX_COUNT.
SECTION_MAX_COUNT = int(os.getenv("SECTION_MAX_COUNT", "8"))
# How many sections are written CONCURRENTLY on the map-reduce path. Sections
# are independent (each has its own retrieved context), so overlapping the
# model calls cuts wall-clock time; keep this modest to respect free-tier
# rate limits (failover still covers 429s). Override with SECTION_CONCURRENCY.
SECTION_CONCURRENCY = int(os.getenv("SECTION_CONCURRENCY", "2"))

# Corrective re-retrieval (CRAG-style): when the critique reports missing
# topics, run a fresh retrieval PER TOPIC and add those chunks to the context
# before revising — the reviser can then actually fix omissions instead of
# being asked to add material it was never shown.
CORRECTIVE_TOPICS_MAX = 4  # cap topics queried per revision round
CORRECTIVE_K_PER_TOPIC = 3  # chunks retrieved per missing topic

# Above this size, the planner's even sample covers too little of the document
# (e.g. ~8% of a 100-page PDF), so a topic on only 2-3 pages can be invisible
# to it. Fix: a full-coverage DIGEST scan — the cheap helper model reads the
# ENTIRE document in large segments and lists each segment's topics, and the
# planner outlines from that 100%-coverage inventory instead of a sample.
DIGEST_DOC_THRESHOLD = int(os.getenv("DIGEST_DOC_THRESHOLD", "60000"))
DIGEST_SEGMENT_CHARS = 25000
DIGEST_MAX_SEGMENTS = 12  # 12 x 25k = full coverage of the 300k input cap


def _format_context(chunks) -> str:
    """Render retrieved chunks as numbered passages the agent can cite."""
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


# Deterministic citation verification: [n] markers are only kept if n is a
# chunk ID that was actually retrieved and shown to the model. This turns
# "don't invent citations" from a prompt instruction into a code guarantee.
_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def enforce_citations(notes: str, valid_ids) -> str:
    """Strip any [n] citation whose n is not a real retrieved-chunk ID.

    Pure text post-processing — no model call. Leaves markdown links
    (``[text](url)``) untouched because they never match a bare ``[digits]``
    pattern followed by nothing.
    """
    valid = set()
    for i in valid_ids:
        try:
            valid.add(int(i))
        except (TypeError, ValueError):
            continue

    def _sub(match):
        return match.group(0) if int(match.group(1)) in valid else ""

    cleaned = _CITATION_RE.sub(_sub, notes or "")
    # Tidy whitespace left behind by removals (trailing spaces before newlines).
    return re.sub(r"[ \t]+(\n)", r"\1", cleaned)


# ---------------------------------------------------------------------------
# Notes windowing: never let head-truncation hide the tail of long notes
# ---------------------------------------------------------------------------

# Hard cap on notes fed to a FULL-REWRITE step (revise / rewrite / edit).
# Sized to fit the largest sectioned output (8 sections x ~500 words ~ 28k
# chars) with headroom; a rewrite prompt must NEVER see a truncated copy,
# because the model can only return what it was shown -- truncation here
# silently deletes the tail of the document.
NOTES_REWRITE_CAP = int(os.getenv("NOTES_REWRITE_CAP", "30000"))


def _notes_excerpt(notes: str, max_chars: int) -> str:
    """Return the notes whole if they fit, else an EVEN SAMPLE across the
    entire notes. Used for read-only consumers (critique coverage, quiz,
    flashcards, chat): head-truncation (`notes[:N]`) made the tail of long
    notes invisible, producing false 'missing topic' flags and quizzes that
    never covered later sections."""
    notes = notes or ""
    if len(notes) <= max_chars:
        return notes
    n_seg = 6
    seg = max(1, max_chars // n_seg)
    parts = []
    # First n_seg-1 segments evenly spaced from the start of the notes...
    stride = max(1, (len(notes) - seg) // (n_seg - 1))
    for i in range(n_seg - 1):
        start = i * stride
        parts.append(notes[start:start + seg])
    # ...and the LAST segment anchored to the very END, so the tail of the
    # notes is always represented.
    parts.append(notes[-seg:])
    return "\n[...]\n".join(p for p in parts if p)


_CITE_RULE = (
    "Support each point with a citation to the passage number(s) it came from, "
    "in square brackets right after the point, e.g. [1] or [2][5]. Only cite "
    "numbers that appear in the CONTEXT. Do not invent citations."
)

# ---------------------------------------------------------------------------
# Tuning tables
# ---------------------------------------------------------------------------

LENGTH_TARGETS = {
    "short": "250-350 words",
    "medium": "500-700 words",
    "long": "1000-1500 words",
}

# Output token budget per length (enough to finish without truncation).
# Roughly 1.6 tokens/word plus headroom for markdown, citations, and headers.
LENGTH_MAX_TOKENS = {
    "short": 1200,
    "medium": 2200,
    "long": 4096,
    "xl": 6000,  # used when revising long sectioned notes
}


def _max_tokens(length: str) -> int:
    return LENGTH_MAX_TOKENS.get((length or "medium").lower(), LENGTH_MAX_TOKENS["medium"])


# Per-SECTION word budgets for long documents (map-reduce path).
SECTION_WORDS = {
    "short": "120-180 words",
    "medium": "220-320 words",
    "long": "350-500 words",
}
SECTION_MAX_TOKENS = {
    "short": 700,
    "medium": 1100,
    "long": 1600,
}

MODE_GUIDANCE = {
    "exam": "Focus on exam-critical facts, definitions, formulas, and likely "
    "test questions. Be concise and high-yield.",
    "revision": "Optimise for quick last-minute revision: key points, memory "
    "hooks, and easily scannable structure.",
    "deep study": "Explain concepts thoroughly with reasoning, context, and "
    "connections between ideas. Prioritise understanding.",
    "summary": "Produce a faithful, compact summary that captures the core "
    "ideas without losing essential detail.",
}

TONE_GUIDANCE = {
    "academic": "Use precise academic language.",
    "formal": "Use clear, professional, formal language.",
    "casual": "Use a friendly, conversational tone.",
    "simple": "Use simple, plain language a beginner can follow.",
}


def _format_instructions(fmt: str) -> str:
    fmt = (fmt or "bullet").lower()
    if fmt == "numbered":
        return (
            "FORMAT: Use a numbered list. Start each point with `1.`, `2.`, etc. "
            "Group points under bold section headers written as `**Header:**` on "
            "their OWN line (never put list content on the same line as a header)."
        )
    if fmt == "paragraph":
        return (
            "FORMAT: Write in short paragraphs. Begin each section with a bold "
            "header written as `**Header:**` on its OWN line, followed by the "
            "paragraph text. Use inline `**bold**` to emphasise key terms."
        )
    return (
        "FORMAT: Use bullet points. Start each bullet with `• `. Group bullets "
        "under bold section headers written as `**Header:**` on their OWN line. "
        "Use `- ` for sub-points indented under a bullet. Use inline `**bold**` "
        "for key terms. You may use `$...$` for inline math and triple-backtick "
        "fenced blocks for code."
    )


def _instr_block(instructions: str) -> str:
    instructions = (instructions or "").strip()
    if not instructions:
        return ""
    return f"\nADDITIONAL USER INSTRUCTIONS (you MUST follow these):\n{instructions}\n"


# ---------------------------------------------------------------------------
# Agent: Digest (full-coverage topic scan for very large documents)
# ---------------------------------------------------------------------------

def digest_document(text, model=None) -> str:
    """Scan the ENTIRE document segment by segment and return a merged topic
    inventory. Runs on the cheap helper model (its own token quota), so 100%
    coverage costs nothing from the main model's budget. A failed segment is
    skipped rather than failing the run — partial inventory still beats none."""
    segments = [
        text[i:i + DIGEST_SEGMENT_CHARS]
        for i in range(0, len(text), DIGEST_SEGMENT_CHARS)
    ][:DIGEST_MAX_SEGMENTS]

    inventory = []
    for i, seg in enumerate(segments, 1):
        prompt = f"""You are scanning part {i} of {len(segments)} of a document to build a
topic inventory. List the distinct topics and key concepts covered in THIS part.

Respond with ONLY a bullet list (max 10 bullets). Each bullet is a short,
specific topic phrase (3-8 words). No commentary, no numbering, no headers.

PART {i}:
\"\"\"{seg}\"\"\""""
        try:
            out = call_model(prompt, max_tokens=250, model=model, temperature=0.1)
        except Exception:  # noqa: BLE001
            continue
        if out and out.strip():
            inventory.append(out.strip())
    return "\n".join(inventory)


# ---------------------------------------------------------------------------
# Agent: Plan
# ---------------------------------------------------------------------------

def plan_outline(text, mode, tone, length, model=None, instructions="", doc_chars=None, topic_inventory="") -> dict:
    doc_chars = doc_chars or len(text or "")
    if doc_chars > 120000:
        outline_rule = (
            "This is a LARGE document (a book chapter or long report). Produce "
            "8-12 outline sections that together cover ALL of its major topics — "
            "do not skip parts of the document."
        )
    elif doc_chars > SECTION_DOC_THRESHOLD:
        outline_rule = (
            "This is a substantial document. Produce 5-8 outline sections that "
            "together cover ALL of its major topics — do not skip parts."
        )
    else:
        outline_rule = "Produce 3-5 outline sections covering the material."

    inventory_block = ""
    if (topic_inventory or "").strip():
        inventory_block = f"""
TOPIC INVENTORY — built by scanning the ENTIRE document part by part. This is
the complete list of topics the document contains. Your outline MUST cover all
major topics below (group closely related ones under one section):
{topic_inventory[:8000]}
"""

    prompt = f"""You are the PLANNING agent in a notes-generation pipeline.

Analyse the SOURCE material (an even sample spanning the WHOLE document) and
produce a study plan. {outline_rule}
{inventory_block}
Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone}
Target length: {LENGTH_TARGETS.get(length.lower(), '300-400 words')}
{_instr_block(instructions)}
Respond with ONLY a JSON object (no prose, no code fences) of this exact shape:
{{
  "outline": ["section heading 1", "section heading 2", "..."],
  "checklist": ["key point that MUST be covered", "..."],
  "difficulty": "beginner | intermediate | advanced",
  "suggested_format": "bullet | numbered | paragraph"
}}

SOURCE:
\"\"\"{text[:24000]}\"\"\""""

    data = safe_json(call_model(prompt, max_tokens=700, model=model, temperature=0.1, json_mode=True))

    if not data or "outline" not in data:
        return {
            "outline": ["Overview", "Key Concepts", "Important Details", "Summary"],
            "checklist": ["Define core terms", "Cover main ideas", "Highlight key takeaways"],
            "difficulty": "intermediate",
            "suggested_format": "bullet",
        }

    data.setdefault("outline", ["Overview", "Key Concepts", "Summary"])
    data.setdefault("checklist", ["Cover main ideas"])
    data.setdefault("difficulty", "intermediate")
    data.setdefault("suggested_format", "bullet")
    return data


# ---------------------------------------------------------------------------
# Agent: Write (prompt builder + streaming)
# ---------------------------------------------------------------------------

def _write_prompt(context, mode, tone, length, fmt, plan, instructions="") -> str:
    outline = plan.get("outline", [])
    checklist = plan.get("checklist", [])
    outline_str = "\n".join(f"- {o}" for o in outline) if outline else "- (derive a sensible outline)"
    checklist_str = "\n".join(f"- {c}" for c in checklist) if checklist else ""

    return f"""You are the WRITING agent in a notes-generation pipeline.

Write high-quality study notes grounded in the CONTEXT passages below.

Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone} — {TONE_GUIDANCE.get(tone.lower(), '')}
Length target: {LENGTH_TARGETS.get(length.lower(), '500-700 words')}. Treat this as a
MINIMUM to reach, not a ceiling — be thorough and comprehensive. Cover every outline
point in depth with concrete facts, definitions, examples, and explanations drawn from
the context. Do not pad with filler, but do not stop short: err on the side of MORE
detail and completeness. It is better to slightly exceed the target than to fall under it.

Follow this outline:
{outline_str}

Make sure you cover these points:
{checklist_str}

{_format_instructions(fmt)}
{_CITE_RULE}
{_instr_block(instructions)}
Write ONLY the notes themselves — no preamble, no closing remarks.

CONTEXT (numbered passages — cite these):
{context[:40000]}"""


def write_notes(context, mode, tone, length, fmt, plan, model=None, instructions="") -> str:
    return call_model(
        _write_prompt(context, mode, tone, length, fmt, plan, instructions),
        max_tokens=_max_tokens(length),
        model=model,
    )


def write_notes_stream(context, mode, tone, length, fmt, plan, model=None, instructions=""):
    yield from call_model_stream(
        _write_prompt(context, mode, tone, length, fmt, plan, instructions),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.5,
    )


# ---------------------------------------------------------------------------
# Agent: Section writer (map-reduce path for long documents)
# ---------------------------------------------------------------------------

def write_section_stream(section, context, mode, tone, length, fmt, checklist=None, model=None, instructions=""):
    """Write ONE outline section from its own retrieved context (streamed)."""
    related = "\n".join(f"- {c}" for c in (checklist or [])[:6])
    words = SECTION_WORDS.get((length or "medium").lower(), "120-180 words")

    prompt = f"""You are the WRITING agent producing ONE SECTION of a larger set of
study notes. Write ONLY the body of the section titled "{section}" — do NOT
repeat the section title, do NOT write other sections, no preamble.

Mode: {mode} — {MODE_GUIDANCE.get(mode.lower(), '')}
Tone: {tone} — {TONE_GUIDANCE.get(tone.lower(), '')}
Section length: {words}. Treat this as a minimum — cover this section's points
thoroughly with concrete facts and explanations from the context. Prefer more
detail over brevity.

Cover any of these plan points that belong to this section:
{related or '- (use your judgment)'}

{_format_instructions(fmt)}
{_CITE_RULE}
{_instr_block(instructions)}
CONTEXT (numbered passages retrieved for THIS section — cite these):
{context[:16000]}"""

    yield from call_model_stream(
        prompt,
        max_tokens=SECTION_MAX_TOKENS.get((length or "medium").lower(), 750),
        model=model,
        temperature=0.5,
    )


# ---------------------------------------------------------------------------
# Agent: Critique
# ---------------------------------------------------------------------------

def critique_notes(notes, plan, mode, source="", model=None, doc_sample="") -> dict:
    """
    Grounded critique. FAITHFULNESS is judged against `source` — the numbered
    CONTEXT passages the writer was actually given (the ground truth for what
    the notes were allowed to claim). COVERAGE is judged against `doc_sample`,
    an even breadth sample of the wider document, so topics the retrieval
    missed can still be reported as missing.

    `needs_revision` triggers on real signal only: the model's own flag, a
    score below threshold, or unsupported claims. `missing_topics` alone is
    advisory — it feeds corrective re-retrieval, not an automatic rewrite.
    """
    checklist = plan.get("checklist", [])
    checklist_str = "\n".join(f"- {c}" for c in checklist) if checklist else "(none)"

    sample_block = ""
    if (doc_sample or "").strip():
        sample_block = f"""
DOCUMENT SAMPLE — an even sample of the wider document, for judging COVERAGE
only (a topic present here but absent from the notes may be a missing topic;
do NOT use this block to judge faithfulness):
\"\"\"{doc_sample[:8000]}\"\"\"
"""

    prompt = f"""You are the CRITIQUE agent in a notes-generation pipeline. Be a
strict, fair reviewer for the "{mode}" study mode.

Judge the NOTES on THREE things:
1. FAITHFULNESS — does every claim in the notes actually appear in / follow from
   the CONTEXT passages below? The CONTEXT is the ONLY ground truth for this:
   list any statement that is fabricated, distorted, or unsupported by it.
2. COVERAGE — are any important points from the checklist or the document
   missing from the notes?
3. QUALITY — clarity, structure, and usefulness for studying.

Checklist that should be covered:
{checklist_str}

Respond with ONLY a JSON object of this exact shape:
{{
  "score": <integer 1-10>,
  "needs_revision": <true|false>,
  "unsupported_claims": ["claim in the notes NOT supported by the context", "..."],
  "missing_topics": ["important point the notes omitted", "..."],
  "issues": ["other quality problem", "..."],
  "strengths": ["what was done well", "..."]
}}

Scoring: deduct heavily for any unsupported_claims (faithfulness matters most).
A score of {REVISE_THRESHOLD} or above with NO unsupported claims means no
revision is needed.

CONTEXT (the passages the notes were written from — the ground truth):
\"\"\"{source[:12000]}\"\"\"
{sample_block}
NOTES:
\"\"\"{_notes_excerpt(notes, 20000)}\"\"\""""

    data = safe_json(
        call_model(prompt, max_tokens=700, model=model, temperature=0.1, json_mode=True)
    )

    # Conservative fallback: if we can't parse the critique, assume revision is
    # needed rather than silently passing.
    if not data or "score" not in data:
        return {
            "score": 5,
            "needs_revision": True,
            "unsupported_claims": [],
            "missing_topics": [],
            "issues": ["Critique could not be parsed; revising to be safe."],
            "strengths": [],
        }

    try:
        score = int(data.get("score", 5))
    except (TypeError, ValueError):
        score = 5
    score = max(1, min(10, score))

    unsupported = data.get("unsupported_claims", []) or []
    missing = data.get("missing_topics", []) or []
    # missing_topics is deliberately NOT a trigger on its own — an LLM critic
    # almost always lists something, which previously forced a revision on
    # nearly every run. Missing topics instead drive corrective re-retrieval
    # inside the revise loop when a revision does happen.
    needs = (
        bool(data.get("needs_revision", False))
        or score < REVISE_THRESHOLD
        or len(unsupported) > 0
    )

    return {
        "score": score,
        "needs_revision": needs,
        "unsupported_claims": unsupported,
        "missing_topics": missing,
        "issues": data.get("issues", []) or [],
        "strengths": data.get("strengths", []) or [],
    }


# ---------------------------------------------------------------------------
# Agent: Revise (prompt builder + streaming)
# ---------------------------------------------------------------------------

def _revise_prompt(notes, critique, mode, plan, fmt, context="", instructions="") -> str:
    issues = critique.get("issues", [])
    missing = critique.get("missing_topics", [])
    unsupported = critique.get("unsupported_claims", [])
    issues_str = "\n".join(f"- {i}" for i in issues) if issues else "- (general polish)"
    missing_str = "\n".join(f"- {m}" for m in missing) if missing else "- (none)"
    unsupported_str = "\n".join(f"- {u}" for u in unsupported) if unsupported else "- (none)"

    context_block = (
        f"\n\nCONTEXT (numbered passages — the ONLY source of truth; cite by number):\n{context[:40000]}"
        if context
        else ""
    )

    return f"""You are the REVISION agent in a notes-generation pipeline.

Improve the NOTES below. Keep the same study mode ("{mode}") and formatting.

REMOVE or CORRECT these unsupported/fabricated claims (NOT in the context):
{unsupported_str}

ADD these missing topics (they ARE in the context):
{missing_str}

Also fix these quality issues:
{issues_str}

Rules: every claim must be grounded in the CONTEXT. Do not invent facts. Keep
existing correct content. {_CITE_RULE}

{_format_instructions(fmt)}
{_instr_block(instructions)}
Return ONLY the full, revised notes — no commentary.
{context_block}

CURRENT NOTES:
\"\"\"{notes[:NOTES_REWRITE_CAP]}\"\"\""""


def revise_notes(notes, critique, mode, plan, fmt, model=None, instructions="", context="", length="medium") -> str:
    return call_model(
        _revise_prompt(notes, critique, mode, plan, fmt, context, instructions),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.4,
    )


def revise_notes_stream(notes, critique, mode, plan, fmt, model=None, instructions="", context="", length="medium"):
    yield from call_model_stream(
        _revise_prompt(notes, critique, mode, plan, fmt, context, instructions),
        max_tokens=_max_tokens(length),
        model=model,
        temperature=0.4,
    )


# ---------------------------------------------------------------------------
# Agent: Rewrite (shorter / longer)
# ---------------------------------------------------------------------------

def rewrite_notes(notes, direction, mode="exam", tone="academic", fmt="bullet", model=None) -> str:
    if direction == "shorter":
        change = "Condense these notes to roughly half the length, keeping only the most important points."
    elif direction == "longer":
        change = "Expand these notes with more detail, examples, and explanation — roughly 1.5x longer."
    else:
        change = "Rewrite these notes to improve clarity while keeping the same length."

    prompt = f"""You are the REWRITING agent. {change}

Keep the "{mode}" study focus and a {tone} tone.

{_format_instructions(fmt)}

Return ONLY the rewritten notes — no commentary.

NOTES:
\"\"\"{notes[:NOTES_REWRITE_CAP]}\"\"\""""

    # Budget scales with input so the model can return the FULL rewritten text
    # (a fixed budget silently truncated long rewrites).
    est_tokens = max(1, len(notes)) // 3
    if direction == "longer":
        budget = min(4500, max(3000, est_tokens * 2))
    else:
        budget = min(3500, max(1800, est_tokens))
    return call_model(prompt, max_tokens=budget, model=model)


# ---------------------------------------------------------------------------
# Inline edit: apply an instruction to a selected passage
# ---------------------------------------------------------------------------

def edit_selection(notes, selection, instruction, model=None) -> str:
    prompt = f"""You are editing study notes. Apply the INSTRUCTION ONLY to the
SELECTED passage; leave the rest of the notes unchanged. Preserve the existing
formatting and any [n] citations. Return the COMPLETE updated notes only — no
commentary.

INSTRUCTION: {instruction or "improve this passage"}

SELECTED PASSAGE:
\"\"\"{selection[:2000]}\"\"\"

FULL NOTES:
\"\"\"{notes[:NOTES_REWRITE_CAP]}\"\"\""""
    # Must be able to return the COMPLETE notes, not just the edited passage.
    budget = min(8000, max(2800, len(notes) // 3))
    return call_model(prompt, max_tokens=budget, model=model, temperature=0.4)


# ---------------------------------------------------------------------------
# Agent: Title (auto-name a session)
# ---------------------------------------------------------------------------

def generate_title(notes, model=None) -> str:
    prompt = f"""Give a short, specific title (3-6 words, Title Case) that names the
topic of these study notes. Respond with ONLY the title — no quotes, no punctuation
at the end.

NOTES:
\"\"\"{notes[:1500]}\"\"\""""
    title = call_model(prompt, max_tokens=30, model=model, temperature=0.1).strip()
    # tidy: single line, strip surrounding quotes
    title = title.splitlines()[0].strip().strip('"').strip("'") if title else ""
    return title[:60]


# ---------------------------------------------------------------------------
# Agent: Quiz
# ---------------------------------------------------------------------------

def _flat(text) -> str:
    """Collapse a value to one clean line (the plain-text wire format is
    line-oriented, so embedded newlines would corrupt parsing)."""
    return " ".join(str(text or "").split())


def _valid_questions(data, n: int) -> list:
    """Validate model-returned quiz JSON. Returns [] if unusable, else a list
    of fully-formed questions (question text, options A-D, valid answer)."""
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list):
        return []
    out = []
    for q in qs:
        if not isinstance(q, dict):
            continue
        text = _flat(q.get("question"))
        opts = q.get("options")
        if isinstance(opts, list) and len(opts) >= 4:
            opts = {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]}
        if not isinstance(opts, dict):
            continue
        norm = {str(k).strip().upper()[:1]: _flat(v) for k, v in opts.items()}
        if not all(norm.get(letter) for letter in "ABCD"):
            continue
        answer = str(q.get("answer") or "").strip().upper()[:1]
        if answer not in "ABCD" or not text:
            continue
        out.append({
            "question": text,
            "options": {letter: norm[letter] for letter in "ABCD"},
            "answer": answer,
            "explanation": _flat(q.get("explanation")),
        })
    return out[:n]


def _render_quiz(questions: list) -> str:
    """Deterministically render validated questions into the plain-text format
    the frontend/exports parse (Q#) / A)-D) / Answer: / Explanation:)."""
    blocks = []
    for i, q in enumerate(questions, 1):
        lines = [f"Q{i}) {q['question']}"]
        lines += [f"{letter}) {q['options'][letter]}" for letter in "ABCD"]
        lines.append(f"Answer: {q['answer']}")
        if q.get("explanation"):
            lines.append(f"Explanation: {q['explanation']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_QUIZ_JSON_SCHEMA = """{"questions": [{"question": "<question text>",
"options": {"A": "<option>", "B": "<option>", "C": "<option>", "D": "<option>"},
"answer": "<A|B|C|D>", "explanation": "<one-sentence explanation>"}]}"""


def generate_quiz(notes, n=5, model=None) -> str:
    """Generate a quiz via structured JSON (validated, then rendered to the
    plain-text wire format deterministically). Models drift from free-form
    text formats constantly; JSON mode + validation makes output reliable.
    Falls back to the legacy plain-text prompt if JSON parsing fails."""
    prompt = f"""You are the QUIZ agent. Create exactly {n} multiple-choice
questions that test understanding of the NOTES.

Return ONLY a JSON object in exactly this shape (no markdown, no commentary):
{_QUIZ_JSON_SCHEMA}

Every question must have exactly four options A-D and one correct answer letter.

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=1600, model=model, temperature=0.3, json_mode=True))
        questions = _valid_questions(data, n)
        if questions:
            return _render_quiz(questions)
    except Exception:  # noqa: BLE001
        pass
    return _generate_quiz_text(notes, n=n, model=model)


def _generate_quiz_text(notes, n=5, model=None) -> str:
    """Legacy plain-text fallback (kept so a JSON hiccup can't break quizzes)."""
    prompt = f"""You are the QUIZ agent. Create exactly {n} multiple-choice
questions that test understanding of the NOTES.

Use EXACTLY this plain-text format for each question (no markdown, no extra text):

Q1) <question text>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <A|B|C|D>
Explanation: <one-sentence explanation>

Leave a blank line between questions. Number them Q1, Q2, ... up to Q{n}.

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    return call_model(prompt, max_tokens=1200, model=model, temperature=0.3)


def verify_quiz(notes, quiz, model=None) -> str:
    """
    Validate the answer key: re-check each marked answer against the NOTES.
    The verifier returns JSON corrections ({"corrections": [{"q": 1,
    "answer": "B", "explanation": "..."}]}) which are applied to the parsed
    quiz and re-rendered — so a chatty verifier can no longer corrupt the
    quiz format. Any failure returns the original quiz unchanged.
    """
    if not quiz or not quiz.strip():
        return quiz

    prompt = f"""You are a QUIZ VERIFIER. For each question in the QUIZ, check whether
the marked answer letter is actually correct according to the NOTES.

Return ONLY a JSON object listing the corrections needed (empty list if all
answers are correct), in exactly this shape:
{{"corrections": [{{"q": <question number>, "answer": "<A|B|C|D>",
"explanation": "<one-sentence corrected explanation>"}}]}}

NOTES:
\"\"\"{_notes_excerpt(notes, 9000)}\"\"\"

QUIZ:
\"\"\"{quiz}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=800, model=model, temperature=0.0, json_mode=True))
    except Exception:  # noqa: BLE001
        return quiz

    corrections = data.get("corrections") if isinstance(data, dict) else None
    if not isinstance(corrections, list) or not corrections:
        return quiz

    parsed = _parse_quiz_text(quiz)
    if not parsed:
        return quiz
    for corr in corrections:
        if not isinstance(corr, dict):
            continue
        try:
            idx = int(corr.get("q")) - 1
        except (TypeError, ValueError):
            continue
        answer = str(corr.get("answer") or "").strip().upper()[:1]
        if 0 <= idx < len(parsed) and answer in "ABCD":
            parsed[idx]["answer"] = answer
            expl = _flat(corr.get("explanation"))
            if expl:
                parsed[idx]["explanation"] = expl
    return _render_quiz(parsed)


def _parse_quiz_text(quiz: str) -> list:
    """Parse the plain-text quiz format back into structured questions
    (mirror of the frontend parser, used to apply verifier corrections)."""
    questions, current = [], None
    for raw_line in (quiz or "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        q_match = re.match(r"^Q?\s*\d+[).:]\s*(.+)$", line, re.IGNORECASE)
        opt_match = re.match(r"^([A-D])[).:]\s*(.+)$", line)
        ans_match = re.match(r"^Answer\s*:?\s*([A-D])", line, re.IGNORECASE)
        exp_match = re.match(r"^Explanation\s*:?\s*(.+)$", line, re.IGNORECASE)
        if q_match and not opt_match:
            if current and current.get("question") and len(current.get("options", {})) == 4:
                questions.append(current)
            current = {"question": q_match.group(1).strip(), "options": {},
                       "answer": "", "explanation": ""}
        elif opt_match and current is not None:
            current["options"][opt_match.group(1).upper()] = opt_match.group(2).strip()
        elif ans_match and current is not None:
            current["answer"] = ans_match.group(1).upper()
        elif exp_match and current is not None:
            current["explanation"] = exp_match.group(1).strip()
    if current and current.get("question") and len(current.get("options", {})) == 4:
        questions.append(current)
    return questions


# ---------------------------------------------------------------------------
# Agent: Flashcards
# ---------------------------------------------------------------------------

def _valid_cards(data, n: int) -> list:
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, list):
        return []
    out = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        front, back = _flat(card.get("front")), _flat(card.get("back"))
        if front and back:
            out.append({"front": front, "back": back})
    return out[:n]


def _render_flashcards(cards: list) -> str:
    blocks = []
    for i, card in enumerate(cards, 1):
        blocks.append(f"CARD {i}\nFront: {card['front']}\nBack: {card['back']}")
    return "\n\n".join(blocks)


def generate_flashcards(notes, n=8, model=None) -> str:
    """Structured-JSON flashcards (validated + deterministically rendered),
    with the legacy plain-text prompt as fallback."""
    prompt = f"""You are the FLASHCARD agent. Create exactly {n} flashcards
from the NOTES.

Return ONLY a JSON object in exactly this shape (no markdown, no commentary):
{{"cards": [{{"front": "<concise question or term>", "back": "<clear, correct answer>"}}]}}

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    try:
        data = safe_json(call_model(
            prompt, max_tokens=1400, model=model, json_mode=True))
        cards = _valid_cards(data, n)
        if cards:
            return _render_flashcards(cards)
    except Exception:  # noqa: BLE001
        pass
    return _generate_flashcards_text(notes, n=n, model=model)


def _generate_flashcards_text(notes, n=8, model=None) -> str:
    """Legacy plain-text fallback."""
    prompt = f"""You are the FLASHCARD agent. Create exactly {n} flashcards
from the NOTES.

Use EXACTLY this plain-text format for each card (no markdown, no extra text):

CARD 1
Front: <concise question or term>
Back: <clear, correct answer>

Leave a blank line between cards. Number them CARD 1 ... CARD {n}.

NOTES:
\"\"\"{_notes_excerpt(notes, 12000)}\"\"\""""

    return call_model(prompt, max_tokens=1200, model=model)


# ---------------------------------------------------------------------------
# Chat: ask questions grounded in the notes
# ---------------------------------------------------------------------------

def chat_about_notes_stream(notes, question, history=None, model=None):
    """Stream a tutor-style answer grounded in the provided notes."""
    history = history or []
    convo = ""
    for turn in history[-6:]:
        role = "Student" if turn.get("role") == "user" else "Tutor"
        convo += f"{role}: {(turn.get('content') or '')[:2000]}\n"

    prompt = f"""You are a helpful study TUTOR. Answer the student's question using
primarily the NOTES below as context. If the notes don't cover it, you may use
general knowledge but say so briefly. Be clear and concise. You may use `$...$`
for math and fenced code blocks.

NOTES:
\"\"\"{_notes_excerpt(notes, 10000)}\"\"\"

Conversation so far:
{convo}
Student: {question}
Tutor:"""

    yield from call_model_stream(prompt, max_tokens=900, model=model)


# ---------------------------------------------------------------------------
# Gatekeeper: accept only academic / study material
# ---------------------------------------------------------------------------

def classify_academic(text, model=None) -> dict:
    prompt = f"""You are a strict gatekeeper for an ACADEMIC study-notes generator.
Decide whether the SOURCE is genuine academic / study material — a school,
college, or exam subject such as the sciences, mathematics, computer science,
engineering, medicine, the humanities, history, social sciences, economics, law,
or languages.

REJECT material that is primarily: celebrity or entertainment trivia, gossip,
sports results, product marketing/advertising, personal or casual content, or
anything not intended for serious study.

Respond with ONLY a JSON object:
{{"academic": true|false, "subject": "<subject or 'n/a'>", "reason": "<one short sentence>"}}

SOURCE:
\"\"\"{text[:4000]}\"\"\""""

    data = safe_json(call_model(prompt, max_tokens=200, model=model, temperature=0.0, json_mode=True))

    # Permissive on parse failure — don't block legitimate content over a glitch.
    if not isinstance(data, dict) or "academic" not in data:
        return {"academic": True, "subject": "n/a", "reason": ""}
    return {
        "academic": bool(data.get("academic", True)),
        "subject": data.get("subject", "n/a") or "n/a",
        "reason": data.get("reason", "") or "",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _emit(type_, step, content="", data=None):
    return {"type": type_, "step": step, "content": content, "data": data}


def run_agent(text, mode, tone, length, fmt, model=None, instructions="",
              include_quiz=True, include_flashcards=True):
    """
    Drive the full pipeline, yielding event dicts as each stage progresses.

    Event types:
      status, plan_done, sources, notes_delta, notes_done, critique_done,
      revise_start, notes_revised, title_done, quiz_done, flashcards_done,
      done, error

    `sources` may be emitted AGAIN during the revise loop when corrective
    re-retrieval adds chunks for missing topics — the payload is always the
    full merged list (a superset of the previous one), so clients can simply
    replace their sources state.
    """
    try:
        # Mechanical packaging agents (gatekeeper, title, quiz, flashcards) run
        # on a cheaper model with its OWN free-tier daily quota, so they don't
        # spend the main model's token budget — writing & critique keep the
        # strong model where quality actually matters.
        helper = helper_model(model)

        # 0. Academic gatekeeper — this tool only handles study material.
        yield _emit("status", "gate", "Checking topic…")
        gate = classify_academic(text, model=helper)
        if not gate.get("academic", True):
            yield _emit(
                "blocked",
                "blocked",
                gate.get("reason") or "This doesn't look like academic study material.",
                gate,
            )
            return

        # Build the retrieval index over the source (RAG)
        retriever = Retriever(text)

        # 1-2. Plan. For very large documents the even sample alone covers too
        # little (a topic on only a few pages can be invisible in it), so first
        # the helper model scans the WHOLE document and builds a topic
        # inventory the planner must cover — 100% coverage at planning time.
        topic_inventory = ""
        if len(text) > DIGEST_DOC_THRESHOLD:
            n_parts = min(
                (len(text) + DIGEST_SEGMENT_CHARS - 1) // DIGEST_SEGMENT_CHARS,
                DIGEST_MAX_SEGMENTS,
            )
            yield _emit("status", "plan", f"Scanning full document ({n_parts} parts)…")
            topic_inventory = digest_document(text, model=helper)

        yield _emit("status", "plan", "Planning outline...")
        plan = plan_outline(
            retriever.sample(24000), mode, tone, length,
            model=model, instructions=instructions, doc_chars=len(text),
            topic_inventory=topic_inventory,
        )
        yield _emit("plan_done", "plan", "", plan)

        active_fmt = fmt or plan.get("suggested_format", "bullet")
        outline = plan.get("outline", []) or ["Overview", "Key Concepts", "Summary"]
        checklist = plan.get("checklist", []) or []

        # Long documents: MAP-REDUCE. Each outline section gets its OWN
        # retrieval across the whole document and is written from its own
        # context — so no part of a large source is left out.
        sectioned = len(text) > SECTION_DOC_THRESHOLD and len(outline) >= 3

        if sectioned:
            # Bound the number of sections to keep model-call volume sane on
            # free-tier providers (the writer still covers the whole doc).
            write_outline = outline[:SECTION_MAX_COUNT]
            # Retrieve per-section context up front; expose the union as sources.
            section_ctx = []
            chunk_map = {}
            for sec in write_outline:
                sec_chunks = retriever.retrieve(f"{sec} — {mode} study notes", k=SECTION_RETRIEVAL_K)
                section_ctx.append((sec, sec_chunks))
                for c in sec_chunks:
                    chunk_map[c["id"]] = c
            all_chunks = [chunk_map[i] for i in sorted(chunk_map)]
            context = _format_context(all_chunks)
            yield _emit("sources", "write", "", all_chunks)

            # 3-4. Write sections CONCURRENTLY (each has independent context),
            # but stream them to the client strictly IN ORDER: every section's
            # deltas go into its own queue; the main generator drains queue 1
            # live while later sections are already being written in the
            # background. Event sequence is identical to the sequential path.
            parts = []

            def _push(s):
                parts.append(s)
                return _emit("notes_delta", "write", s)

            def _write_worker(sec, ctx, out_q):
                try:
                    for delta in write_section_stream(
                        sec, ctx, mode, tone, length, active_fmt,
                        checklist=checklist, model=model, instructions=instructions,
                    ):
                        out_q.put(("delta", delta))
                except Exception as exc:  # noqa: BLE001
                    out_q.put(("error", exc))
                finally:
                    out_q.put(("end", None))

            queues = [_queue.Queue() for _ in section_ctx]
            pool = ThreadPoolExecutor(max_workers=max(1, SECTION_CONCURRENCY))
            try:
                for (sec, sec_chunks), out_q in zip(section_ctx, queues):
                    pool.submit(_write_worker, sec, _format_context(sec_chunks), out_q)

                for i, ((sec, _sec_chunks), out_q) in enumerate(zip(section_ctx, queues), 1):
                    yield _emit(
                        "status", "write",
                        f"Writing section {i}/{len(section_ctx)}: {sec}…",
                    )
                    yield _push(f"**{sec}:**\n")
                    while True:
                        kind, payload = out_q.get()
                        if kind == "delta":
                            yield _push(payload)
                        elif kind == "error":
                            raise payload
                        else:  # "end"
                            break
                    yield _push("\n\n")
            finally:
                # If the client disconnects mid-stream, cancel sections that
                # haven't started; running ones finish into abandoned queues.
                pool.shutdown(wait=False, cancel_futures=True)

            notes = "".join(parts).strip()
            yield _emit("notes_done", "write", notes)
        else:
            # Small sources: single-pass write over one retrieval (fast path).
            # Pull a length-scaled slice so "long" actually has enough source
            # material to hit its word target.
            query = " ".join(outline + checklist + [mode])
            single_k = SINGLE_PASS_K.get((length or "medium").lower(), 14)
            chunks = retriever.retrieve(query, k=single_k)
            chunk_map = {c["id"]: c for c in chunks}
            context = _format_context(chunks)
            yield _emit("sources", "write", "", chunks)

            yield _emit("status", "write", "Writing notes...")
            parts = []
            for delta in write_notes_stream(
                context, mode, tone, length, active_fmt, plan, model=model, instructions=instructions
            ):
                parts.append(delta)
                yield _emit("notes_delta", "write", delta)
            notes = "".join(parts).strip()
            yield _emit("notes_done", "write", notes)

        # Critique: FAITHFULNESS is judged against the CONTEXT the writer was
        # actually given (previously it was judged against an unrelated even
        # sample, which produced false "unsupported claim" flags). COVERAGE is
        # still judged against a breadth sample of the whole document. Revise
        # with a bigger budget when the notes are sectioned.
        doc_sample = retriever.sample(8000)
        revise_length = "xl" if sectioned else length

        def _crit_msg(c):
            sc = c.get("score", 0)
            uns = len(c.get("unsupported_claims", []))
            if c.get("needs_revision"):
                extra = f", {uns} unsupported claim(s)" if uns else ""
                return f"Quality {sc}/10{extra} — revising ✍"
            return f"Quality {sc}/10 — faithful, no revision needed ✓"

        # 5-6. Critique (grounded against the writer's CONTEXT for faithfulness,
        # plus a breadth sample for coverage)
        yield _emit("status", "critique", "Checking faithfulness & coverage...")
        critique = critique_notes(
            notes, plan, mode, source=context, model=model, doc_sample=doc_sample
        )
        yield _emit("critique_done", "critique", _crit_msg(critique), critique)

        best_notes = notes
        best_critique = critique

        # 7. Iterative revise loop: (corrective re-retrieval) → revise →
        # re-critique, keep the best version.
        rounds = 0
        # Safety: a full-rewrite revision must see the WHOLE notes. If they
        # exceed the rewrite cap, revising would silently drop the tail —
        # keep the draft (citation cleanup below still runs).
        if len(notes) > NOTES_REWRITE_CAP and critique.get("needs_revision"):
            critique["needs_revision"] = False
            yield _emit(
                "status", "revise",
                "Notes are too long for a safe full revision — keeping the draft.",
            )
        while critique.get("needs_revision") and rounds < MAX_REVISION_ROUNDS:
            rounds += 1

            # Corrective re-retrieval (CRAG): the reviser can only add missing
            # topics if the context actually CONTAINS them. Query the retriever
            # with each missing topic and merge any new chunks into the context
            # before revising; the frontend gets the merged sources list.
            missing = critique.get("missing_topics") or []
            if missing:
                added = False
                for topic in missing[:CORRECTIVE_TOPICS_MAX]:
                    try:
                        extra = retriever.retrieve(str(topic), k=CORRECTIVE_K_PER_TOPIC)
                    except Exception:  # noqa: BLE001
                        continue
                    for c in extra:
                        if c["id"] not in chunk_map:
                            chunk_map[c["id"]] = c
                            added = True
                if added:
                    merged = [chunk_map[i] for i in sorted(chunk_map)]
                    context = _format_context(merged)
                    yield _emit("sources", "revise", "", merged)
                    yield _emit(
                        "status", "revise",
                        f"Retrieved extra context for {min(len(missing), CORRECTIVE_TOPICS_MAX)} missing topic(s)…",
                    )

            yield _emit("status", "revise", f"Revising (round {rounds}/{MAX_REVISION_ROUNDS})...")
            yield _emit("revise_start", "revise", "")
            parts = []
            for delta in revise_notes_stream(
                notes, critique, mode, plan, active_fmt, model=model,
                instructions=instructions, context=context, length=revise_length,
            ):
                parts.append(delta)
                yield _emit("notes_delta", "revise", delta)
            notes = "".join(parts).strip()
            yield _emit("notes_revised", "revise", notes)

            # Re-critique the revised notes (against the possibly augmented context).
            yield _emit("status", "critique", f"Re-checking (round {rounds})...")
            critique = critique_notes(
                notes, plan, mode, source=context, model=model, doc_sample=doc_sample
            )
            yield _emit("critique_done", "critique", _crit_msg(critique), critique)

            if critique.get("score", 0) >= best_critique.get("score", 0):
                best_notes, best_critique = notes, critique

        if rounds == 0:
            yield _emit("status", "revise", "No revision needed ✓")

        # A later revision can score lower — fall back to the best version seen.
        if best_notes != notes:
            notes, critique = best_notes, best_critique
            yield _emit("notes_revised", "revise", notes)
            yield _emit("critique_done", "critique", _crit_msg(critique), critique)

        # Citation verification (deterministic, no model call): drop any [n]
        # citation that doesn't point at a chunk the model was actually shown.
        cleaned = enforce_citations(notes, chunk_map.keys())
        if cleaned != notes:
            notes = cleaned
            yield _emit("notes_revised", "revise", notes)

        # Auto-title (best effort)
        try:
            title = generate_title(notes, model=helper)
            if title:
                yield _emit("title_done", "title", title)
        except Exception:  # noqa: BLE001
            pass

        # 8-9. Quiz (generate, then verify the answer key against the notes).
        # Skipped by default in the app — the user generates these on demand
        # from the Learn sidebar via /api/quiz and /api/flashcards.
        if include_quiz:
            yield _emit("status", "quiz", "Generating quiz...")
            quiz = generate_quiz(notes, n=5, model=helper)
            quiz = verify_quiz(notes, quiz, model=helper)
            yield _emit("quiz_done", "quiz", quiz)

        # 10-11. Flashcards
        if include_flashcards:
            yield _emit("status", "flashcards", "Creating flashcards...")
            cards = generate_flashcards(notes, n=8, model=helper)
            yield _emit("flashcards_done", "flashcards", cards)

        # 12. Done
        yield _emit("done", "complete", "All done!")

    except Exception as exc:  # noqa: BLE001
        yield _emit("error", "error", f"Pipeline error: {exc}")