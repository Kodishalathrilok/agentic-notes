"""
Multi-agent notes pipeline.

run_agent(...) drives the whole pipeline as a synchronous generator that
yields event dicts of the shape: { type, step, content, data }.

Standalone helpers (used by the regenerate / chat / title endpoints):
    generate_quiz, generate_flashcards, rewrite_notes,
    generate_title, chat_about_notes_stream
"""

from models import call_model, call_model_stream, safe_json
from retriever import Retriever

# Quality thresholds for the self-improvement loop
REVISE_THRESHOLD = 8  # revise until score reaches this (1-10)
MAX_REVISION_ROUNDS = 2  # cap revision passes to bound latency


def _format_context(chunks) -> str:
    """Render retrieved chunks as numbered passages the agent can cite."""
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


_CITE_RULE = (
    "Support each point with a citation to the passage number(s) it came from, "
    "in square brackets right after the point, e.g. [1] or [2][5]. Only cite "
    "numbers that appear in the CONTEXT. Do not invent citations."
)

# ---------------------------------------------------------------------------
# Tuning tables
# ---------------------------------------------------------------------------

LENGTH_TARGETS = {
    "short": "150-200 words",
    "medium": "300-400 words",
    "long": "500-700 words",
}

# Output token budget per length (enough to finish without truncation).
LENGTH_MAX_TOKENS = {
    "short": 900,
    "medium": 1600,
    "long": 2800,
}


def _max_tokens(length: str) -> int:
    return LENGTH_MAX_TOKENS.get((length or "medium").lower(), 1600)

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
# Agent: Plan
# ---------------------------------------------------------------------------

def plan_outline(text, mode, tone, length, model=None, instructions="") -> dict:
    prompt = f"""You are the PLANNING agent in a notes-generation pipeline.

Analyse the SOURCE material and produce a study plan.

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
\"\"\"{text[:16000]}\"\"\""""

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
Length target: {LENGTH_TARGETS.get(length.lower(), '300-400 words')}.

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
# Agent: Critique
# ---------------------------------------------------------------------------

def critique_notes(notes, plan, mode, source="", model=None) -> dict:
    """
    Grounded critique: judges the NOTES against the original SOURCE for
    faithfulness (no fabricated claims) and coverage (nothing important missing),
    plus overall quality for the study mode.
    """
    checklist = plan.get("checklist", [])
    checklist_str = "\n".join(f"- {c}" for c in checklist) if checklist else "(none)"

    prompt = f"""You are the CRITIQUE agent in a notes-generation pipeline. Be a
strict, fair reviewer for the "{mode}" study mode.

Compare the NOTES against the SOURCE. Judge THREE things:
1. FAITHFULNESS — does every claim in the notes actually appear in / follow from
   the SOURCE? List any statement that is fabricated, distorted, or unsupported.
2. COVERAGE — are any important points from the SOURCE (or the checklist) missing?
3. QUALITY — clarity, structure, and usefulness for studying.

Checklist that should be covered:
{checklist_str}

Respond with ONLY a JSON object of this exact shape:
{{
  "score": <integer 1-10>,
  "needs_revision": <true|false>,
  "unsupported_claims": ["claim in the notes NOT supported by the source", "..."],
  "missing_topics": ["important source point the notes omitted", "..."],
  "issues": ["other quality problem", "..."],
  "strengths": ["what was done well", "..."]
}}

Scoring: deduct heavily for any unsupported_claims (faithfulness matters most).
A score of {REVISE_THRESHOLD} or above with NO unsupported claims means no
revision is needed.

SOURCE:
\"\"\"{source[:12000]}\"\"\"

NOTES:
\"\"\"{notes[:6000]}\"\"\""""

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
    needs = (
        bool(data.get("needs_revision", False))
        or score < REVISE_THRESHOLD
        or len(unsupported) > 0
        or len(missing) > 0
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
\"\"\"{notes[:7000]}\"\"\""""


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
\"\"\"{notes[:8000]}\"\"\""""

    budget = 3000 if direction == "longer" else 1800
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
\"\"\"{notes[:14000]}\"\"\""""
    return call_model(prompt, max_tokens=2800, model=model, temperature=0.4)


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

def generate_quiz(notes, n=5, model=None) -> str:
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
\"\"\"{notes[:6000]}\"\"\""""

    return call_model(prompt, max_tokens=1200, model=model, temperature=0.3)


def verify_quiz(notes, quiz, model=None) -> str:
    """
    Validate the answer key: re-check each marked answer against the NOTES and
    return a corrected quiz in the SAME format. Falls back to the original quiz
    if the verifier returns something implausibly short.
    """
    if not quiz or not quiz.strip():
        return quiz

    prompt = f"""You are a QUIZ VERIFIER. For each question below, check whether the
marked "Answer:" letter is actually correct according to the NOTES. If an answer
is wrong, fix the Answer letter and update the Explanation. Keep every question
and the EXACT same plain-text format (Q#) / A) B) C) D) / Answer: / Explanation:).

Return ONLY the full corrected quiz — no commentary.

NOTES:
\"\"\"{notes[:6000]}\"\"\"

QUIZ:
\"\"\"{quiz}\"\"\""""

    try:
        result = call_model(prompt, max_tokens=1300, model=model, temperature=0.0).strip()
    except Exception:  # noqa: BLE001
        return quiz

    # Guard against a degenerate verifier response.
    if len(result) < max(40, int(len(quiz) * 0.5)) or "Answer" not in result:
        return quiz
    return result


# ---------------------------------------------------------------------------
# Agent: Flashcards
# ---------------------------------------------------------------------------

def generate_flashcards(notes, n=8, model=None) -> str:
    prompt = f"""You are the FLASHCARD agent. Create exactly {n} flashcards
from the NOTES.

Use EXACTLY this plain-text format for each card (no markdown, no extra text):

CARD 1
Front: <concise question or term>
Back: <clear, correct answer>

Leave a blank line between cards. Number them CARD 1 ... CARD {n}.

NOTES:
\"\"\"{notes[:6000]}\"\"\""""

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
        convo += f"{role}: {turn.get('content', '')}\n"

    prompt = f"""You are a helpful study TUTOR. Answer the student's question using
primarily the NOTES below as context. If the notes don't cover it, you may use
general knowledge but say so briefly. Be clear and concise. You may use `$...$`
for math and fenced code blocks.

NOTES:
\"\"\"{notes[:7000]}\"\"\"

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
    if "academic" not in data:
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


def run_agent(text, mode, tone, length, fmt, model=None, instructions=""):
    """
    Drive the full pipeline, yielding event dicts as each stage progresses.

    Event types:
      status, plan_done, notes_delta, notes_done, critique_done,
      revise_start, notes_revised, title_done, quiz_done, flashcards_done,
      done, error
    """
    try:
        # 0. Academic gatekeeper — this tool only handles study material.
        yield _emit("status", "gate", "Checking topic…")
        gate = classify_academic(text, model=model)
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

        # 1-2. Plan (sees a breadth sample across the whole document)
        yield _emit("status", "plan", "Planning outline...")
        plan = plan_outline(
            retriever.sample(16000), mode, tone, length, model=model, instructions=instructions
        )
        yield _emit("plan_done", "plan", "", plan)

        active_fmt = fmt or plan.get("suggested_format", "bullet")

        # Retrieve the most relevant passages for writing, and expose them as
        # citable sources to the UI.
        query = " ".join(plan.get("outline", []) + plan.get("checklist", []) + [mode])
        chunks = retriever.retrieve(query, k=8)
        context = _format_context(chunks)
        yield _emit("sources", "write", "", chunks)

        # 3-4. Write (streamed, grounded in retrieved context with citations)
        yield _emit("status", "write", "Writing notes...")
        parts = []
        for delta in write_notes_stream(
            context, mode, tone, length, active_fmt, plan, model=model, instructions=instructions
        ):
            parts.append(delta)
            yield _emit("notes_delta", "write", delta)
        notes = "".join(parts).strip()
        yield _emit("notes_done", "write", notes)

        def _crit_msg(c):
            sc = c.get("score", 0)
            uns = len(c.get("unsupported_claims", []))
            if c.get("needs_revision"):
                extra = f", {uns} unsupported claim(s)" if uns else ""
                return f"Quality {sc}/10{extra} — revising ✍"
            return f"Quality {sc}/10 — faithful, no revision needed ✓"

        # 5-6. Critique (grounded against the SOURCE for faithfulness + coverage)
        yield _emit("status", "critique", "Checking faithfulness & coverage...")
        critique = critique_notes(notes, plan, mode, source=text, model=model)
        yield _emit("critique_done", "critique", _crit_msg(critique), critique)

        best_notes = notes
        best_critique = critique

        # 7. Iterative revise loop: revise → re-critique, keep the best version.
        rounds = 0
        while critique.get("needs_revision") and rounds < MAX_REVISION_ROUNDS:
            rounds += 1
            yield _emit("status", "revise", f"Revising (round {rounds}/{MAX_REVISION_ROUNDS})...")
            yield _emit("revise_start", "revise", "")
            parts = []
            for delta in revise_notes_stream(
                notes, critique, mode, plan, active_fmt, model=model,
                instructions=instructions, context=context, length=length,
            ):
                parts.append(delta)
                yield _emit("notes_delta", "revise", delta)
            notes = "".join(parts).strip()
            yield _emit("notes_revised", "revise", notes)

            # Re-critique the revised notes.
            yield _emit("status", "critique", f"Re-checking (round {rounds})...")
            critique = critique_notes(notes, plan, mode, source=text, model=model)
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

        # Auto-title (best effort)
        try:
            title = generate_title(notes, model=model)
            if title:
                yield _emit("title_done", "title", title)
        except Exception:  # noqa: BLE001
            pass

        # 8-9. Quiz (generate, then verify the answer key against the notes)
        yield _emit("status", "quiz", "Generating quiz...")
        quiz = generate_quiz(notes, n=5, model=model)
        quiz = verify_quiz(notes, quiz, model=model)
        yield _emit("quiz_done", "quiz", quiz)

        # 10-11. Flashcards
        yield _emit("status", "flashcards", "Creating flashcards...")
        cards = generate_flashcards(notes, n=8, model=model)
        yield _emit("flashcards_done", "flashcards", cards)

        # 12. Done
        yield _emit("done", "complete", "All done!")

    except Exception as exc:  # noqa: BLE001
        yield _emit("error", "error", f"Pipeline error: {exc}")
