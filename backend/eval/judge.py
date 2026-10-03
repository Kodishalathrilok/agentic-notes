"""
LLM-as-judge scoring for the eval harness.

A separate, strict judge (temperature 0, JSON mode) scores generated notes for
faithfulness / coverage / clarity, and checks quiz answer-key correctness.
The judge only ever sees the SOURCE — never the pipeline's own self-critique —
so it is an independent measurement.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import safe_json  # noqa: E402
from eval.judge_call import call_judge  # noqa: E402


def judge_notes(source: str, notes: str, model=None) -> dict:
    prompt = f"""You are a STRICT, impartial grader. Evaluate the NOTES purely
against the SOURCE. Do not use outside knowledge.

Score 1-10 on each axis:
- "faithfulness": are ALL claims in the notes supported by the source? Penalise
  heavily for anything fabricated, distorted, or not in the source.
- "coverage": do the notes capture the source's important points?
- "clarity": are the notes clear, well-structured, and useful for studying?

Also list the specific unsupported/incorrect statements you find.

Respond with ONLY a JSON object:
{{
  "faithfulness": <int 1-10>,
  "coverage": <int 1-10>,
  "clarity": <int 1-10>,
  "hallucinations": ["specific unsupported claim", "..."]
}}

SOURCE:
\"\"\"{source}\"\"\"

NOTES:
\"\"\"{notes}\"\"\""""

    data = safe_json(call_judge(prompt, model, max_tokens=600))

    def clamp(v):
        """A score, or None when the judge did not return one.

        This used to return 0, which is a VERDICT: on a 1-10 scale it reads as
        notes so bad they score below the floor. A provider hiccup therefore
        became indistinguishable from a catastrophic quality result — and being
        the lowest possible value, it dominated the mean. Observed: one
        unparseable judge reply dragged a baseline of 10.0 down to 7.5 and
        manufactured a +1.25 "lift" for the full pipeline out of nothing.
        None means "not measured", and the caller drops the sample.
        """
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return None

    return {
        "faithfulness": clamp(data.get("faithfulness")),
        "coverage": clamp(data.get("coverage")),
        "clarity": clamp(data.get("clarity")),
        "hallucinations": data.get("hallucinations", []) or [],
    }


def judge_quiz(source: str, notes: str, quiz: str, model=None) -> dict:
    if not quiz or not quiz.strip():
        return {"total": 0, "correct": 0}

    prompt = f"""You are a STRICT grader checking a multiple-choice quiz's ANSWER KEY.
For each question, decide whether the marked "Answer:" letter is actually correct
according to the SOURCE and NOTES.

Respond with ONLY a JSON object:
{{
  "total": <number of questions>,
  "correct": <how many marked answers are actually correct>,
  "wrong": [{{"q": <number>, "given": "<letter>", "correct": "<letter>"}}]
}}

SOURCE:
\"\"\"{source}\"\"\"

NOTES:
\"\"\"{notes}\"\"\"

QUIZ:
\"\"\"{quiz}\"\"\""""

    data = safe_json(call_judge(prompt, model, max_tokens=500))

    def num(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    total = num(data.get("total"))
    correct = num(data.get("correct"))
    if total:
        correct = max(0, min(total, correct))
    return {"total": total, "correct": correct, "wrong": data.get("wrong", []) or []}
