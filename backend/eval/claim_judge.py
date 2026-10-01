"""Claim-level judging for the long-document eval.

The judge never just says "supported". Every verdict carries a word-for-word
quote, and Python checks the quote is really there before the verdict counts:

- per CLAIM in the notes: does the SOURCE state it, and do the passages it
  CITES state it? Quotes are checked against the source and against the
  cited passages respectively.
- per labelled FACT: do the notes state it? The quote is checked against
  the notes.
- per planted CONTRADICTION: do the notes flag the conflict? Both values must
  also appear in the notes.
- per QUESTION-BANK item: do the notes answer a question the source only
  asks? The quote is checked against the notes.

A verdict whose quote cannot be found is "unverifiable": counted and
reported, never silently turned into a pass or a fail.

Claims are extracted here, not with the pipeline's own _claim_lines: the
eval must not share its definition of a claim with the code it measures.
"""

import re

from models import safe_json
from eval.judge_call import JudgeUnavailableError, call_judge

_CITE_RE = re.compile(r"\[(\d+)\]")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_RULE_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_ALL_BOLD_RE = re.compile(r"^\*\*[^*]+\*\*:?$")
_BULLET = " \t-*•>0123456789.)"
MIN_CLAIM_WORDS = 4
MIN_QUOTE_WORDS = 5


def _plain(text) -> str:
    """Lower-cased, whitespace-collapsed, without markdown emphasis or [n]."""
    t = str(text or "").lower().replace("**", "").replace("`", "")
    t = _CITE_RE.sub(" ", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    return " ".join(t.split())


def quote_found(quote, haystack_plain: str, min_words: int = MIN_QUOTE_WORDS) -> bool:
    """Is `quote` (min_words or more) really in the already-_plain haystack?"""
    q = _plain(quote).strip(" \"'.…;:,")
    if len(q.split()) < min_words:
        return False
    return q in haystack_plain


def extract_claims(notes):
    """[{"n", "line", "text", "ids"}] for every line that asserts something."""
    claims, in_code = [], False
    for i, raw in enumerate((notes or "").split("\n")):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            continue
        if _HEADING_RE.match(line) or _RULE_RE.match(line) or _TABLE_RULE_RE.match(line):
            continue
        body = line.lstrip(_BULLET).strip()
        if _ALL_BOLD_RE.match(body):
            continue
        text = " ".join(_CITE_RE.sub(" ", body).replace("**", "").split())
        if len(text.split()) < MIN_CLAIM_WORDS:
            continue
        ids = list(dict.fromkeys(int(m) for m in _CITE_RE.findall(line)))
        claims.append({"n": len(claims) + 1, "line": i, "text": text, "ids": ids})
    return claims


def _json_call(prompt, judge_model, max_tokens, attempts=2):
    """A judge call that returns a dict, retried once; raises if never usable."""
    last = None
    for _ in range(attempts):
        try:
            data = safe_json(call_judge(prompt, judge_model, max_tokens=max_tokens))
        except JudgeUnavailableError as exc:
            last = exc
            continue
        if data:
            return data
        last = JudgeUnavailableError("judge reply was not JSON")
    raise last or JudgeUnavailableError("judge failed")


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

CLAIM_CLASSES = ("supported", "unsupported", "miscited", "unverifiable", "unjudged")


def _claims_prompt(batch, source, chunk_map):
    blocks = []
    for k, c in enumerate(batch, 1):
        if c["ids"]:
            cited = "\n".join(
                f"    [{i}] " + ((chunk_map.get(i) or {}).get("text")
                                 or "(no passage with this number exists)")
                for i in c["ids"])
            blocks.append(f"CLAIM {k}: {c['text']}\n  CITED PASSAGES:\n{cited}")
        else:
            blocks.append(f"CLAIM {k}: {c['text']}\n  CITED PASSAGES: (none)")
    return f"""You are an independent grader checking study notes against the document
they were written from. Use ONLY the SOURCE below - never outside knowledge.

For EVERY claim return:
- "in_source": true only if the SOURCE states or directly entails everything
  the claim asserts. Paraphrase is fine. A source that is merely ABOUT the
  topic is not support. A claim that answers a question the source only ASKS
  is not supported.
- "source_quote": copy, word for word, the one SOURCE sentence that comes
  closest to the claim (whether or not it supports it). No ellipses.
- "cited_backs": for a claim with cited passages, true only if those passages
  themselves state it; false if they do not; null if it cites nothing.
- "cited_quote": for a cited claim, copy word for word the sentence from its
  CITED PASSAGES that comes closest to the claim; "" if it cites nothing.

Respond with ONLY JSON:
{{"verdicts": [{{"n": 1, "in_source": true, "source_quote": "...", "cited_backs": null, "cited_quote": ""}}]}}

SOURCE:
\"\"\"{source}\"\"\"

CLAIMS:
{chr(10).join(blocks)}"""


def classify_claim(claim, verdict, source_plain, chunk_map):
    """One claim's class from its verdict, with every quote checked."""
    if not isinstance(verdict, dict):
        return "unjudged"
    if not quote_found(verdict.get("source_quote"), source_plain):
        return "unverifiable"
    if verdict.get("in_source") is not True:
        return "unsupported"
    if not claim["ids"]:
        return "supported"
    if any(i not in chunk_map for i in claim["ids"]):
        return "miscited"  # a citation that resolves to nothing backs nothing
    if verdict.get("cited_backs") is False:
        return "miscited"
    cited_plain = _plain(" ".join((chunk_map[i] or {}).get("text", "") for i in claim["ids"]))
    if verdict.get("cited_backs") is True and quote_found(verdict.get("cited_quote"),
                                                          cited_plain):
        return "supported"
    return "unverifiable"


def judge_claims(notes, source, chunk_map, judge_model, batch_size=20):
    """Classify every claim in `notes`. Raises if any batch could not be judged."""
    claims = extract_claims(notes)
    source_plain = _plain(source)
    results = []
    for start in range(0, len(claims), batch_size):
        batch = claims[start:start + batch_size]
        data = _json_call(_claims_prompt(batch, source, chunk_map), judge_model,
                          max_tokens=300 + 160 * len(batch))
        verdicts = {}
        for v in data.get("verdicts") or []:
            try:
                verdicts.setdefault(int(v.get("n")), v)
            except (TypeError, ValueError, AttributeError):
                continue
        for k, c in enumerate(batch, 1):
            results.append({**c, "class": classify_claim(c, verdicts.get(k), source_plain,
                                                         chunk_map)})
    return results


def claim_metrics(results):
    counts = {k: 0 for k in CLAIM_CLASSES}
    for r in results:
        counts[r["class"]] += 1
    decided = counts["supported"] + counts["unsupported"] + counts["miscited"]
    cited_decided = sum(1 for r in results if r["ids"]
                        and r["class"] in ("supported", "unsupported", "miscited"))
    cited_miscited = sum(1 for r in results if r["ids"] and r["class"] == "miscited")
    total = len(results)
    return {
        "claims_total": total,
        "claims_cited": sum(1 for r in results if r["ids"]),
        "counts": counts,
        "claim_precision": (counts["supported"] / decided) if decided else None,
        "unsupported_rate": (counts["unsupported"] / decided) if decided else None,
        "miscitation_rate": (cited_miscited / cited_decided) if cited_decided else None,
        "unverifiable_rate": (counts["unverifiable"] / total) if total else None,
    }


# ---------------------------------------------------------------------------
# Facts, contradictions, question bank (all judged against the NOTES)
# ---------------------------------------------------------------------------

def _items_by_id(data, key):
    out = {}
    for v in data.get(key) or []:
        if isinstance(v, dict) and v.get("id") is not None:
            out.setdefault(str(v["id"]), v)
    return out


def judge_facts(notes, labels, judge_model):
    facts = labels.get("facts") or []
    listing = "\n".join(f"{f['id']}: {f['fact']}" for f in facts)
    prompt = f"""You are an independent grader. For each FACT, decide whether the NOTES
state it correctly. A fact the notes contradict, or only gesture at, is not
covered.

For every fact return "covered" (true/false) and "notes_quote": the sentence
from the NOTES, copied word for word, that states it ("" if not covered).

Respond with ONLY JSON:
{{"facts": [{{"id": "F01", "covered": true, "notes_quote": "..."}}]}}

FACTS:
{listing}

NOTES:
\"\"\"{notes}\"\"\""""
    data = _json_call(prompt, judge_model, max_tokens=400 + 90 * len(facts))
    got = _items_by_id(data, "facts")
    notes_plain = _plain(notes)
    late = set((labels.get("traps") or {}).get("late_fact", {}).get("fact_ids") or [])
    per_fact, unverified = {}, 0
    for f in facts:
        v = got.get(f["id"]) or {}
        covered = v.get("covered") is True
        ok = covered and quote_found(v.get("notes_quote"), notes_plain, min_words=3)
        if covered and not ok:
            unverified += 1
        per_fact[f["id"]] = ok
    late_ids = [i for i in per_fact if i in late]
    return {
        "fact_recall": (sum(per_fact.values()) / len(per_fact)) if per_fact else None,
        "late_fact_recall": (sum(per_fact[i] for i in late_ids) / len(late_ids))
        if late_ids else None,
        "facts_covered": sum(per_fact.values()),
        "facts_total": len(per_fact),
        "fact_claims_unverified": unverified,
        "per_fact": per_fact,
    }


CONTRADICTION_STATUSES = ("flagged", "picked_one", "both_unflagged", "omitted")


def judge_contradictions(notes, labels, judge_model):
    items = (labels.get("traps") or {}).get("contradictions") or []
    if not items:
        return {"contradictions_total": 0, "contradictions_caught": 0,
                "contradictions_caught_rate": None, "statuses": {}}
    listing = "\n".join(
        f"{c['id']}: about {c['about']} - the source gives \"{c['value_a']}\" in one "
        f"place and \"{c['value_b']}\" in another." for c in items)
    prompt = f"""The SOURCE document these NOTES were written from contradicts itself
on each point listed below. For each, say how the NOTES handle it:
- "flagged": the notes give both values AND say they disagree.
- "picked_one": the notes give only one of the values.
- "both_unflagged": the notes give both values in different places without
  saying they disagree.
- "omitted": the notes mention neither value.
Return "notes_quote": the sentence from the NOTES, word for word, that best
shows your answer ("" for omitted).

Respond with ONLY JSON:
{{"contradictions": [{{"id": "C1", "status": "flagged", "notes_quote": "..."}}]}}

POINTS:
{listing}

NOTES:
\"\"\"{notes}\"\"\""""
    data = _json_call(prompt, judge_model, max_tokens=300 + 120 * len(items))
    got = _items_by_id(data, "contradictions")
    notes_plain = _plain(notes)
    statuses, caught = {}, 0
    for c in items:
        v = got.get(c["id"]) or {}
        status = str(v.get("status") or "").strip().lower()
        if status not in CONTRADICTION_STATUSES:
            status = "unjudged"
        both_present = (_plain(c["value_a"]) in notes_plain
                        and _plain(c["value_b"]) in notes_plain)
        verified = status == "omitted" or quote_found(v.get("notes_quote"), notes_plain,
                                                      min_words=3)
        if not verified:
            status = "unverifiable"
        if status == "flagged" and both_present:
            caught += 1
        statuses[c["id"]] = status
    return {"contradictions_total": len(items), "contradictions_caught": caught,
            "contradictions_caught_rate": caught / len(items), "statuses": statuses}


QB_STATUSES = ("reported_as_task", "answered", "absent")


def judge_question_bank(notes, labels, judge_model):
    qb = (labels.get("traps") or {}).get("question_bank") or {}
    qs = qb.get("questions") or []
    if not qs:
        return {"qb_questions": 0, "qb_violations": 0, "qb_violation_rate": None,
                "statuses": {}}
    listing = "\n".join(f"{q['id']}: \"{q['quote']}\" (topic: {q['topic']})" for q in qs)
    prompt = f"""The SOURCE document contains a list of practice or assessment QUESTIONS
that it asks but does NOT answer. For each question below, say how the NOTES
treat its topic:
- "answered": the notes explain or answer it with substantive content (a
  derivation, steps, a mechanism, an explanation) - whether or not that
  content is correct.
- "reported_as_task": the notes only mention that the document asks it.
- "absent": the notes do not mention it.
Return "notes_quote": the sentence from the NOTES, word for word, that best
shows your answer ("" for absent).

Respond with ONLY JSON:
{{"questions": [{{"id": "Q1", "status": "reported_as_task", "notes_quote": "..."}}]}}

QUESTIONS:
{listing}

NOTES:
\"\"\"{notes}\"\"\""""
    data = _json_call(prompt, judge_model, max_tokens=300 + 110 * len(qs))
    got = _items_by_id(data, "questions")
    notes_plain = _plain(notes)
    statuses, violations = {}, 0
    for q in qs:
        v = got.get(q["id"]) or {}
        status = str(v.get("status") or "").strip().lower()
        if status not in QB_STATUSES:
            status = "unjudged"
        elif status != "absent" and not quote_found(v.get("notes_quote"), notes_plain,
                                                    min_words=3):
            status = "unverifiable"
        if status == "answered":
            violations += 1
        statuses[q["id"]] = status
    return {"qb_questions": len(qs), "qb_violations": violations,
            "qb_violation_rate": violations / len(qs), "statuses": statuses}

