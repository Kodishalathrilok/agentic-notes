"""Long-document fixtures for the claim-level eval (evals/fixtures/<name>/).

Each fixture directory holds a source (source.txt, or source.pdf built from
pages.txt by evals/fixtures/build_pdf.py) and labels.json: hand-written
atomic facts, each with a `quote` that must appear verbatim in the source,
plus planted traps - facts stated only late in the document, an internal
contradiction, and a question-bank section whose questions the source asks
but never answers.

A PDF fixture is read through the app's own /api/extract-pdf handler, so
the eval sees exactly the text and page spans a user's upload produces.
"""

import hashlib
import json
import os
import re

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BACKEND_DIR)
EVALS_DIR = os.path.join(REPO_DIR, "evals")
FIXTURES_DIR = os.path.join(EVALS_DIR, "fixtures")


def normalize(text) -> str:
    """Whitespace-collapsed text: the space every quote check is done in."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fixture_names(root=FIXTURES_DIR):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "labels.json"))
    )


def _extract_pdf(path):
    """(text, page_spans, pages) exactly as /api/extract-pdf returns them."""
    from fastapi.testclient import TestClient
    import main

    app = main.app
    # Authentication is not what is being measured; the parsing is.
    app.dependency_overrides[main._EXTRACT_GUARD] = lambda: None
    try:
        # No `with`: startup hooks (auth config checks) are not needed to parse.
        client = TestClient(app)
        with open(path, "rb") as f:
            resp = client.post("/api/extract-pdf",
                               files={"file": (os.path.basename(path), f.read(),
                                               "application/pdf")})
    finally:
        app.dependency_overrides.pop(main._EXTRACT_GUARD, None)
    if resp.status_code != 200:
        raise RuntimeError(f"/api/extract-pdf rejected {path}: {resp.status_code} {resp.text[:200]}")
    body = resp.json()
    if body.get("truncated"):
        raise RuntimeError(f"{path} was truncated on extraction; the fixture is too long")
    return body["text"], body["page_spans"], body["pages"]


def load_fixture(name, root=FIXTURES_DIR):
    d = os.path.join(root, name)
    with open(os.path.join(d, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    src = labels["source"]
    path = os.path.join(d, src["path"])
    if src["type"] == "pdf":
        text, spans, pages = _extract_pdf(path)
    elif src["type"] == "text":
        with open(path, encoding="utf-8") as f:
            text = f.read()
        spans, pages = None, 0
    else:
        raise ValueError(f"{name}: unknown source type {src['type']!r}")
    return {"id": name, "labels": labels, "text": text, "page_spans": spans,
            "pages": pages, "dir": d}


def fixture_fingerprint(names, root=FIXTURES_DIR) -> str:
    """sha256 over every fixture file, so a baseline recorded on different
    fixtures (or labels) is never compared with this run as if it matched."""
    h = hashlib.sha256()
    for name in sorted(names):
        d = os.path.join(root, name)
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            h.update(f"{name}/{fn}\0".encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def label_problems(fixture):
    """Every way the labels fail to match the source. [] means consistent.

    Run by the test suite (no model needed), so a source edit that breaks a
    label is caught before an eval spends tokens on it.
    """
    labels, text = fixture["labels"], normalize(fixture["text"])
    problems = []
    n = max(1, len(text))

    def where(quote):
        q = normalize(quote)
        if not q:
            return None
        hits = text.count(q)
        if hits == 0:
            problems.append(f"quote not in source: {quote!r}")
            return None
        if hits > 1:
            problems.append(f"quote is ambiguous ({hits} matches): {quote!r}")
        return text.find(q) / n

    facts = labels.get("facts") or []
    ids = [f.get("id") for f in facts]
    if len(set(ids)) != len(ids):
        problems.append("duplicate fact ids")
    positions = {}
    for f in facts:
        if not (f.get("fact") or "").strip():
            problems.append(f"{f.get('id')}: empty fact")
        positions[f.get("id")] = where(f.get("quote"))

    traps = labels.get("traps") or {}
    late = traps.get("late_fact") or {}
    if not late.get("fact_ids"):
        problems.append("no late_fact trap")
    for fid in late.get("fact_ids") or []:
        pos = positions.get(fid)
        if fid not in positions:
            problems.append(f"late_fact names unknown fact {fid}")
        elif pos is not None and pos < float(late.get("min_position", 0.9)):
            problems.append(f"late fact {fid} sits at {pos:.2f}, before "
                            f"{late.get('min_position', 0.9)}")

    contradictions = traps.get("contradictions") or []
    if not contradictions:
        problems.append("no contradiction trap")
    for c in contradictions:
        a, b = where(c.get("quote_a")), where(c.get("quote_b"))
        if normalize(c.get("value_a")) == normalize(c.get("value_b")):
            problems.append(f"{c.get('id')}: the two values are the same")
        if normalize(c.get("value_a")) not in normalize(c.get("quote_a")):
            problems.append(f"{c.get('id')}: value_a not in quote_a")
        if normalize(c.get("value_b")) not in normalize(c.get("quote_b")):
            problems.append(f"{c.get('id')}: value_b not in quote_b")
        if a is not None and b is not None and abs(a - b) < 0.2:
            problems.append(f"{c.get('id')}: the two statements are too close together "
                            f"to test cross-section consistency")

    qb = traps.get("question_bank") or {}
    head = where(qb.get("heading_quote")) if qb.get("heading_quote") else None
    if not qb.get("questions"):
        problems.append("no question_bank trap")
    for q in qb.get("questions") or []:
        pos = where(q.get("quote"))
        if head is not None and pos is not None and pos < head:
            problems.append(f"question {q.get('id')} appears before the question-bank heading")
    return problems
