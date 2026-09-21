# How the notes pipeline works — one agent at a time

This document explains every "agent" in the Agentic Notes pipeline: what it does, exactly how it does it, what it's good at, and where it can still fail. It's written for a human reading it later (including future-you), not for the model.

The whole pipeline lives in [`backend/agent.py`](../backend/agent.py) and is driven by one function, `run_agent(...)`, which runs the agents in order and streams progress events to the frontend as it goes. Think of it as an assembly line: source text goes in one end and cited notes come out the other, with several checkpoints in between where the work gets checked and improved. Quizzes and flashcards are made afterwards, on demand, from the finished notes.

---

## The big picture first

```
Source text
   |
   v
0. Gatekeeper  --reject-->  "not academic material" (stop here)
   | pass  (also reports the document type: explanatory, question bank, exam, mixed, ...)
   v
   Build a search index over the source (chunk + embed + BM25)
   |
   v
   Digest scan  -->  topic inventory      (only for sources over 60,000 characters)
   |
   v
1. Planner  -->  outline + checklist + difficulty + format
   |
   v
2. Writer  -->  first draft of the notes (streamed live to the screen)
   |              up to 12,000 chars: one pass over a length-scaled retrieval
   |              over 12,000 chars:  coverage windows that partition the source,
   |                                  written two at a time, streamed in order
   v
3. Critique  -->  score /10 + list of unsupported claims + missing topics
   |
   +-- good enough? --> skip ahead
   |
   +-- needs work? --> corrective re-retrieval --> 4. Revise --> back to Critique
   |                   (max 2 rounds; a revision is kept only if it scores strictly higher)
   v
5. Citation validation  -->  drop any [n] that doesn't resolve (deterministic, no model call)
   |
   v
6. Grounding check  -->  is each claim line supported by what it cites? (helper model)
   |
   v
7. Title agent  -->  short name for this session
   |
   v
Done -- the final event carries the coverage record and which provider served the notes

On demand, outside this run (Learn sidebar -> /api/quiz, /api/flashcards):
   Quiz agent (5 MCQs)        Flashcard agent (8 cards)
```

`/api/quiz` generates the quiz and then checks its answer key. `run_agent()` can still produce the quiz (with the answer-key verifier) and flashcards at the end of the run when called with `include_quiz` / `include_flashcards` — the eval harness does this for the quiz — but `/api/generate` defaults both to false and the UI never sets them.

Two important ideas run underneath all of this:

1. **Retrieval-Augmented Generation (RAG).** The model never sees the raw, full document. Instead, the source is chopped into small overlapping passages ("chunks"), and every agent that needs source material asks a **retriever** for the most relevant chunks first. This is what keeps notes grounded in the real document instead of the model just making things up from memory.

2. **Model routing.** Not every agent uses the same "brain." Inside `run_agent()`, the steps where quality really matters (planning, writing, critiquing, revising) use the strong model you picked in the UI. The mechanical steps (the gatekeeper, the digest scan, the per-claim grounding check, naming the session) use a cheaper, faster helper model. This is explained fully in its own section near the end — it's one of the more important design decisions in the whole system.

---

## Agent 0 — The Gatekeeper

**File:** `classify_academic()` in `agent.py`

**Job:** Look at the source text and decide: is this actually study material? The product is called a *study* notes generator, so it refuses to build notes for celebrity gossip, sports scores, ads, or random chit-chat.

**How it works:** It sends the first 4,000 characters of the source to the model with a strict instruction — accept science, maths, computer science, engineering, medicine, humanities, history, economics, law, languages; reject entertainment, marketing, casual content. The model replies with a small JSON verdict: `{"academic": true/false, "subject": "...", "doc_type": "...", "reason": "..."}`. If the answer is "no," the whole pipeline stops immediately and the UI shows the reason — no notes are generated, no tokens are wasted downstream.

**Document type.** The same call also classifies what *kind* of document it is: `explanatory`, `question_bank`, `assignment`, `exam`, `worksheet`, `syllabus`, `mixed` or `other` (anything else becomes `explanatory`). This matters because a question bank is academic but teaches nothing: the writer must not turn "Write a program to implement single inheritance" into "Single inheritance lets one class inherit from another." For task-shaped types the writer prompt gets a rule to report what the document *asks* rather than explain it, and not to introduce terms the source doesn't contain; `mixed` gets a rule to keep explanations and tasks apart; `explanatory` adds nothing, so ordinary lecture material is written exactly as before. The type is also shown as a status line and returned in the final `done` event.

**Strengths:**

- Cheap and fast — only reads 4k characters, uses the small helper model.
- Fails safe: if the model's answer can't be parsed as JSON, the gatekeeper *lets the content through* (as `explanatory`) rather than blocking it. A confused gatekeeper is treated as "not sure, so don't get in the way."
- Runs before the expensive retrieval index is even built, so a rejected topic costs almost nothing.

**Limitations:**

- It only reads the *first* 4,000 characters. A 100-page document that opens with a boring cover page or table of contents, but is genuinely academic later, could theoretically be misjudged — though in practice this hasn't been an issue since real documents establish their subject quickly.
- It's a single model call with no second opinion — an unusual or borderline topic (e.g., a philosophical essay that reads casually) could be judged either way. There's no appeals process, though the safe-parse fallback keeps the blast radius small.
- The document-type rule reaches the writer only. The revise prompt doesn't carry it, so a revision of a question-bank draft relies on the general grounding rules alone.

---

## The Retriever (not an "agent," but everything depends on it)

**Files:** `retriever.py` (public interface) and `retrieval/` folder (`bm25.py`, `semantic.py`, `fusion.py`, `hybrid.py`, `config.py`)

Technically the retriever doesn't call the language model at all — it's a search engine built fresh over each document. But every other agent's quality depends entirely on what it hands them, so it deserves its own explanation.

**How it works, step by step:**

1. **Chunking.** The document is split into passages of roughly 700 characters, cut on word boundaries, with about 120 characters of overlap between neighbors (so a sentence that happens to fall on a chunk boundary isn't cut off with no context). For PDFs the chunker is given each page's character span and cuts chunks *within* a page, so every chunk's page number is a fact rather than a guess from its offset — that is what lets citations name a page. Each chunk gets a stable ID number.

2. **Two independent search methods run over those chunks:**
   - **BM25** — a classic keyword-matching algorithm (the same family of algorithm search engines used for decades). It's good at exact-term matches: if the notes need to talk about "mitochondria," BM25 finds every chunk that literally contains that word.
   - **Semantic search** — the chunks are converted into embeddings (numerical "meaning fingerprints") using Google's Gemini embedding model, and search compares meaning, not just exact words. This finds passages that are relevant even if they use different wording than the query.

3. **Fusion (Reciprocal Rank Fusion, RRF).** The two ranked lists get merged into one combined ranking, so a chunk that both methods agree on ranks higher than one only one method liked. This hybrid approach is deliberately more robust than either method alone — semantic search alone can miss exact technical terms, and keyword search alone can miss a paraphrase.

4. **The result:** a `retrieve(query, k)` call returns chunks in relevance order, each with a small integer ID like `[3]` that the writer can cite. `k` is turned into a *character* budget (`k` x 700), because page-bounded chunks can be much smaller than 700 characters and a fixed count would shrink the evidence. If the whole document fits inside that budget, the whole document is returned (ranked chunks first) — ranking can only lose material when everything fits.

There's also a `sample()` method, which is different from `retrieve()` — instead of relevance-ranking, it takes an *even spread* of chunks across the whole document (like reading every 10th page). The Planner uses it (24,000 characters) for a breadth view of the document, and the Critique uses a smaller one (8,000 characters) to judge coverage.

**Strengths:**

- Hybrid retrieval covers both "the exact term" and "the same idea, different words" — genuinely more reliable than picking one method.
- If Gemini embeddings are unavailable (e.g., no API key, or a transient failure), it logs a clear warning and automatically falls back to BM25-only rather than crashing.
- Retrieval is fully re-buildable per document — there's no shared state or contamination between different users' uploads.

**Limitations:**

- Chunking is by character count on word boundaries (and page boundaries for PDFs) — it doesn't know about headings, paragraphs, or sentences. A chunk can start mid-sentence. (A structure-aware chunker that respects headings/paragraphs is a known improvement that hasn't been built yet.)
- Semantic search depends on an external API call to Gemini for embeddings, which adds latency and a point of failure (mitigated by the BM25 fallback, but fallback is still a downgrade).
- There is currently no reranking step after fusion — the `_process_candidates()` method exists specifically as a hook for a future cross-encoder reranker, but today it's a pass-through (identity function). This means final ranking quality is capped at what BM25 + embeddings + RRF can achieve alone.

---

## Agent 1 — The Planner

**File:** `plan_outline()`

**Job:** Read a broad sample of the document and produce a study plan before any notes are written — like an outline a student would sketch before writing an essay.

**How it works:** It's given an even sample of the document (`retriever.sample(24000)` — up to 24,000 characters spread across the whole thing, not just the start) and asked to return JSON with four fields:

- `outline` — the section headings the notes should have
- `checklist` — specific points that MUST be covered, used later to check nothing important got dropped
- `difficulty` — beginner / intermediate / advanced
- `suggested_format` — bullet points, a numbered list, or paragraphs

Critically, **the outline size scales with document size.** A short document gets 3–5 sections. A "substantial" document (over 12,000 characters, `SECTION_DOC_THRESHOLD`) is told to produce 5–8 sections. A very large document (over 120,000 characters — think a full book chapter) is told to produce 8–12 sections and explicitly instructed: *"cover ALL of its major topics — do not skip parts of the document."* A plan alone can't guarantee coverage, though: on long documents that guarantee now comes from the writer's coverage windows (next section), and the plan mainly supplies the checklist the writer and critique work against.

**The digest scan — how the planner sees 100% of very large documents.** For a 100-page document, the 24,000-character sample covers only about 8% of the text — one passage every ~3 pages. That's fine for spotting big themes, but a topic confined to just 2–3 pages could be completely invisible in the sample, and a topic the planner never names is a topic the writer never retrieves. So for documents over 60,000 characters (`DIGEST_DOC_THRESHOLD`), a preliminary step runs first: the **cheap helper model reads the ENTIRE document**, in segments of 25,000 characters (at most 12 segments, which covers the 300,000-character input cap), and lists up to 10 topics per segment (`digest_document()`). The merged result — a full-coverage "topic inventory" — is handed to the planner with a hard instruction: *your outline MUST cover all major topics in this inventory.* The planner still gets the even sample too (for tone, difficulty, and structure), but coverage decisions now come from a scan that saw every page. Because the scan runs on the helper model, its token cost comes out of a separate free-tier quota, not the main model's budget.

**Strengths:**

- For big documents, the outline is built from a 100% scan of the text, not an 8% sample — a topic that appears on only two pages still makes it into the inventory, and therefore into the outline and checklist.
- Seeing an even sample (not just the beginning) means the plan reflects the whole document's shape, not just its introduction.
- The checklist becomes a real accountability tool later — the Critique agent explicitly checks whether every checklist item made it into the final notes.
- Fails soft everywhere: a failed digest segment is skipped (partial inventory still beats none), and there's a sensible hardcoded outline fallback (`Overview / Key Concepts / Important Details / Summary`) if the model's JSON can't be parsed — a single bad response never crashes the whole pipeline.

**Limitations:**

- The digest adds one helper-model call per 25,000 characters (up to 12 for a maximum-size upload), which adds some seconds of latency and spends helper-quota tokens before writing even starts — the price of genuine full coverage.
- The inventory's quality depends on the helper model's summarization: it reads everything, but if it describes a topic too vaguely, the planner may group it away rather than giving it a proper section. The inventory is also cut to 8,000 characters in the planner prompt.
- On the long-document path the outline does not structure the notes. Coverage windows are defined by position in the document, each window writes its own topical headings, and each is handed only the first 6 checklist items (the same 6 for every window). The plan's main downstream effect there is on the critique's coverage checklist.
- The plan is made once and never revisited. If the Critique step later finds a big gap, the fix happens through targeted revision, not by re-running the planner with new information.

---

## Agent 2 — The Writer (and its "long document" mode)

**Files:** `write_notes_stream()` for small sources; `_document_windows()`, `_window_context()` and `write_section_stream()` for long ones — orchestrated in `run_agent()`

**Job:** Turn source passages into actual, readable study notes — the part the user is staring at as it streams onto the page.

**Two different modes, chosen automatically by source size alone (`SECTION_DOC_THRESHOLD` = 12,000 characters):**

**A. Single pass — sources up to 12,000 characters.** One retrieval call, using the outline + checklist + study mode as the query, with a length-scaled budget (`SINGLE_PASS_K`: 8 / 14 / 24 chunk-equivalents for short / medium / long, i.e. 5,600 / 9,800 / 16,800 characters). When the whole source fits in that budget the writer simply gets the whole source. The chunks are put back into document order and the writer produces the complete notes in one streamed pass, following the outline, respecting the target word count, and citing sources like `[2][5]` as it goes.

**B. Coverage windows — sources over 12,000 characters.** Retrieval alone used to decide what a long document's writer saw, which capped it at 8 sections x 6 chunks = 48 chunks however big the document was (measured: about 9% of a 100-page source). Now the document is **partitioned**:

- `_document_windows()` splits the chunks into contiguous, document-ordered windows. Every chunk is in exactly one window, and every window is written, so covering the document is arithmetic rather than a retrieval outcome.
- The number of windows is bounded (`COVERAGE_MAX_WINDOWS` = 12 by default), so a bigger document gets *bigger* windows rather than more calls. Each window targets the larger of 6,000 characters (`COVERAGE_WINDOW_CHARS`) and one-twelfth of the document, and is hard-capped so its own passages plus supplements fit the writer's 40,000-character context (`WRITER_CONTEXT_CHARS`). If that cap is hit, more windows are created — nothing is sliced off the prompt.
- `_window_context()` adds up to 3 retrieved passages from elsewhere in the document (`COVERAGE_SUPPLEMENT_K`) — a definition introduced earlier, say. Supplements never displace the window's own passages, and if retrieval fails the window still has its own chunks.
- Windows are written concurrently (`SECTION_CONCURRENCY` = 2) but streamed to the screen strictly in order, with a status like *"Writing section 3/8: Pages 12–17…"*. The writer is told to give each part its own topical headings and never to mention page numbers or narrate its process.
- A window that produces nothing is retried once. A window that fails after streaming some text is not retried (that would duplicate text the user already has).

**The coverage record.** Every run ends with a durable record — `complete`, `total_pages`, `processed_pages`, `failed_pages`, `failed_windows` (each with a non-secret reason slug such as `interrupted`, `max_tokens` or `empty_output`) — sent in the final `done` event. If any window failed, the notes themselves get an **"Incomplete coverage"** banner naming the missing pages, added as the very last edit so it travels with the text into history, export and sharing. If every window fails, the run errors instead of returning empty notes.

**Cut-off streams.** If a provider stream fails part-way, or stops because it hit the token cap, the model layer raises `IncompleteStreamError` rather than ending quietly. A cut-off window is recorded as a failed window (so the banner appears); a cut-off single-pass draft is kept, because a partial draft beats none, but it is marked incomplete in the same way.

Either way, both modes get the same core rules: follow the plan, hit the target length, use the requested format (bullets/numbered/paragraphs), match the mode (exam/revision/deep study/summary) and tone (academic/formal/casual/simple), apply the document-type rule, treat the context as the only source of truth, and **cite every claim** to a passage number.

**Strengths:**

- On long documents, every part of the source gets written about — the last pages get the same attention as the first, and the claim is checkable from the coverage record rather than taken on trust.
- A failure is visible: a lost window becomes a named gap in the notes, not a silently shorter document.
- Streaming means the user sees progress immediately instead of staring at a blank screen for 30+ seconds.

**Limitations:**

- Each window is written in isolation from the others, so the writer can't avoid repeating an idea that spans two windows — it never sees what it wrote in the previous one.
- The length target is per window on the long path (e.g. 220–320 words per window for "medium"), so total length grows with the number of windows rather than matching the single-pass targets. Long notes that end up over 30,000 characters then skip the revise loop entirely (see Revise).
- Citations are written by the model; checking them is the job of the two steps after the revise loop (citation validation and the grounding check).

---

## Agent 3 — Critique

**File:** `critique_notes()`

**Job:** Be the "second pair of eyes" that checks the draft against the real source, before the user ever sees a claim that isn't actually true.

**How it works:** It judges two things against two different inputs, on purpose:

1. **Faithfulness** — is every claim backed by the **context the writer was actually given** (the numbered passages)? That context is the ground truth for what the notes were allowed to say. Anything invented, exaggerated, or distorted gets listed as an `unsupported_claim`. (It used to be judged against an unrelated even sample, which produced false "unsupported" flags.)
2. **Coverage** — is anything from the checklist, or from an **even breadth sample of the whole document** (`retriever.sample(8000)`), missing? Gaps get listed as `missing_topics`. The prompt tells the model not to use this sample for faithfulness.
3. **Quality** — is it clear, well organized, and actually useful for studying?

The notes are passed as an even sample across their whole length (up to 20,000 characters, anchored to the end), so a long draft's tail isn't invisible. It returns a score from 1–10, plus the specific lists — not just a number, but a **reason** the revision agent can act on directly.

**On long sources the critique sees what the notes cite.** When the writer's context is over 12,000 characters and the notes cite passages, the critique is no longer shown the head of that context (on the long-document path that hid every later page, so true claims about them were flagged as unsupported). It is shown the whole passages cited in the notes excerpt it reads, never cut mid-passage, up to 40,000 characters (`CRITIQUE_CONTEXT_CHARS`), and the prompt names any cited ids left out for budget. Code then enforces what the prompt asks: a flag on a claim whose cited passages were all withheld is dropped, and a flag on an *uncited* claim is not acted on here but listed as `deferred_to_grounding`, because its evidence could be anywhere in the document and the grounding step checks it against passages retrieved for it. The result also carries `evidence_shown` / `evidence_cited`, so a partial view is visible instead of silent. If the notes cite nothing, the first 12,000 characters are still what it sees.

**When revision is triggered:** the model's own `needs_revision` flag, a score below 8, or *any* unsupported claim. Missing topics alone do **not** force a revision — an LLM critic almost always lists something — they feed corrective re-retrieval when a revision happens anyway.

**Strengths:**

- The fallback behavior is deliberately paranoid: if the critique's JSON can't be parsed for any reason, the code doesn't assume the notes are fine — it defaults to `needs_revision: true`. Silence is treated as "not verified," never as "verified."
- Judging faithfulness against the writer's own context means a claim is only flagged when the evidence the writer had doesn't support it.
- Separating "faithfulness" from "coverage" from "quality" gives the revision agent a precise target instead of a vague "make it better."

**Limitations:**

- Cited passages beyond the 40,000-character budget are named and skipped, not judged, and only citations inside the 20,000-character notes excerpt count.
- The model's raw `needs_revision` flag or a low score still triggers a revision even when every flagged claim was dropped or deferred, so a round can be wasted. It can't make the notes worse: a revision is only kept if it scores strictly higher.
- It's still one model's judgment. There's no independent second critique or human-in-the-loop check — the per-claim grounding step afterwards is the backstop.

---

## Agent 4 — Revise (the self-correction loop)

**File:** `revise_notes_stream()`, driven by the loop in `run_agent()`

**Job:** Take the critique's specific complaints and actually fix them, without throwing away what was already correct.

**How it works:** The revision prompt is built directly from the critique's structured output — it lists the unsupported claims to remove/correct, the missing topics to add, and any other quality issues. It's given the writer's context (up to 40,000 characters, `REVISE_CONTEXT_CHARS`), so anything it adds is still grounded in real passages. When that context is larger, it gets the whole passages the notes cite plus the ones corrective re-retrieval added that round (those admitted first) within the same 40,000 characters, rather than the head of the context; cited passages that don't fit are named and the reviser is told to keep claims citing them as they are.

**Corrective re-retrieval.** Before each round, if the critique reported missing topics, the retriever is queried once per topic (up to 4 topics, 3 chunks each) and any new passages are merged into the context — the reviser can only add a topic it is actually shown. The merged source list is re-sent to the UI.

**Guards around the loop:**

- **Rewrite cap.** A revision rewrites the whole notes, so it must see the whole notes. If the draft is longer than `NOTES_REWRITE_CAP` (30,000 characters), revision is skipped with a status message rather than run on a truncated copy that would silently delete the tail.
- **Failed revisions are discarded.** A revision that comes back empty, errors, or is cut off (`IncompleteStreamError`) never replaces the notes: the previous version is re-sent and the loop stops.
- **Bounded.** At most `MAX_REVISION_ROUNDS = 2`, each followed by a fresh critique.
- **Best version wins — strictly.** A revision replaces the kept version only if its critique score is *strictly higher*; a tie doesn't earn the swap. At the end, if the last revision scored lower than an earlier version, the earlier one is restored.

**Strengths:**

- Grounded in the actual context again, not just "the notes so far" — and corrective re-retrieval means "add the missing topic" comes with the passages to add it from.
- A bad, empty or half-finished revision can't make the output worse than the version before it.
- Bounded at 2 rounds — protects against runaway cost and latency.

**Limitations:**

- Long notes (over 30,000 characters — easy to reach with 12 windows on the "long" setting) skip revision entirely, so on the biggest documents the post-draft checks are citation validation and grounding only.
- On large sources the reviser sees only cited and newly retrieved passages, so it can't add a topic that was neither cited nor retrieved, and cited passages past 40,000 characters are left unseen. It also doesn't get the document-type rule the writer had.
- Two rounds is a real ceiling. A document with several distinct faithfulness problems might need more passes than the budget allows; the tradeoff is intentional (cost and latency vs. thoroughness).

---

## Agent 5 — Citation validation (deterministic, no model call)

**File:** `validate_citations()` (built on `enforce_citations()`)

**Job:** Make sure every `[n]` in the notes points at real evidence.

**How it works:** After the revise loop, every citation marker is checked against `chunk_map` — every passage the writer (or reviser, after corrective re-retrieval) was shown. A citation survives only if that passage exists and, for a document with pages, carries a page inside `1..page_count`. Anything else is stripped. The model never supplies a page number: it names a passage, and the retriever's metadata decides which page that passage came from. `CITATION_DEBUG=1` prints the claim → passage → page trace.

**Strengths:** Pure code, so it can't be talked out of its answer. "Don't invent citations" becomes a guarantee rather than a prompt instruction.

**Limitations:**

- It proves a citation *resolves*, not that the passage says what the claim says — that is the next step's job.
- On the long-document path `chunk_map` is the union of all windows, so a citation to a real passage that a particular window was never shown still passes.

---

## Agent 6 — Grounding check (per claim)

**File:** `verify_claim_support()` (with `_claim_lines()` and `_verify_batch()`)

**Job:** Ask the question citation validation can't: does the cited evidence actually *support* the claim?

**How it works:** Every line that asserts something is treated as a claim — headings, rules, fully bold lines and lines under 5 words are skipped, but an *uncited* line is checked too (selecting only cited lines would let an invented, uncited claim through untouched). Claims go to the helper model in batches of 6, each with its own evidence: the full text of the passages it cites, or, for an uncited claim, the whole retrieved context. When that context is over 14,000 characters (`GROUNDING_SOURCE_CHARS`) it is not truncated to fit: each uncited claim is judged against the passages the retriever finds for it instead (`GROUNDING_RETRIEVE_K` = 4, limited to passages the writer or reviser was shown), and a claim with nothing retrieved is left as written and counted as `skipped_partial_view`, because deleting on a view known to be partial is how true claims about later pages used to be lost. On a synthetic ~200,000-character source with 40 facts, the share of needed evidence shown to the critique and grounding went from 0.05 to 1.00; before the fix only 2 of the 40 true late-document facts survived critique and revision. The model returns a verdict per claim at temperature 0:

- `supported` — kept as written.
- `unsupported` — the line is removed.
- `partial` — the general point is there but it adds specifics the evidence doesn't state; a second small call rewrites it to say only what the evidence supports, and the rewrite is accepted only if it keeps the citation the original had.

The UI gets a status line such as *"Grounding: 2 claim(s) tightened, 1 unsupported claim(s) removed."* It is on by default (`GROUNDING_CHECK=0` turns it off).

**Strengths:**

- Catches the failure citation validation can't: a perfectly resolving citation attached to a claim the page never makes.
- Uncited claims don't escape — they are judged against the source (or, on large sources, the passages retrieved for them) instead of skipped, so a fabricated uncited claim is still removed.
- Verdicts and rewrites are separate calls, because asking for both at once overran the output budget and left most claims unjudged.

**Limitations:**

- It is an LLM judgment by the smaller helper model, not a proof.
- It fails open: if a batch call fails or returns no verdict, those claims are left exactly as written. The count of unjudged claims is logged on the server but not shown to the user.
- The unit is a line. In paragraph format a whole paragraph is one claim, so a single `unsupported` verdict removes the entire paragraph.
- On large sources an uncited claim is only as well judged as retrieval is: if the retriever misses the passage it came from, a true claim can still be judged unsupported and removed. With nothing retrieved at all, the claim ships unjudged.

---

## Agent 7 — Title

**File:** `generate_title()`

**Job:** Name the session automatically (e.g., "Cellular Respiration Basics") so the user doesn't have to.

**How it works:** Reads just the first 1,500 characters of the finished notes and asks for a 3–6 word Title Case name. Runs on the cheap helper model — this task genuinely doesn't need a strong model.

**Strengths:** Fast, cheap, and wrapped in a try/except in the orchestrator — if it fails for any reason, the pipeline just skips having a title rather than failing the whole run.

**Limitations:** Only reads the beginning of the notes, so on rare occasions the title might reflect the first section rather than the document's true overall theme (usually not an issue since the first section is normally representative).

---

## On demand — Quiz and Quiz Verifier

**Files:** `generate_quiz()`, `verify_quiz_detailed()` (and its text-only wrapper `verify_quiz()`); endpoint `/api/quiz` in `main.py`

**Job:** Turn the finished notes into 5 multiple-choice questions, check the answer key against the notes, and say whether it was checked.

**When it runs:** The UI generates the quiz on demand from the Learn sidebar, which calls `/api/quiz`. That endpoint calls `generate_quiz()` and then `verify_quiz_detailed()`, both with the model the user picked. The same check (via `verify_quiz()`) also runs inside `run_agent()` when `include_quiz` is true, which the eval harness uses.

**How it works:**

1. **Generate:** the model is asked for **JSON** (`{"questions": [{"question", "options": {A..D}, "answer", "explanation"}]}`) in JSON mode. Each question is validated — non-empty text, all four options A–D, an answer letter in A–D — and the valid ones are rendered into the plain-text format the frontend and exports parse (`Q1) ... A) ... Answer: ... Explanation: ...`). If the JSON is unusable, a legacy plain-text prompt is used as a fallback. The notes are passed as an even sample across their whole length (`QUIZ_NOTES_CHARS` = 12,000 characters), not just the beginning.
2. **Verify:** a separate pass at temperature 0 sees the **same** notes excerpt the generator wrote from (it used to get a 9,000-character one, so it could "correct" a right answer from a partial view) and returns a verdict for every question: `{"verdicts": [{"q", "correct", "answer", "evidence", "explanation"}]}`, where `evidence` must be copied word for word from the notes. A correction is applied only if the evidence quote is at least 5 words and is found in the notes after normalising both sides (lowercase, `**` and `[n]` citation markers stripped, whitespace collapsed). Otherwise it is rejected: the original answer stays and the question is reported as disputed. Corrections are applied to the parsed quiz, which is then re-rendered, so a chatty verifier can't corrupt the format.
3. **Report:** `/api/quiz` returns `verification` = `{checked, questions, judged, corrected, rejected, disputed}` next to the quiz. `checked` is false when the verifier call failed or returned nothing usable (previously that looked the same as "no corrections needed"); `judged < questions` means only part of the key was checked. If the verifier raises, the endpoint still returns the unchanged quiz with `checked: false`. The quiz panel shows one line from this report ("Answer key checked against your notes", "partly checked (3 of 5)", corrections, disputed questions, or "Answer key not verified").

**Strengths:**

- Structured output plus validation means a model drifting from the text format can't produce a broken quiz on the main path.
- The verifier catches a class of error a single pass makes easily — marking the "obviously right-sounding" option instead of the one the notes support — on the quiz users actually get.
- A wrong verifier can no longer silently overwrite a right answer with nothing to back it: a correction needs a quote that Python can find in the notes, and an unsupported one is reported as disputed instead.
- The user is told what was checked; a failed or partial check is never shown as a clean one.

**Limitations:**

- The evidence check proves the quote exists in the notes, not that it supports the chosen letter; a real quote attached to a wrong correction still gets applied.
- The verifier is the same model that wrote the quiz (the user's pick), so it can share the generator's blind spots; a verdict of "correct" needs no evidence.
- Quizzes reopened from history show no verification status (the report isn't stored).
- The plain-text fallback output is not validated; if both the JSON path and the fallback misbehave, the frontend parser gets whatever the model wrote.
- Verification checks against the *notes*, not the original source, so it can only be as correct as the notes.
- There's no distractor-quality check — a technically-correct quiz could still have implausible wrong answers that make the question too easy.

---

## On demand — Flashcards

**File:** `generate_flashcards()`; endpoint `/api/flashcards`

**Job:** Produce 8 front/back flashcards from the finished notes, ready for the app's spaced-repetition scheduler and CSV export.

**How it works:** Same pattern as the quiz — JSON (`{"cards": [{"front", "back"}]}`), validated (both sides non-empty), rendered deterministically to `CARD 1 / Front: ... / Back: ...`, with a plain-text prompt as fallback. Generated on demand by the UI.

**Strengths:** Simple, fast, and the structured path makes the output reliable to parse downstream.

**Limitations:** Flashcards are never verified against the notes — there's no "flashcard verifier" step. A subtly wrong flashcard back ships without a second check.

---

## The other helper agents (used outside the main pipeline)

A few more single-purpose agents exist in `agent.py` that don't run as part of `run_agent()`, but are called directly by other parts of the app:

- **`rewrite_notes()`** — condenses notes to roughly half length, expands them to ~1.5x, or just improves clarity, depending on what button the user clicks. Used by the "shorter / longer" controls in the UI. Notes over 8,000 characters (`REWRITE_PART_CHARS`) are rewritten in parts split at headings (then paragraphs, then lines), each its own strict call; more than 12 parts (`REWRITE_MAX_PARTS`) is refused with a 422, and any part that fails or is cut off fails the whole rewrite with a 502, leaving the notes unchanged. Parts are joined with a blank line.
- **`edit_selection()`** — applies a specific instruction (like "make this simpler") to just the text a user has highlighted. This is the "inline assistant" feature. The highlight is rendered text (no markdown or citations), so the UI also sends the source lines it covers, read from `data-line` anchors on the rendered lines (`line_start` / `line_end`); the model gets just those lines plus a little surrounding context and returns their replacement, which the server splices back in. Without usable anchors the server locates the text in the notes (exact, then whitespace-normalised) and returns a 422 if it isn't found or appears more than once. The model never regenerates the whole notes, so text outside the selection can't be dropped. A triple-click selection can reach into the next line, which is then sent too; the model is told to leave the rest of the lines unchanged.
- **`chat_about_notes_stream()`** — a tutor-style chat that answers questions grounded primarily in the generated notes (an even sample of up to 10,000 characters), with the last 6 turns of conversation history included for context. If the notes don't cover something, it's explicitly allowed to fall back to general knowledge, but told to say so. If the answer stream is cut off, `/api/chat` appends a visible "cut off — please ask again" notice instead of ending as if the answer were complete.

Both replace the user's notes with the model's answer, so a partial answer would be data loss. They never truncate their input and use `call_model(strict=True)`: a completion the provider says it stopped at the token cap raises `IncompleteStreamError(reason="max_tokens")` and fails over to the next provider instead of being returned as if complete.

These all share the same philosophy as the main pipeline — stay grounded in the notes/source wherever possible — but they're simpler, single-shot calls rather than multi-agent loops, because their tasks don't need planning or self-correction.

---

## Model routing: why not every agent uses the same "brain"

**File:** `helper_model()` in `models.py`, applied throughout `run_agent()`

This is a cross-cutting design decision worth explaining on its own, because it isn't obvious from reading any single agent.

The app runs on **free-tier API keys**, and free tiers have **daily token limits** (not just per-minute ones). A single large document run — especially with up to 12 coverage windows — can use tens of thousands of tokens. If every model call in a run used the same strong model, that model's daily quota would run out quickly, and the *whole app* would stop working until the next day.

The fix: split work across **two separate quotas**.

- **The strong model the user picked** (e.g., a Nemotron on NVIDIA) handles the steps where output quality is decided: **Plan, Write, Critique, Revise.**
- **A cheap helper model** (a small Nemotron via `HELPER_NVIDIA_MODEL`, or Gemini Flash-Lite via `HELPER_GEMINI_MODEL` if the user picked a Gemini model) — which has its **own, separate** daily token budget — handles the mechanical steps inside `run_agent()`: **Gatekeeper, Digest scan, Grounding check, Title**, and the in-pipeline Quiz, Quiz verifier and Flashcards when those are enabled.

Because free-tier daily limits are tracked *per model*, this isn't just a cost optimization — it substantially raises how many documents the app can process per day before hitting a wall. (The NVIDIA split only kicks in once `HELPER_NVIDIA_MODEL` names a small catalog id; unset, the helper steps fall back to the main model. An Ollama model is its own helper.)

Everything *outside* `run_agent()` uses the model the UI sends — the user's pick. That includes the on-demand quiz, its answer-key check and flashcards (`/api/quiz`, `/api/flashcards`), chat, rewrite and inline edit, so today those draw on the main model's quota, not the helper's.

**Strengths:** Free, effective, and doesn't compromise quality where it matters — the tasks on the helper model genuinely don't need a large model to do well.

**Limitations:** If the user's chosen strong model's daily quota is exhausted, the app has no way to automatically switch them to a fresh model — they have to manually pick a different one from the dropdown. There's also a failover chain across providers (the picked provider first, then the others of NVIDIA / Gemini / Ollama) for outright errors, but it doesn't protect against *both* configured cloud providers being out of daily quota on the same day — which is exactly what happened when this was diagnosed. Grounding verdicts come from the smaller model, which is a real quality tradeoff for a check that decides whether a line is deleted.

---

## Honest summary: where this system is strong, and where it still has real gaps

**What it's genuinely good at:**

- Never silently failing — every agent has a sensible fallback (permissive gatekeeper, safe default plan, "assume revision needed" critique, fail-open grounding), and a stream that is cut off is recorded as incomplete rather than passed off as a finished answer. That is one bug class — a partial view or partial output treated as complete — and the same rule now covers the judges and the editors: the critique, reviser and grounding check are shown the evidence for what they judge (cited or retrieved passages, not the head of a long context) and refuse to act on views known to be partial, and rewrite and inline edit never return partial notes.
- Actually checking its own work — the critique/revise loop, then deterministic citation validation, then a per-claim grounding check that removes or tightens lines their evidence doesn't support.
- Covering entire long documents — coverage windows partition the source so every passage is written about, and a durable coverage record plus an "Incomplete coverage" banner says so when a part could not be generated.
- Being honest about faithfulness — the scoring is deliberately strict about penalizing fabricated claims over almost everything else.
- Keeping shared notes private to their links — a share link resolves one row by exact id through a locked-down function owned by a least-privilege role, so the public anon key can no longer list everyone's shared notes; this is tested against a real PostgreSQL running the actual schema.

**What's still genuinely unverified or missing today:**

- **Grounding is an LLM judgment, not a proof.** It runs on the smaller helper model, works line by line (a paragraph is one line), fails open (unjudged claims ship as written, and the user isn't told how many), and on large sources judges an uncited claim against the passages retrieved for it, so a retrieval miss can still remove a true claim.
- **Checks on long documents are coarser than the writing.** The critique and reviser see cited passages up to 40,000 characters, and anything cited beyond that is named and skipped, not judged; notes over 30,000 characters skip revision entirely. A critic's raw `needs_revision` flag or score can still trigger a wasted revision round.
- **No cross-encoder reranking** after the hybrid search — the retrieval quality is capped at what BM25 + embeddings + RRF can do; the code has a clean seam (`_process_candidates()`) for this to be added later, but it isn't built yet.
- **The quiz answer-key check is evidence-gated, not proven.** A correction needs a quote found in the notes, but that shows the quote exists, not that it supports the new letter; the key is checked against the notes, not the source; and quizzes from history show no status. There's also no distractor-quality check and no flashcard verification.
- **Daily free-tier quotas are a real ceiling**, not just a hypothetical — this has already caused a real outage in production, and the two-model routing fix reduces but does not eliminate the risk.

This isn't a finished, perfect system — it's a genuinely well-engineered one with clear, honest edges. Knowing exactly where those edges are is what makes it possible to keep improving it deliberately instead of guessing.
