---
title: Agentic AI Notes Generator
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 📚 Agentic Notes

[![CI](https://github.com/Kodishalathrilok/agentic-notes/actions/workflows/ci.yml/badge.svg)](https://github.com/Kodishalathrilok/agentic-notes/actions/workflows/ci.yml)

**Turns any study source — text, PDF, audio, a photo, a URL or a YouTube video — into study notes where every claim is traced to the page it came from, and claims the source doesn't support are removed by code rather than left to the prompt.**

**Live:** [thrilokkk-agentic-notes.hf.space](https://thrilokkk-agentic-notes.hf.space) · [Space page](https://huggingface.co/spaces/Thrilokkk/agentic-notes) · [3-minute demo script and interview Q&A](DEMO.md)

### Try it without signing up

Generating notes needs an account, but shared notes open for anyone — no sign-in, real output, real citations:

<!-- ============================================================
     PASTE SHARE LINKS HERE (replace the placeholder line below).
     Generate notes in the app, press Share, paste the link.
     ============================================================ -->
- _Example notes: **links not added yet** — see the placeholder comment in this file._

### What makes this different

- **Per-claim grounding.** Every line that asserts something is judged against its own evidence, and removed only when the verdict quotes that evidence and the quote is really there — `verify_claim_support()` in [`backend/agent.py`](backend/agent.py).
- **Citation validation in code.** A `[n]` survives only if it points at a passage the model was actually shown, on a real page. No model call, and it runs again after grounding edits the notes — `validate_citations()`.
- **Coverage windows.** Sources over 12,000 characters are partitioned so every chunk is written about exactly once. The retrieval-only approach this replaced showed the writer about 9% of a 100-page source (measured; see the comment above `_document_windows()`).
- **Offline tests.** 600+ backend tests run with no network and no API keys, because the model calls are replaced by fakes — [`backend/tests/`](backend/tests), 29 files.
- **A claim-level eval harness.** Three long fixtures with hand-labelled facts and planted traps, graded by a judge from a different model family — [`backend/eval/claim_eval.py`](backend/eval/claim_eval.py). Built and tested; **its first full run is still pending**, so no model-graded score is quoted anywhere in this README.

### In more detail

- **The pipeline checks its own work.** A critique agent judges the draft against the passages the writer was given, flags unsupported claims and missing topics, triggers corrective re-retrieval for the missing topics, and revises (up to 2 rounds) — a revision replaces the kept version only if it scores strictly higher.
- **Claims are checked against their sources.** Notes cite retrieved passages inline (`[3]`). A deterministic check drops any citation that doesn't point at a passage the model was shown (or, for PDFs, at a passage with a valid page), and a per-claim grounding pass then asks a helper model whether each claim line is actually supported by the passage it cites. A line is removed only when the verdict quotes that claim's own evidence and the quote is really there; otherwise it is kept and counted as unverified. An over-reaching line is tightened only if the rewrite passes a second check, may keep only citations it already had, and citations are validated again afterwards.
- **Long documents are covered end to end.** Sources over 12,000 characters are split into coverage windows that partition the document — every chunk belongs to exactly one window and every window is written — and sources over 60,000 characters also get a full-document digest scan before planning. The windows are then merged without a model call: one heading per topic, a bullet that repeats an earlier one dropped, sections in page order, and no empty heading or line about "the passages". If a window fails, the notes say so with an "Incomplete coverage" banner naming the pages nothing in the notes cites, instead of passing as complete.
- **The engineering around it:** 600+ backend tests plus frontend unit tests, CI that deploys only when the tests and the build pass, SSRF-guarded URL fetching, share links that can't be used to list other users' notes, request size limits, and per-user rate limiting. The claim-level eval can compare a run against a stored baseline (`--gate`), but no baseline has been recorded with model-graded numbers yet, so today that comparison has nothing to compare against.

## How it works

```mermaid
flowchart LR
    A[Source\ntext · PDF · audio · image · URL · YouTube] --> B[Gatekeeper\nacademic check + doc type]
    A --> C[Hybrid index\nBM25 + embeddings, rank fusion]
    B -->|study material| D[Digest scan\nwhole document, sources over 60k chars]
    C --> D
    B -.->|not study material| X[Stopped\nindex discarded]
    D --> E[Planner\noutline + checklist]
    E --> F[Writer\nsingle pass, or coverage windows over 12k chars]
    F --> G[Critique\nclaims vs. cited passages]
    G -->|needs revision| H[Corrective re-retrieval\n+ Revise]
    H --> G
    G -->|approved or 2 rounds| I[Citation validation\ndeterministic]
    I --> J[Grounding check\nper claim, helper model]
    J -->|lines removed or rewritten| I2[Citation validation\nagain]
    I2 --> K[Title]
    J -->|nothing changed| K
    K --> L[Notes]
    L -.->|on demand| M[Quiz · Flashcards · Tutor chat]
```

1. **Gatekeeper** rejects non-study material and classifies the document type (explanatory, question bank, exam, mixed…); task-shaped documents get a writer rule that forbids turning a task into a statement of fact.
2. **Index**: the source is chunked (page-bounded for PDFs) and indexed for hybrid retrieval. This runs on a background thread *while* the gatekeeper decides, so a rejected document does not wait for an index nobody will use.
3. **Digest** (sources over 60,000 characters): a helper model reads the whole document in 25,000-character segments and builds a topic inventory the planner must cover.
4. **Planner** produces an outline and a checklist of must-cover points.
5. **Writer**: sources up to 12,000 characters are written in one streamed pass from a length-scaled retrieval (the whole source when it fits). Larger sources are split into contiguous coverage windows — at most 12 by default, each targeted at the larger of 6,000 characters and one-twelfth of the source, and capped so a window plus its supplements fits the writer's 40,000-character context (past that cap more windows are added; nothing is truncated). Each window is written from its own chunks plus a few retrieved passages from elsewhere, two at a time, and streamed to the UI in document order.
6. **Critique → revise** loop (max 2 rounds), then **citation validation** and the **grounding check** — with citations validated a second time if grounding removed or rewrote anything — then an auto-generated **title**.

Quiz and flashcards are not part of the notes run: the UI generates them on demand through `/api/quiz` and `/api/flashcards`. Every stage of the notes run streams to the UI in real time over SSE, so you watch the agents plan, write, critique, and revise live.

### Retrieval

The source is chunked (~700-character passages with overlap, cut on word boundaries, and kept within a single page for PDFs) and indexed two ways — **BM25** (lexical) and **semantic embeddings** (Gemini) — and results are merged with reciprocal rank fusion. If embeddings are unavailable (no key, or an API failure) retrieval falls back to BM25 only. There is no reranker after fusion yet. On the single-pass path the writer gets the passages most relevant to the plan; on the windowed path retrieval only adds supplementary evidence — which passages get written about is decided by the window partition, not by ranking. A retrieval benchmark (`backend/eval/`) measures hit-rate against a labeled dataset.

### Reliability details that took real work

- **Structured outputs for quizzes and flashcards.** Both are requested as JSON, validated (four options A–D and a valid answer letter per question; non-empty front and back per card) and rendered to the UI format deterministically, with a plain-text prompt as fallback if the JSON is unusable. The on-demand `/api/quiz` the UI calls then checks the answer key (`verify_quiz_detailed`): the verifier reads the same 12,000-character notes excerpt the generator wrote from (it used to get 9,000), gives a verdict per question with a quote from the notes as evidence, and a correction is applied only if that quote (at least 5 words) is actually found in the notes after normalising markdown bold and `[n]` citations — otherwise the original answer stays and the question is marked disputed. The response carries a `verification` report, and the quiz panel says whether the key was checked, partly checked, corrected or disputed (or "not verified" if the check failed; the quiz is still returned). Known limits: the evidence check proves the quote exists in the notes, not that it supports the chosen letter; the key is checked against the notes, not the original source; and quizzes reopened from history show no status.
- **No tail-loss on long notes.** Agents that read the notes without rewriting them (critique, quiz, flashcards, chat) get an even sample across the whole notes, anchored to the end, instead of the first N characters. The in-pipeline revision refuses to run on notes over 30,000 characters rather than revising a truncated copy.
- **Rewrite and inline edit never return partial notes.** One-click rewrite sends notes over 8,000 characters as parts split at headings (at most 12 parts; longer notes get a 422 asking to rewrite a section at a time), and if any part fails or is cut off, the whole rewrite fails with a 502 and nothing is changed. Inline edit sends only the source lines the highlight covers (the UI reads them from `data-line` anchors on the rendered lines; locating the text is the fallback, with a 422 if it isn't found or isn't unique), and the server splices the new lines back in, so the model never regenerates the whole notes. Both use strict model calls: a completion the provider stopped at the token cap counts as a failure and fails over to the next provider. Neither is sent the sources, so their output is **not re-checked**: a citation the notes did not already carry is removed (`strip_new_citations()`), the response is marked `unverified`, and the notes show an "Edited — not re-verified" badge until the next generation. Known limits: a triple-click selection can reach into the next line, which is then sent too (the model is told to keep it unchanged), a multi-part rewrite joins its parts with a blank line, and an edited claim is not grounded against the source — only its citations are constrained.
- **A cut-off stream is never passed off as complete.** If a provider stream fails part-way, or stops at the token cap, the model layer raises `IncompleteStreamError` instead of returning quietly. A coverage window cut off at the token cap, or ending on a visibly unfinished bullet, is discarded and written again as two halves; a half cut off again keeps its whole lines and is recorded as failed with its own pages. A window whose connection drops after it streamed text is not written again: its whole lines stay, a recognisably unfinished last line is dropped, and it is recorded as failed (the "Incomplete coverage" banner then names any of its pages the notes do not cite). A cut-off single-pass draft is kept but marked incomplete, a cut-off revision is discarded in favour of the previous notes, and the tutor chat appends a "cut off" notice.
- **Judges see the evidence they judge.** Cut-off streams, the old rewrite/edit truncation and the old judge views were one bug class: a partial view treated as complete. On long sources the critique used to see only the first 12,000 characters of the writer's context and grounding only the first 14,000, so true claims about later pages looked unsupported. Now the critique is shown the whole passages the notes cite (up to 40,000 characters) and told which cited passages were left out; its flags on claims whose cited passages weren't shown are dropped, and flags on uncited claims are deferred to grounding. The reviser gets the cited passages plus that round's newly retrieved ones (up to 40,000 characters). Grounding judges each uncited claim against the passages retrieved for it, and leaves the claim as written when nothing is retrieved rather than judging it on a partial view. On a synthetic ~200,000-character source with 40 facts, the share of the needed evidence the critique and grounding were shown went from 0.05 to 1.00; before the fix only 2 of the 40 true late-document facts survived critique and revision. Known limits: retrieval can miss the right passage, so a true uncited claim can still be judged unsupported and removed; cited passages past the 40,000-character budget are named and skipped, not judged; the reviser can't add a topic that was neither cited nor retrieved; and the critic's own `needs_revision` flag or a low score can still trigger a wasted revision round (a revision only replaces the notes if it scores strictly higher).
- **Blocking calls stay off the event loop.** The server runs one uvicorn worker that serves every SSE stream, so remote Supabase token verification, YouTube transcript fetches and PDF/DOCX/Markdown/CSV exports run in a thread pool. Measured `/api/health` latency while a 1-second blocking call is in flight: 1.01 s before, 0.006 s after.
- **Graceful model routing.** NVIDIA NIM (Nemotron) and Gemini Flash with automatic failover, mechanical sub-tasks routed to cheaper models to preserve free-tier quota, and a local Ollama fallback for fully offline use. Gemini ids are pinned for speed and self-heal to Google's moving alias if a version is retired, so neither a dead id nor an overloaded brand-new one can stall the pipeline.

## Features

- **Inputs:** paste text · PDF · image (OCR) · audio (transcription) · article URL · YouTube transcript
- **Outputs:** cited notes · interactive multiple-choice quiz and spaced-repetition flashcards (generated on demand) · PDF / Markdown / DOCX / CSV export
- **Study tools:** tutor chat grounded in your notes · select-and-edit any passage with an instruction · one-click rewrite (shorter / longer)
- **Controls:** study mode (exam, summary, deep-dive…), tone, length, format, model picker
- **Accounts:** Supabase auth (email + Google), cloud session history, shareable public note links you can stop sharing at any time — with a local-only mode when auth isn't configured

## Security & operations

- SSRF-protected URL fetching (private/internal addresses blocked, redirects re-validated per hop, download size capped)
- Request size limits on every endpoint; chunked uploads with hard caps (PDF 20 MB, audio 25 MB, image 10 MB)
- Per-user rate limiting (short sliding windows plus daily caps on the token-spending endpoints) with Supabase JWT verification
- Share links can't be used to list other people's notes. The old RLS policy (`is_public = true` for anon) let anyone holding the public anon key list every user's shared notes, `user_id` included. Shared notes are no longer readable through the table: a link resolves through `public.get_shared_session(share_id)`, which returns one row by exact id with display columns only (no `user_id`). It is `SECURITY DEFINER` with an empty `search_path`, executable only by `anon`/`authenticated`, and owned by a dedicated `share_reader` role (no login, no RLS bypass, no access to `user_id`, and an RLS policy that only shows it shared rows), so even a bug in its `WHERE` clause could only return notes that are already public. Owners can stop sharing, and a dead link shows a clear "unavailable" page. The RLS rules are tested against a real PostgreSQL 16 running the actual `supabase/schema.sql` as a non-superuser. Deploying to an existing project is two steps (see the file's header): run STEP 1, deploy the frontend, then run STEP 2 to drop the old policy
- A production start with authentication switched off refuses to boot (`verify_auth_config`) unless `ALLOW_ANONYMOUS=true` is set explicitly
- Bounded worker pool + concurrency gate so heavy generations can't starve the server; blocking work runs off the event loop
- CI on every push and pull request to `main`: backend tests and the frontend tests + build; a push to `main` deploys to Hugging Face Spaces only after both pass

## Tech stack

| Layer          | Tech                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| Backend        | Python · FastAPI · SSE streaming                                           |
| Frontend       | React · Vite · Tailwind CSS                                                |
| Models         | NVIDIA NIM (Nemotron) · Gemini Flash · Ollama (local fallback)             |
| Retrieval      | BM25 (rank-bm25) + Gemini embeddings · rank fusion · BM25-only fallback    |
| Auth & storage | Supabase (JWT verification server-side, RLS-scoped sessions, share-link RPC) |
| Docs           | pypdf · reportlab · python-docx · Gemini audio + vision                    |
| CI/CD          | GitHub Actions → Hugging Face Spaces (Docker)                              |

## Run it locally

Use a virtual environment. The pinned `fastapi`/`starlette` versions conflict
with what other common packages (gradio, streamlit) require, so a shared global
install will break one side or the other.

```bash
# 1. Backend
cd backend
cp .env.example .env          # add NVIDIA_API_KEY (build.nvidia.com) and/or GEMINI_API_KEY
python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn main:app --reload   # http://127.0.0.1:8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

Run the backend from the `backend/` directory, or launch it with an absolute
path — `main.py` and friends resolve `.env` relative to their own location, so
the keys load either way, but `main:app` still has to be importable.

`python main.py` also works and binds `0.0.0.0:8000` instead of loopback.

Works with either provider key on its own. Setting both gives you NVIDIA as
primary with Gemini as automatic failover:

| Variable                                                         | Purpose                                          |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| `NVIDIA_API_KEY` / `NVIDIA_MODEL`                                | Primary writing model (exact catalog id required) |
| `HELPER_NVIDIA_MODEL` / `HELPER_GEMINI_MODEL`                    | Cheaper model for the mechanical steps (unset NVIDIA helper = main model) |
| `GEMINI_API_KEY`                                                 | Failover, semantic embeddings, image OCR, audio transcription |
| `OLLAMA_URL` / `OLLAMA_MODEL`                                    | Local offline fallback                            |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` or `SUPABASE_JWT_SECRET`    | Server-side token verification (required for a production start). Asymmetric (ES256/RS256) signing keys are verified locally via the project's JWKS automatically; `SUPABASE_JWT_SECRET` is only for legacy HS256 projects |
| `AUTH_ISSUER`                                                    | Expected token `iss` (default `SUPABASE_URL/auth/v1`); set for custom domains |
| `AUTH_VERIFY_CONCURRENCY`, `AUTH_VERIFY_PER_CLIENT`, `AUTH_JWKS_TTL_S` | Optional auth tuning: concurrent `/auth/v1/user` calls (default 8), per client address (default 2), JWKS cache TTL (default 600s). `SUPABASE_JWT_SECRET` without `SUPABASE_URL` cannot verify ES256/RS256 tokens (startup warning) |
| `ALLOW_ANONYMOUS`                                                | Explicitly allow a production start with auth off |
| `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`                   | Auth + cloud history (frontend `.env`)        |
| `DAILY_GENERATIONS`, `DAILY_EXTRACTS`, `DAILY_REGENS`, `DAILY_CHATS` | Per-account daily caps                        |
| `MAX_TEXT_CHARS`, `WORKER_THREADS`, `MAX_CONCURRENT_GENERATIONS`, `GENERATION_DEADLINE_S` | Server tuning                                 |
| `MAX_INFLIGHT_PER_USER`                                          | Expensive requests one account may have running at once (default 3; 429 beyond) |
| `EXPORT_THREADS`                                                 | Threads for export rendering (default 2), separate from the main worker pool |
| `MAX_PDF_MB`, `MAX_AUDIO_MB`, `MAX_IMAGE_MB`, `MAX_PDF_PAGES`    | Upload caps (20 / 25 / 10 MB) and pages parsed per PDF (500) |
| `MAX_JSON_BODY_BYTES`                                            | Largest JSON request body (default 2.5 MiB); bigger bodies get 413 before they are read |
| `TRUSTED_PROXY_HOPS`                                             | Reverse proxies in front of the app that append to `X-Forwarded-For` (default 1, right for Hugging Face Spaces / Render / Railway / Fly). The per-IP identity used when auth is off is the entry that many places from the right; entries further left are client-supplied and ignored. **On direct exposure (no proxy) it must be `0`**, and uvicorn must stop trusting forwarded headers: set `FORWARDED_ALLOW_IPS=127.0.0.1` (the Dockerfile defaults it to `*` for the proxied platforms). |
| `FORWARDED_ALLOW_IPS`                                            | Read by uvicorn: which peers may set `X-Forwarded-Proto`/`-For`. The Dockerfile sets `*` for proxied platforms (keeps redirects on https). **Direct exposure: `127.0.0.1`**, together with `TRUSTED_PROXY_HOPS=0` |

## Tests & evaluation

```bash
cd backend
pip install -r requirements-dev.txt
pytest        # 600+ tests: pipeline, long-document coverage, judge evidence on long sources, rewrite/edit integrity, quiz answer-key verification, parsers, retrieval, security, Supabase RLS / share links (real PostgreSQL via pgserver; skipped if it isn't installed), stream integrity, event-loop blocking, truncation-safety

cd ../frontend
npm test      # Vitest: SSE parser and inline-edit line-anchor resolution
```

Two evals live in `backend/eval/`, both graded by a judge that must be a different model family from the writer (`--judge-model` is required):

- `python -m eval.claim_eval` runs the full pipeline 3 times on each of three long fixtures (over 61,000 characters each, one a 44-page PDF) with hand-labelled facts and planted traps: facts stated only at the end, an internal contradiction, and a question bank. It reports claim precision, miscitation rate, fact recall, contradictions caught, question-bank violations, claims removed by grounding, estimated tokens, cost and time, as mean and spread, and with `--gate` fails on any metric worse than the stored baseline by more than its margin.
- `python -m eval.run_eval` is the older short-fixture eval (holistic faithfulness, coverage, quiz accuracy).
- `python -m eval.benchmark_retrieval` measures recall@5/@10 and MRR on 42 queries over the long fixtures.

See [`evals/README.md`](evals/README.md) for fixtures, metric definitions, baselines and the gate.

### What has been measured, before and after Phase 1

Every number in this table comes from a run that needs **no model and no API key**. Nothing here is model-graded: see the last row.

| Measurement | Before (`329a031`) | After | Where it comes from |
| --- | --- | --- | --- |
| Backend tests passing, each commit's own suite, on Linux | 590 | 656 (at `78bd704`) | recorded in `evals/baselines/` |
| Backend tests passing, at `55dab94`, on Windows | n/a | 644 | run for this README. One file is skipped: the PostgreSQL row-level-security tests need `pgserver`, which has no Windows wheel. The Linux figures above include that file |
| Frontend tests passing (Vitest) | n/a | 22 | run for this README |
| Grounding-fix tests (`test_grounding_fixes.py`), run against each commit's `agent.py` | 6 of 23 | 23 of 23 | "before" recorded in `evals/baselines/329a031.json`; "after" re-run for this README |
| Retrieval benchmark: chunks per document | 3 | 106–127 | re-run for this README |
| Retrieval benchmark: BM25 recall@5 | 1.000 (every query returned every chunk) | 0.893 | re-run for this README, 42 queries |
| Retrieval benchmark: BM25 MRR | 1.000 | 0.861 | re-run for this README |
| Eval fixtures long enough to take the long-document path (over 12,000 characters) | 0 of 4 | 3 of 3 | measured from the fixture files |
| Claim precision, fact recall, miscitation, contradictions caught, cost | did not exist | **Claim-level eval: harness built, first full run pending.** | `backend/eval/claim_eval.py` |

How to read it:

- **590 and 656 are each commit's own test suite**, not today's tests run against the old code. The suite grew by 66 tests (43 for the eval harness, 23 for the grounding fixes) and every one passes. The 644 on Windows is the same suite with 7 more tests added and the PostgreSQL file skipped. The long-document quality pass after it added 86 (730 on Windows).
- **The retrieval rows do not mean retrieval got worse.** Retrieval code is unchanged. The old 1.000 came from a 3-chunk document on which ranking could not fail: every query returned every chunk. On the new dataset BM25 finds no relevant chunk in the top 10 for 4 of the 12 paraphrased queries (recall 0.625 in that category, 1.000 in the other five).
- **Hybrid retrieval is not measured here.** With no embeddings key, the semantic and hybrid columns of the benchmark fall back to BM25, so all three print the same numbers.

Two caveats about the eval that hold even after its first full run:

- The eval writes with `gemini-3.5-flash-lite`. **Production writes with Nemotron.** A result therefore measures the pipeline on a different writer from the one users get.
- The judge must be a different model family from the writer. The default judge is a Nemotron model, so it can grade the Gemini eval writer but **could not grade the production writer**, which is the same family. Measuring production needs a different judge.

An older run of the short-fixture eval scored faithfulness 9.5, with the writer and the judge being the same model. It is recorded in `evals/baselines/329a031.json` as "self-judged by the writer, not comparable" and is not quoted as a result.

## Known limitations

- **No model-graded result exists yet.** The claim-level eval is built and unit-tested; it has not had a full run.
- **Edited text is not re-grounded.** A rewrite or inline edit cannot add a citation the notes did not have, and is marked "not re-verified", but the edited claim itself is not checked against the source.
- **Grounding fails open.** A claim the judge could not rule on, or ruled on without quoting evidence, is kept as written.
- **The long-document merge is not measured on a real run.** It is tested on faked model output only. It removes a bullet that repeats an earlier one almost word for word; the same point reworded is kept. Windows that used different heading levels are nested rather than merged. A lone unpunctuated line left by a dropped connection cannot be told from a whole one and stays.
- **Rate limits live in process memory** (`backend/auth.py`), so a restart resets the daily caps. On a host that sleeps and restarts, "daily" is not really daily.
- **One server process.** A single uvicorn worker, and at most 4 generations at once; the next request gets a 503.
- **Free-tier model quotas.** Providers answer bursts with 429 and 503. The router retries and fails over, but a run can still be slow or fail.
- **Retrieval has no reranker**, chunks are cut by length and page rather than by heading, and BM25 alone misses a third of paraphrased queries.
- **No per-run trace.** Stage timings and the serving provider are logged; the prompts and responses of one run cannot be replayed.

## Project structure

```
agentic-notes/
├── backend/
│   ├── main.py              # FastAPI app: SSE generation, extraction, exports, auth
│   ├── agent.py             # Multi-agent pipeline orchestrator
│   ├── models.py            # NVIDIA/Gemini/Ollama router, JSON-mode calls
│   ├── retriever.py         # Chunking + index orchestration
│   ├── retrieval/           # BM25, semantic, fusion, hybrid
│   ├── eval/                # LLM-judge eval harness + retrieval benchmark
│   ├── auth.py              # Supabase JWT verification + rate limiting
│   ├── pdf_export.py        # PDF / Markdown / DOCX / CSV rendering
│   └── tests/               # 600+ backend tests
├── frontend/
│   └── src/                 # React app: streaming UI, quiz, flashcards, chat, auth
└── .github/workflows/       # CI: test → build → deploy
```

## Next

- First full run of the claim-level eval, and a recorded baseline to compare against
- Split `backend/agent.py` (3,100+ lines) along its existing section banners
- Ragas as a second opinion on the same fixtures
- Per-run tracing: every prompt and response of one run, in order
- Type hints and a type checker
- Notes across several documents
- Resumable runs, so a long generation survives a closed tab
- Retrieval index caching across requests
- Pairwise (A/B) critique comparison instead of absolute scoring
- Keep-alive job for the free-tier database

---

Built by [Kodishala Thrilok](https://github.com/Kodishalathrilok). If this repo is useful to you, a ⭐ is appreciated.
