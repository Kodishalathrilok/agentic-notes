# Demo script and interview Q&A

Everything here points at a file or a measured number. Where something has not
been checked, it says **unsure** instead of guessing.

---

## Before you start

- Open the live app, signed in: <https://thrilokkk-agentic-notes.hf.space>
- Have a text-based PDF ready, roughly 10 to 40 pages. Under 12,000 characters
  it takes the single-pass path; over that it takes coverage windows
  (`SECTION_DOC_THRESHOLD`, `backend/agent.py`). Pick the one you want to show.
- **Have a shared-note link open in another tab as a fallback.** The app runs
  on free-tier model keys, which answer bursts with 429 and 503. The router
  retries and fails over (`backend/models.py`), but a live run can still be
  slow or fail. A shared link opens with no sign-in and costs nothing.
- **Unsure:** how long a run takes on the day. It depends on the provider and
  the document. Do one run beforehand and time it.

## The 3-minute script

| Time | Click | Say |
|---|---|---|
| 0:00 | Landing page | "It turns any study source into notes where every claim is traced to the page it came from. Claims the source doesn't support are removed by code, not left to the prompt." |
| 0:20 | **Launch the app**, add the PDF, **Generate Notes** (or Ctrl+Enter) | "The server streams every stage over SSE, so you watch it work." |
| 0:35 | Point at the pipeline strip while it runs | "A gatekeeper checks it's study material while the search index builds in parallel. Then a planner, a writer, a critic, and a revise loop: at most 2 rounds, and a revision only replaces the notes if it scores strictly higher." |
| 1:15 | Click a citation chip, e.g. `p. 4` | "The PDF jumps to that page. The model never supplies a page number. It names a passage, and the retriever's metadata decides the page. A citation to a passage the model was never shown is deleted by a deterministic check." |
| 1:45 | Open **Pipeline Insights** | "This is the critic's score, the claims it called unsupported, and what was missing. A score under 8, any unsupported claim, or the critic asking for one triggers a revision." |
| 2:10 | Point at the grounding status line | "After that, every claim line is judged against its own evidence. A line is only removed if the verdict quotes that evidence and the quote is really there. A bare 'unsupported' label deletes nothing." |
| 2:30 | **More actions → Make shorter**, then **Show changes** | "A rewrite never returns partial notes: if any part fails, nothing changes. This is the diff against the previous version." |
| 2:45 | **History → Share**, open the link in a private window | "No sign-in. The link goes through one database function that returns a single row by id and never exposes who owns it. **Stop sharing** kills it." |
| 3:00 | Stop | "600+ backend tests run offline with no API keys. What isn't done yet: the claim-level eval is built and tested but hasn't had its first full run, so I don't quote a model-graded score." |

### If something breaks during the demo

- **A run fails or stalls:** switch to the shared-note tab and say why. Free
  tiers rate-limit; that is the honest reason and it is a real constraint of
  the project, written up in the README.
- **A part of a long document fails:** the notes get an "Incomplete coverage"
  banner naming the missing pages. Show it. It is a feature: the alternative
  was notes that silently passed as complete.

---

## Interview Q&A

### Why build retrieval and orchestration yourself instead of LlamaIndex or LangGraph?

I used libraries where they exist. BM25 is `rank-bm25`, embeddings come from
Gemini's API, the server is FastAPI, PDFs are `pypdf`
(`backend/requirements.txt`). I did not write a BM25 or a web framework.

What I wrote myself, and why:

- **Chunking** (`chunk_document()`, `backend/retriever.py`). Chunks have to stop
  at page boundaries for a citation's page to be a fact. With plain 700-character
  chunks, a chunk covered about 3 slides and was labelled with the first, so
  every citation landed 1 to 3 pages early (the docstring records this).
- **Rank fusion** (`backend/retrieval/fusion.py`). It is about 20 lines.
- **Orchestration** (`run_agent()`, `backend/agent.py`). It is a generator that
  yields events, so streaming, ordering and cancellation are one mechanism.

What a framework would have given me that this project lacks: checkpointing
and resumable runs, retries as configuration, and tracing. Those are real
gaps. Resumable runs and tracing are on the roadmap.

I have **not** used LlamaIndex or LangGraph in this project. On a team I would
start from them and keep the verification layer, because deterministic citation
validation and per-claim grounding are not something either gives you.

### How would you use Ragas?

As a second opinion, not a replacement. The project already has a claim-level
eval (`backend/eval/claim_eval.py`) with its own metric definitions in
`evals/README.md`: claim precision, miscitation rate, fact recall,
contradictions caught. Ragas has standard metrics in the same area
(faithfulness, context precision, context recall).

I would run both on the same three long fixtures and look at where they
disagree, because a disagreement is where one of the two is wrong. It is not
done; it is on the roadmap, after the first full run of my own eval.

### Why is the writer's context budget 40,000 characters?

It is derived, and the derivation is the comment above `WRITER_CONTEXT_CHARS`
(`backend/agent.py`, line 190):

- The single-pass writer had already shipped with 40,000 characters on the same
  providers, so that much input was proven.
- Worst-case prompt scaffolding was measured at 3,600 characters, so a full
  prompt is about 43,600 characters.
- Output is budgeted separately.
- That is roughly 2.7% of the smaller provider limit seen in testing.

The reason it exists at all is a bug. A window's prompt used to end with
`context[:16000]`, so a window could be assigned pages 1 to 17 and forward only
the first few: selection covered 100% of the document while the writer received
62% (same comment, line 185). No test caught it because the tests stopped at
selection. Windows are now built to fit the budget, and when a document is too
big for the default 12 windows, the number of windows grows instead.

### Why is `agent.py` so big, and how would you split it?

It is 3,140 lines, which is too many for one file. It grew one stage at a time
and I did not stop to split it.

It already has section banners, and they are the split:

| New module | Sections today (lines) |
|---|---|
| `coverage.py` | Document coverage windows (157–342) |
| `grounding.py` | Citation validation and the grounding check (343–816) |
| `stages.py` | Digest, plan, write, critique, revise (1017–1614) |
| `editing.py` | Rewrite and inline edit (1615–1916) |
| `study_aids.py` | Title, quiz, flashcards, chat (1917–2296) |
| `orchestrator.py` | Gatekeeper and `run_agent` (2297–end) |

Why it is not done yet: 13 test files replace `agent.call_model` by patching it
on the module. After a split those patches have to follow the function to its
new module, or the tests quietly stop mocking the model. It is a mechanical
change, but it touches a lot of tests, so it should be its own piece of work
with nothing else in it.

### How do you debug a bad generation?

Today, with four things:

- **Stage timings.** Every run logs `[timing]` lines per stage plus a total
  (`_RunTimings`, `backend/agent.py`). Names and durations only, never content.
- **The `done` event.** It reports which provider and model actually produced
  the notes, whether a fallback happened and why, the document type, and the
  coverage record. A silent failover to another provider is visible.
- **`CITATION_DEBUG=1`.** Prints each citation's claim → passage → page, and
  each line the grounding check removed or rewrote.
- **`/api/health`.** Active provider, embeddings backend, whether auth is on.

What is missing: there is no per-run trace. I cannot open one run and see every
prompt and response in order, and there is no id tying the stages of one run
together across logs. Tracing is the next thing I would add.

### What does the eval prove, and what doesn't it?

**Measured, with no model involved** (`evals/baselines/`, README table):

- The grounding fixes: 6 of 23 targeted tests passed on the old code, 23 of 23 now.
- Retrieval on 42 labelled queries over the long fixtures: BM25 recall@5 is
  0.893, MRR 0.861. BM25 finds no relevant chunk in the top 10 for 4 of the 12
  paraphrased queries.
- All 3 eval fixtures reach the long-document path. The old 4 fixtures never did.

**Not proven:**

- **There is no model-graded score.** The claim-level eval is built and tested,
  and has not had its first full run. I do not quote a faithfulness number.
- An older run scored faithfulness 9.5, but the writer and the judge were the
  same model. It is recorded as "self-judged by the writer, not comparable"
  (`evals/baselines/329a031.json`) and is not evidence of anything.
- The eval writes with `gemini-3.5-flash-lite`. Production writes with Nemotron.
  So even a full run measures the pipeline on a different writer than users get.
- The default judge is a Nemotron model. The judge must be a different model
  family from the writer (`model_family()`, `backend/eval/claim_eval.py`), so
  this judge cannot grade the production writer at all. That needs another judge.
- Hybrid retrieval (BM25 plus embeddings) was not measured: the session that
  recorded the benchmark had no embeddings key.
- Three fixtures is a small sample.

### Is this really "agentic"?

It is a fixed pipeline of specialised stages with one feedback loop, not agents
choosing their own tools. The stages do pass structured results to each other,
and the critic's output decides whether to revise and what to retrieve again.
I would call it a multi-stage pipeline with self-correction.

### What breaks under load?

- One uvicorn worker serves every stream (`Dockerfile`).
- At most 4 generations run at once (`MAX_CONCURRENT_GENERATIONS`,
  `backend/main.py`); the next request gets a 503 straight away.
- Rate limits are held in process memory (`backend/auth.py`), so a restart
  resets the daily caps. On a host that sleeps and restarts, "daily" is not
  really daily. A shared store such as Redis is the fix.
- Free-tier provider quotas run out long before the server does.
