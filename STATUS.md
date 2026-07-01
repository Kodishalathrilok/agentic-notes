# Agentic AI Notes Generator — Project Status

A full-stack, multi-agent AI study-notes generator. Feed it text, a PDF, a photo,
a web link, a YouTube video, or audio, and a pipeline of AI agents produces
faithfulness-checked notes, a verified quiz, and spaced-repetition flashcards —
streamed live, with citations, an eval suite, accounts, and CI/CD.

- **Repo:** https://github.com/Kodishalathrilok/agentic-notes
- **Live (Hugging Face Space):** https://thrilokkk-agentic-notes.hf.space
- **Stack:** FastAPI · React + Vite + Tailwind · Groq/Gemini/Ollama · Supabase · Docker

---

## 1. The agentic pipeline (backend/agent.py)

Runs as a synchronous generator streamed to the UI over SSE.

| # | Agent / step | What it does |
|---|---|---|
| 0 | **Gatekeeper** | Rejects non-academic input (celebrity/entertainment/ads) before spending tokens |
| 1 | **Plan** | Outline + checklist + difficulty (JSON mode); sees a breadth sample of the source |
| 2 | **Write** | Drafts notes from retrieved passages, streamed token-by-token, with `[n]` citations |
| 3 | **Critique** | **Grounded** against the source — flags unsupported/fabricated claims + missing topics |
| 4 | **Revise** | Iterative loop: revise → re-critique, up to 2 rounds, keeps the best-scoring version |
| 5 | **Title** | Auto-names the session |
| 6 | **Quiz** | 5 MCQs, then a **verifier** re-checks every answer key against the notes |
| 7 | **Flashcards** | 8 front/back cards |

**Quality mechanisms:** grounded faithfulness critique, iterative self-correction,
JSON-mode structured outputs with conservative fallbacks, quiz answer verification,
per-agent temperatures.

---

## 2. Multi-provider LLM layer (backend/models.py)

Provider-agnostic router with **automatic failover**:

- **Groq** (Llama 3.3 70B / 3.1 8B / Gemma2) — primary, fastest streaming
- **Gemini** (2.0 Flash / Flash-Lite) — failover + 1M context + vision
- **Ollama** — local offline fallback

Routes by model id; if one provider errors (e.g. a rate limit), it transparently
falls over to the next — for both streaming and non-streaming calls. Model picker
shows all configured providers. API keys are redacted from error logs.

---

## 3. RAG + citations (backend/retriever.py)

- **Chunking** of the source (word-based, handles caption transcripts).
- **Pluggable embeddings:** Gemini `gemini-embedding-001` (semantic) when a key is
  set, else a pure-Python **TF-IDF** fallback (no deps, no key).
- Agents write from the **top-k retrieved passages** (not a truncated dump).
- **Clickable `[n]` citations** in the notes jump to the source passage.

---

## 4. Inputs (multimodal)

| Input | Endpoint | Notes |
|---|---|---|
| Text | — | paste directly (up to 50,000 chars) |
| PDF | `/api/extract-pdf` | pypdf text extraction |
| **Image** | `/api/extract-image` | **Gemini vision OCR** (textbook page, slides, handwriting) |
| URL | `/api/extract-url` | article text **or YouTube transcript** |
| Audio | `/api/transcribe` | Groq Whisper |

---

## 5. Outputs & study tools (frontend)

- **Notes:** formatted (headings/bullets/bold), **KaTeX math**, code blocks,
  copy, inline **edit**, **make shorter/longer**, **revision diff**, **read aloud** (TTS).
- **Inline AI assistant:** select any text → **Explain** (streamed) or **Rewrite** in place.
- **Quiz:** interactive answering, scoring, regenerate, verified answer keys.
- **Flashcards:** flip, keyboard nav, shuffle, **spaced repetition** (Again/Good/Easy,
  due filter), **Anki CSV export**, regenerate.
- **Exports:** PDF, Markdown, DOCX, flashcards CSV.
- **History:** search, rename, tags.

---

## 6. Accounts & data (Supabase)

- **Auth:** email/password + **"Continue with Google"** (OAuth). Optional — the app
  falls back to localStorage when Supabase isn't configured.
- **Cloud-synced history:** per-user `sessions` table with **row-level security**.
- **Shareable links:** make a note public → `?share=<id>` opens a read-only view.

---

## 7. Evaluation (backend/eval/)

- **LLM-judge harness** scores faithfulness / coverage / clarity / quiz accuracy on
  fixed fixtures, comparing **baseline vs full** pipeline.
- **In-app Eval dashboard** visualizes the report.
- **CI gate:** a workflow re-runs the eval and **fails the build if faithfulness
  drops** below threshold.

---

## 8. Engineering / infra

- **Tests:** 28 pytest tests (parsers, exports, retriever, mocked-LLM pipeline,
  API validation) — all offline.
- **CI:** GitHub Actions runs backend tests + frontend build on every push/PR (green badge).
- **CD:** auto-deploys to the Hugging Face Space on green `main` (needs `HF_TOKEN` secret).
- **Deploy:** single-service Docker image (FastAPI serves the built React app) on
  **Hugging Face Spaces** (free, no card).

---

## 9. Status checklist

**Done ✅**
- Multi-agent pipeline with grounded critique + iterative revision
- Academic gatekeeper
- Multi-provider LLM router (Groq → Gemini → Ollama failover)
- RAG with citations (Gemini embeddings + TF-IDF fallback)
- Image OCR (Gemini vision), PDF, URL/YouTube, audio inputs
- Inline AI assistant (explain/rewrite on selection)
- Quiz (verified) + flashcards (SRS) + all exports
- Eval harness + dashboard + CI faithfulness gate
- Tests + CI/CD + Docker + live deploy on Hugging Face
- Animated landing page
- Supabase accounts, cloud history, shareable links (email/password working)

**Pending / setup ⏳**
- **Google OAuth:** code done; needs Google Cloud OAuth credentials + Supabase
  Google provider enabled (in progress).
- Add `HF_TOKEN` GitHub secret to activate auto-deploy (if not yet done).
- Add `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY` as HF **Variables** so accounts
  work on the live Space.
- Rotate any API keys that were exposed during setup.

**Possible next features**
- Study analytics dashboard (scores over time, streak, weak topics)
- More quiz types (true/false, fill-in-blank, short-answer)
- Mind-map / concept-map view
- Reranking + hybrid retrieval for even better RAG

---

## 10. One-line pitch (for interviews)

> A self-correcting multi-agent notes generator with a provider-agnostic LLM layer
> (Groq/Gemini/Ollama failover), retrieval-grounded generation with citations,
> multimodal input (incl. vision OCR), an automated eval suite that gates merges on
> faithfulness, accounts + row-level-secured data, and full CI/CD to a live deploy.
