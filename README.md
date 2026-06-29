# 📚 Agentic AI Notes Generator

[![CI](https://github.com/Kodishalathrilok/agentic-notes/actions/workflows/ci.yml/badge.svg)](https://github.com/Kodishalathrilok/agentic-notes/actions/workflows/ci.yml)

A full-stack multi-agent AI notes generator. Feed it text, a PDF, or audio and
it runs a pipeline of AI agents — **Plan → Write → Critique → Revise → Quiz →
Flashcards** — streaming each step to the UI in real time over SSE.

The pipeline is **self-correcting** (a grounded critique re-checks every claim
against the source and triggers revision) and **measured** (an LLM-judge eval
harness scores faithfulness, coverage, and quiz accuracy).

## Tech Stack

- **Backend:** Python + FastAPI + SSE streaming
- **Frontend:** React + Vite + Tailwind CSS
- **Models:** Groq API (primary, free) with Ollama (local fallback)
- **PDF export:** reportlab · **PDF reading:** pypdf · **Audio:** Groq Whisper

## Features

- Multi-agent pipeline with a self-critique → revise loop
- **Retrieval-augmented (RAG):** the source is chunked and indexed (TF-IDF), and
  each agent works from the most relevant passages instead of a truncated dump
- **Citations:** every note point links to the source passage it came from
- Real-time streaming agent status (Plan / Write / Critique / Revise / Quiz / Flashcards)
- Three input modes: paste text, upload a PDF, or record audio
- Interactive quiz (click to answer, scoring) and flip-card flashcards
- Export to **PDF** or **Markdown**
- Session **history** saved in localStorage (max 20)
- Dark mode, provider badge (Groq ⚡ / Ollama 🦙)

## Project Structure

```
agentic-notes/
├── backend/
│   ├── main.py            # FastAPI app + endpoints (SSE, extract, transcribe, export)
│   ├── agent.py           # Multi-agent pipeline orchestrator
│   ├── models.py          # Groq → Ollama model router + safe_json
│   ├── pdf_export.py      # Markdown + PDF rendering
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html
    ├── vite.config.js     # proxies /api → http://localhost:8000
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── package.json
    └── src/
        ├── main.jsx
        ├── index.css
        ├── App.jsx
        ├── hooks/useStream.js
        └── components/
            ├── InputPanel.jsx
            ├── ControlPanel.jsx
            ├── AgentStatus.jsx
            ├── NotesOutput.jsx
            ├── QuizPanel.jsx
            ├── FlashcardPanel.jsx
            └── HistoryPanel.jsx
```

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env (get a free key at https://console.groq.com)
python main.py
```

The API runs on `http://localhost:8000`.

> **Windows note:** if `cp` isn't available, use `copy .env.example .env`
> (cmd) or `Copy-Item .env.example .env` (PowerShell).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**.

## Using Ollama instead (no API key needed)

1. Install Ollama from <https://ollama.ai>
2. Pull the model: `ollama pull qwen2.5:3b`
3. Leave `GROQ_API_KEY` empty (or as the placeholder) in `.env`

The backend automatically falls back to Ollama when no valid Groq key is set,
and also if a Groq call fails at runtime. Audio transcription requires Groq
Whisper and is unavailable in Ollama-only mode.

## API Endpoints

| Method | Path                  | Purpose                              |
| ------ | --------------------- | ------------------------------------ |
| GET    | `/api/health`         | provider + config status             |
| POST   | `/api/generate`       | SSE stream of the agent pipeline     |
| POST   | `/api/extract-pdf`    | extract text from an uploaded PDF    |
| POST   | `/api/transcribe`     | transcribe audio via Groq Whisper    |
| POST   | `/api/export/pdf`     | download notes as PDF                 |
| POST   | `/api/export/markdown`| download notes as Markdown            |

## Evaluating pipeline quality

An eval harness measures the agent pipeline with an independent LLM-judge and
compares the **baseline** (plan→write only) against the **full** pipeline
(grounded critique + iterative revise loop + quiz verification).

```bash
cd backend
python -m eval.run_eval                 # both variants, 1 run each
python -m eval.run_eval --runs 2        # average over 2 runs per fixture
python -m eval.run_eval --variant full  # only the full pipeline
```

It scores faithfulness, coverage, clarity, and quiz answer-key accuracy over the
fixtures in `eval/fixtures.py`, prints a comparison table, and writes
`eval/report.md` + `eval/report.json`. Add your own test inputs by appending to
`FIXTURES`. Note: this makes real model calls and uses API quota.

**Dashboard:** the in-app **Eval** page (top-right link) reads `eval/report.json`
and visualizes baseline-vs-full scores, the faithfulness lift, and per-variant
stats.

**CI gate:** the `Eval (faithfulness gate)` GitHub Actions workflow re-runs the
eval (manually or weekly) and **fails the build if faithfulness drops below a
threshold**:

```bash
python -m eval.run_eval --variant full --check-faithfulness 7.0
```

It needs a `GROQ_API_KEY` repository **secret** (Settings → Secrets and variables
→ Actions). It's scheduled/manual — not on every push — to control API cost.

## Deployment

The app deploys as a **single service** (FastAPI serves both the API and the
built React frontend) — no database required. See **[DEPLOY.md](DEPLOY.md)** for
Render / Railway / Fly.io steps and the included `Dockerfile`.

## Notes

- The default Groq model is `llama-3.3-70b-versatile` (Groq's current versatile
  70B model). Override it with `GROQ_MODEL` in `.env` if you prefer another.
- Text input is capped at 50,000 characters (override with `MAX_TEXT_CHARS` in `.env`).
