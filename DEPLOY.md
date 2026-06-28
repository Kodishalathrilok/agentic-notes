# Deploying Agentic Notes

This app deploys as a **single service**: the FastAPI backend serves both the
`/api/*` endpoints and the built React frontend. One URL, no CORS, no database
(history/settings live in the browser's localStorage).

> **Why not Vercel/Netlify functions?** The app streams responses (SSE pipeline,
> streamed chat, 30–60s generations). Use an always-on host that allows
> long-lived streaming connections: **Render, Railway, or Fly.io**. The static
> frontend is bundled into the same service, so you don't deploy it separately.

---

## What's already set up

- **`Dockerfile`** (repo root) — builds the frontend, then bundles it with the
  Python backend into one image.
- **`main.py`** — serves the built SPA from `backend/static` when present and
  reads `PORT` from the environment.
- **`.dockerignore`** — keeps secrets and `node_modules` out of the image.

## Environment variables (set these as host secrets)

| Variable | Required | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Your `gsk_...` key from console.groq.com |
| `GROQ_MODEL` | no | Defaults to `llama-3.3-70b-versatile` |
| `MAX_TEXT_CHARS` | no | Defaults to `50000` |
| `PORT` | auto | Most hosts set this for you |

Never commit `.env` — it's gitignored and excluded from the image.

---

## Option A — Render (Docker)

1. Push the repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. **Runtime: Docker** (it auto-detects the `Dockerfile`).
4. Add the env var `GROQ_API_KEY` (and any optional ones).
5. Create the service. Render builds the image and gives you a public URL.

Notes: the **free tier spins down when idle** (first request after a pause has a
~30s cold start) and has **no persistent disk** — fine here since there's no DB.

## Option B — Railway

1. Push to GitHub → Railway → **New Project → Deploy from GitHub repo**.
2. Railway detects the `Dockerfile` and builds it.
3. Add `GROQ_API_KEY` under **Variables**. Railway injects `PORT` automatically.
4. Deploy → open the generated domain.

## Option C — Fly.io

```bash
fly launch              # detects the Dockerfile, creates fly.toml
fly secrets set GROQ_API_KEY=gsk_your_key_here
fly deploy
```

Fly also supports a persistent **volume** later if you ever add SQLite.

---

## Test the production build locally (optional)

From the repo root (`agentic-notes/`):

```bash
docker build -t agentic-notes .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key_here agentic-notes
```

Open <http://localhost:8000> — this is exactly what the host will serve (API +
frontend from one origin).

---

## Adding a database later (optional)

You only need one if you want **user accounts** and **cross-device history**
(today everything is per-browser via localStorage). When you do:

- **Supabase** — managed Postgres **plus** ready-made auth (recommended).
- **Neon** — managed Postgres only.

That's a follow-up feature (auth + DB), not required to ship.
