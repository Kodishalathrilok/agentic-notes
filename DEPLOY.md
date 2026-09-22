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
| `NVIDIA_API_KEY` | no | Key from build.nvidia.com. **When set, NVIDIA becomes the primary provider** and other keys act as fallbacks |
| `NVIDIA_MODEL` | no | Exact catalog id, e.g. `nvidia/llama-3.1-nemotron-ultra-253b-v1`. Required if `NVIDIA_API_KEY` is set |
| `NVIDIA_MODELS` | no | Extra ids (comma-separated) for the model picker |
| `HELPER_NVIDIA_MODEL` | no | Cheap model for the mechanical agents. Unset = they run on `NVIDIA_MODEL` |
| `NVIDIA_BASE_URL` | no | Defaults to `https://integrate.api.nvidia.com/v1` |
| `GEMINI_API_KEY` | yes¹ | Free key from aistudio.google.com. Also the ONLY provider for image OCR and audio transcription |
| `GEMINI_MODEL` | no | Defaults to `gemini-3.6-flash`. Retired ids self-heal to `gemini-flash-latest`, with a log line telling you to update the pin |
| `HELPER_GEMINI_MODEL` | no | Defaults to `gemini-3.5-flash-lite` |
| `MAX_TEXT_CHARS` | no | Defaults to `300000` |
| `MAX_JSON_BODY_BYTES` | no | Largest JSON request body. Defaults to 2.5 MiB (or 8 bytes × `MAX_TEXT_CHARS` if that is larger); bigger bodies get **413** before they are read or authenticated |
| `MAX_PDF_MB` / `MAX_AUDIO_MB` / `MAX_IMAGE_MB` | no | Upload caps, default `20` / `25` / `10`. The upload routes' body limit is the cap + 1 MB |
| `MAX_PDF_PAGES` | no | Pages parsed per PDF, default `500`. Past it (or past `MAX_TEXT_CHARS` of text) extraction stops and the response says `truncated` |
| `MAX_INFLIGHT_PER_USER` | no | Expensive requests (generate, chat, quiz/flashcards/rewrite/edit, extract, export) one account may have running at once, default `3`; more get **429** |
| `GENERATION_DEADLINE_S` | no | End-to-end time budget for one `/api/generate` stream, default `900` |
| `TRUSTED_PROXY_HOPS` | no | Reverse proxies in front of the app that append to `X-Forwarded-For`, default `1` (right for Render, Railway, Fly and Hugging Face Spaces). The per-IP identity used when auth is off is the entry that many places from the right; entries further left are written by the client and ignored. **If the container is exposed directly (no proxy), set `0`** and set `FORWARDED_ALLOW_IPS=127.0.0.1` (the Dockerfile defaults it to `*` so `X-Forwarded-Proto` from the platform proxy keeps redirects on https) |
| `FORWARDED_ALLOW_IPS` | no | Read by uvicorn. The Dockerfile sets `*` so the platform proxy's `X-Forwarded-Proto` keeps redirects on https. **If the container is exposed directly, set `127.0.0.1`** (and `TRUSTED_PROXY_HOPS=0`). Otherwise uvicorn believes any address a client puts in `X-Forwarded-For`; the app detects this and puts such requests in one shared rate-limit bucket |
| `EXPORT_THREADS` | no | Threads for PDF/DOCX/Markdown/CSV export rendering, default `2`. Exports get their own pool so a burst of them can't stall generations or chats |
| `WORKER_THREADS` | no | Main worker pool. Default `max(16, MAX_CONCURRENT_GENERATIONS + 12)`: each running generation holds one thread, and chats/extracts/quiz calls need the rest |
| `PORT` | auto | Most hosts set this for you |
| `SUPABASE_URL` | **yes** | Server-side auth — see below |
| `SUPABASE_ANON_KEY` | **yes** | Server-side auth — see below |
| `SUPABASE_JWT_SECRET` | no | Alternative to the two above: verifies tokens locally, no network call |
| `ALLOWED_EMAILS` | no | Comma-separated allowlist. Empty = any signed-in user |
| `ALLOW_ANONYMOUS` | no | Set `true` only to deliberately run with no auth |

¹ At least one model provider key is required. `GEMINI_API_KEY` alone, or
`NVIDIA_API_KEY` alone, both work — setting both gives you NVIDIA as primary
with automatic failover to Gemini when NVIDIA is rate-limited or down. Note
that image OCR (`/api/extract-image`) and audio transcription
(`/api/transcribe`) need `GEMINI_API_KEY` either way; they return 503 without
it.

### Server-side auth is not optional

`SUPABASE_URL` / `SUPABASE_ANON_KEY` here are **separate from** the
`VITE_SUPABASE_*` variables in the section below. The `VITE_` ones are baked
into the frontend at build time; these are read by the server at runtime.

Setting only the `VITE_` ones gives you a sign-in button that does nothing:
the frontend sends an access token with every request, but the server never
verifies it and serves all callers as anonymous — so anyone with the URL can
spend your model credits. Since v1.1 a production start in that state refuses
to boot rather than failing silently.

Verify a deploy with:

```bash
curl -s https://<your-host>/api/health          # expect "auth_required": true
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://<your-host>/api/generate \
  -H 'Content-Type: application/json' -d '{"text":""}'
```

The second call must return **401**. A **422** means auth is off — the request
got past the auth check and only then failed input validation.

Never commit `.env` — it's gitignored and excluded from the image.

### Supabase schema (`supabase/schema.sql`)

Run the file in the Supabase SQL Editor; it is idempotent. On a project that
already has the old share policy, follow the order in the file's header
(STEP 1, deploy the frontend, then STEP 2). **STEP 3** adds size limits
(CHECK constraints on `notes`, `quiz`, `flashcards`, `title`, `sources`,
`tags`). It is additive and can be run at any time. The constraints are added
`NOT VALID`, so rows that already exist never block it; the file shows the
optional `validate constraint` statements that also check existing rows.

---

## Option A — Render (Docker)

1. Push the repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. **Runtime: Docker** (it auto-detects the `Dockerfile`).
4. Add the env vars `NVIDIA_API_KEY` / `NVIDIA_MODEL` and/or `GEMINI_API_KEY`.
5. Create the service. Render builds the image and gives you a public URL.

Notes: the **free tier spins down when idle** (first request after a pause has a
~30s cold start) and has **no persistent disk** — fine here since there's no DB.

## Option B — Railway

1. Push to GitHub → Railway → **New Project → Deploy from GitHub repo**.
2. Railway detects the `Dockerfile` and builds it.
3. Add `NVIDIA_API_KEY` / `GEMINI_API_KEY` under **Variables**. Railway injects `PORT` automatically.
4. Deploy → open the generated domain.

## Option C — Fly.io

```bash
fly launch              # detects the Dockerfile, creates fly.toml
fly secrets set NVIDIA_API_KEY=nvapi-your_key_here GEMINI_API_KEY=your_key_here
fly deploy
```

Fly also supports a persistent **volume** later if you ever add SQLite.

---

## Option D — Hugging Face Spaces (free, NO credit card)

The README's YAML front-matter (`sdk: docker`, `app_port: 8000`) makes the repo a
Docker Space. Streaming works, and no payment info is required.

1. Create a free account at <https://huggingface.co/join>.
2. Create a Space at <https://huggingface.co/new-space>:
   - **Name:** `agentic-notes`
   - **SDK:** **Docker** (blank template)
   - **Hardware:** CPU basic (free) · **Visibility:** Public
3. In the Space → **Settings → Variables and secrets**, add **secrets**:
   - `GEMINI_API_KEY` (required — failover, embeddings, image OCR, audio)
   - `NVIDIA_API_KEY` and `NVIDIA_MODEL` (optional — makes NVIDIA primary)
   - `SUPABASE_URL` and `SUPABASE_ANON_KEY` (**required** — this is what makes
     the server verify sign-ins; see "Server-side auth is not optional" above)

   And add these as **Variables** (public — the anon key is safe; it's
   protected by row-level security). They must be Variables, not Secrets,
   because Vite inlines them at *build* time:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`

   Yes, the Supabase URL and anon key go in **twice** — once as a runtime
   secret for the server, once as a build-time variable for the frontend.
   Missing the first pair is the single most common way to end up with an
   unauthenticated deployment.
4. Create a **write** access token at <https://huggingface.co/settings/tokens>.
5. From the repo root, add the Space as a remote and push (use your HF username
   and the token as the password when prompted):

   ```bash
   git remote add hf https://huggingface.co/spaces/<HF_USERNAME>/agentic-notes
   git push hf main --force
   ```

The Space builds the Dockerfile and goes live at
`https://<HF_USERNAME>-agentic-notes.hf.space`. Free Spaces sleep when idle and
wake on the next request.

## Test the production build locally (optional)

From the repo root (`agentic-notes/`):

```bash
docker build -t agentic-notes .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here agentic-notes
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
