# syntax=docker/dockerfile:1

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
# Supabase (optional): pass as build args so Vite can inline them. The anon key
# is public by design (protected by row-level security), so it's safe here.
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
COPY frontend/package*.json ./
# npm ci: install exactly what package-lock.json pins (fails if it is out of
# sync) instead of re-resolving ranges at build time.
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + bundled frontend ----
# Base images are tracked by tag, not pinned to a digest: a digest pin would
# also freeze out the base image's security updates unless someone bumps it.
FROM python:3.12-slim
WORKDIR /app

# Run as an unprivileged user. uid 1000 is what Hugging Face Spaces expects
# for Docker Spaces. The app never writes under /app at runtime (uploads are
# spooled to /tmp; eval/report.json is produced offline by eval/run_eval.py),
# so /app stays root-owned and read-only to the server process.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app
ENV PYTHONDONTWRITEBYTECODE=1

# System deps kept minimal; reportlab/pypdf/etc. are pure-Python wheels.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend source
COPY backend/ ./

# Drop the built SPA where FastAPI serves it (backend/static -> /app/static)
COPY --from=frontend /app/frontend/dist ./static

# PORT is provided by the host (Render/Railway/Fly); don't hardcode it.
ENV DEV=0
EXPOSE 8000

USER app

# Single uvicorn worker keeps SSE/streaming connections simple and reliable.
# --proxy-headers with FORWARDED_ALLOW_IPS='*' (uvicorn reads that env var
# when --forwarded-allow-ips is not given) so X-Forwarded-Proto from the
# platform's HTTPS proxy sets the scheme and redirects stay https. That makes
# uvicorn set request.client to the LEFTMOST X-Forwarded-For entry, which the
# client writes, so nothing uses request.client for identity: auth.client_ip
# takes the entry the platform proxy appended (TRUSTED_PROXY_HOPS, default 1).
# DIRECT EXPOSURE (no proxy in front): run with FORWARDED_ALLOW_IPS=127.0.0.1
# and TRUSTED_PROXY_HOPS=0.
# --timeout-graceful-shutdown is a backstop: the app already ends open streams
# on SIGTERM, but anything still open after 8s is cancelled rather than
# holding shutdown until the platform's SIGKILL (Docker's default is 10s).
ENV FORWARDED_ALLOW_IPS="*"
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --timeout-graceful-shutdown 8"]
