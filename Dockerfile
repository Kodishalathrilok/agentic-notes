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
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + bundled frontend ----
FROM python:3.12-slim
WORKDIR /app

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

# Single uvicorn worker keeps SSE/streaming connections simple and reliable.
# --proxy-headers so request.client.host is the real caller and not the
# platform's router — without it every visitor shares one rate-limit bucket.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
