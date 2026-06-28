# syntax=docker/dockerfile:1

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
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

ENV PORT=8000 DEV=0
EXPOSE 8000

# Single uvicorn worker keeps SSE/streaming connections simple and reliable.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
