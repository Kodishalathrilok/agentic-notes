"""
FastAPI server for the Agentic AI Notes Generator.

Endpoints:
    GET  /api/health
    POST /api/generate          (SSE stream)
    POST /api/extract-pdf       (multipart)
    POST /api/transcribe        (multipart)
    POST /api/export/pdf
    POST /api/export/markdown
"""

import os
import json
import asyncio
import logging
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from agent import (
    run_agent,
    generate_quiz,
    generate_flashcards,
    rewrite_notes,
    chat_about_notes_stream,
    edit_selection,
)
from pdf_export import notes_to_pdf, notes_to_markdown, notes_to_docx, flashcards_to_csv
from retriever import active_embedding_backend
from models import (
    get_active_provider,
    OLLAMA_URL,
    GROQ_API_KEY,
    GROQ_PLACEHOLDER,
    available_models,
    default_model,
    gemini_available,
    extract_text_from_image,
)

load_dotenv()

# Surface retrieval logs (index build, per-retrieval metrics, failure fallbacks)
# in the console — observability only, doesn't affect behavior.
_retrieval_logger = logging.getLogger("retrieval")
if not _retrieval_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [retrieval] %(message)s"))
    _retrieval_logger.addHandler(_h)
    _retrieval_logger.setLevel(logging.INFO)
    _retrieval_logger.propagate = False

app = FastAPI(title="Agentic AI Notes Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "300000"))


def _groq_configured() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != GROQ_PLACEHOLDER


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: str = Field(default="")
    mode: str = Field(default="exam")
    tone: str = Field(default="academic")
    length: str = Field(default="medium")
    format: str = Field(default="bullet")
    model: str = Field(default="")
    instructions: str = Field(default="")


class ChatRequest(BaseModel):
    notes: str = Field(default="")
    question: str = Field(default="")
    history: list = Field(default_factory=list)
    model: str = Field(default="")


class UrlRequest(BaseModel):
    url: str = Field(default="")


class EditSelectionRequest(BaseModel):
    notes: str = Field(default="")
    selection: str = Field(default="")
    instruction: str = Field(default="")
    model: str = Field(default="")


class ExportRequest(BaseModel):
    notes: str = Field(default="")
    quiz: str = Field(default="")
    flashcards: str = Field(default="")


class RegenRequest(BaseModel):
    notes: str = Field(default="")
    model: str = Field(default="")


class RewriteRequest(BaseModel):
    notes: str = Field(default="")
    direction: str = Field(default="shorter")  # shorter | longer
    mode: str = Field(default="exam")
    tone: str = Field(default="academic")
    format: str = Field(default="bullet")
    model: str = Field(default="")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": get_active_provider(),
        "groq_configured": _groq_configured(),
        "ollama_url": OLLAMA_URL,
        "embeddings": active_embedding_backend(),
    }


@app.get("/api/eval-report")
async def eval_report():
    """Serve the latest eval report (eval/report.json) for the dashboard."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "report.json")
    if not os.path.isfile(path):
        return {"available": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"available": True, "report": json.load(f)}
    except Exception:  # noqa: BLE001
        return {"available": False}


@app.get("/api/models")
async def list_models():
    """Available models for the UI picker, across configured providers."""
    return {
        "provider": get_active_provider(),
        "models": available_models(),
        "default": default_model(),
    }


# ---------------------------------------------------------------------------
# Generate (SSE)
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"`text` exceeds the {MAX_TEXT_CHARS} character limit.",
        )

    async def event_generator():
        loop = asyncio.get_event_loop()

        # run_agent is a synchronous generator; step through it in an executor
        # so each blocking model call doesn't stall the event loop.
        gen = run_agent(
            text,
            req.mode,
            req.tone,
            req.length,
            req.format,
            model=req.model,
            instructions=req.instructions,
        )
        sentinel = object()

        def _next():
            return next(gen, sentinel)

        while True:
            event = await loop.run_in_executor(None, _next)
            if event is sentinel:
                break
            yield {"data": json.dumps(event)}
            # Pace discrete step events so the frontend can render each one,
            # but stream token deltas as fast as they arrive.
            if event.get("type") != "notes_delta":
                await asyncio.sleep(0.05)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    from pypdf import PdfReader

    raw = await file.read()
    try:
        reader = PdfReader(BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")

    pages = len(reader.pages)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            chunks.append("")

    text = "\n\n".join(c.strip() for c in chunks if c.strip()).strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF (it may be scanned/image-only).",
        )

    return {"text": text, "pages": pages}


# ---------------------------------------------------------------------------
# Audio transcription (Groq Whisper)
# ---------------------------------------------------------------------------

@app.post("/api/extract-image")
async def extract_image(file: UploadFile = File(...)):
    """OCR a photo of study material (textbook page, slides, handwriting) via Gemini vision."""
    if not gemini_available():
        raise HTTPException(
            status_code=503,
            detail="Image text extraction requires a GEMINI_API_KEY. Set it in backend/.env.",
        )

    raw = await file.read()
    mime = file.content_type or "image/jpeg"
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, lambda: extract_text_from_image(raw, mime))
    except Exception:  # noqa: BLE001 - message kept generic so the API key never leaks
        raise HTTPException(
            status_code=502,
            detail="Couldn't read text from the image (the vision API may be rate-limited). "
            "Please try again in a moment.",
        )

    text = (text or "").strip()[:MAX_TEXT_CHARS]
    if len(text) < 5:
        raise HTTPException(status_code=422, detail="No readable text found in this image.")
    return {"text": text, "chars": len(text)}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not _groq_configured():
        raise HTTPException(
            status_code=503,
            detail="Audio transcription requires a GROQ_API_KEY (Whisper). "
            "Set it in backend/.env.",
        )

    from groq import Groq

    raw = await file.read()
    filename = file.filename or "audio.webm"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        result = client.audio.transcriptions.create(
            file=(filename, raw),
            model="whisper-large-v3",
        )
        text = getattr(result, "text", "") or ""
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}")

    return {"text": text.strip()}


# ---------------------------------------------------------------------------
# Regenerate individual sections
# ---------------------------------------------------------------------------

@app.post("/api/quiz")
async def regen_quiz(req: RegenRequest):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_event_loop()
    quiz = await loop.run_in_executor(None, lambda: generate_quiz(notes, 5, req.model))
    return {"quiz": quiz}


@app.post("/api/flashcards")
async def regen_flashcards(req: RegenRequest):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_event_loop()
    cards = await loop.run_in_executor(None, lambda: generate_flashcards(notes, 8, req.model))
    return {"flashcards": cards}


@app.post("/api/edit-selection")
async def edit_selection_endpoint(req: EditSelectionRequest):
    notes = (req.notes or "").strip()
    selection = (req.selection or "").strip()
    if not notes or not selection:
        raise HTTPException(status_code=422, detail="`notes` and `selection` are required.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: edit_selection(notes, selection, req.instruction, req.model)
    )
    return {"notes": result}


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    if req.direction not in ("shorter", "longer", "clarity"):
        raise HTTPException(status_code=422, detail="Invalid `direction`.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rewrite_notes(notes, req.direction, req.mode, req.tone, req.format, req.model),
    )
    return {"notes": result}


# ---------------------------------------------------------------------------
# Chat with your notes (streaming plain text)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    notes = (req.notes or "").strip()
    question = (req.question or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="Generate notes first to chat about them.")
    if not question:
        raise HTTPException(status_code=422, detail="`question` is required.")

    async def token_generator():
        loop = asyncio.get_event_loop()
        gen = chat_about_notes_stream(notes, question, req.history, model=req.model)
        sentinel = object()

        def _next():
            return next(gen, sentinel)

        while True:
            piece = await loop.run_in_executor(None, _next)
            if piece is sentinel:
                break
            yield piece

    return StreamingResponse(token_generator(), media_type="text/plain")


# ---------------------------------------------------------------------------
# URL ingestion (extract readable article text from a web page)
# ---------------------------------------------------------------------------

def _youtube_id(url: str):
    """Return the video id if this is a YouTube URL, else None."""
    import re as _re

    patterns = [
        r"(?:youtube\.com/watch\?(?:.*&)?v=)([\w-]{11})",
        r"(?:youtu\.be/)([\w-]{11})",
        r"(?:youtube\.com/(?:embed|shorts|live)/)([\w-]{11})",
    ]
    for p in patterns:
        m = _re.search(p, url)
        if m:
            return m.group(1)
    return None


def _seg_text(seg) -> str:
    """Extract text from a transcript segment across library versions."""
    if isinstance(seg, dict):
        return seg.get("text", "")
    return getattr(seg, "text", "")


def _youtube_transcript(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = ["en", "en-US", "en-GB"]

    # New API (>= 1.0): instance-based .fetch() / .list()
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            try:
                fetched = api.fetch(video_id, languages=langs)
            except Exception:
                # any available transcript
                listing = api.list(video_id)
                fetched = next(iter(listing)).fetch()
            return " ".join(_seg_text(s) for s in fetched if _seg_text(s))
    except Exception:
        pass

    # Old API (0.6.x): classmethods
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    except Exception:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        segments = next(iter(listing)).fetch()
    return " ".join(_seg_text(s) for s in segments if _seg_text(s))


@app.post("/api/extract-url")
async def extract_url(req: UrlRequest):
    import requests as _requests
    from bs4 import BeautifulSoup

    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="`url` is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # YouTube: fetch the caption transcript (the real content) instead of HTML.
    yt_id = _youtube_id(url)
    if yt_id:
        try:
            transcript = _youtube_transcript(yt_id).strip()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail="Couldn't get a transcript for this video (it may have "
                f"captions disabled). Try a video with subtitles. ({exc})",
            )
        if len(transcript) < 50:
            raise HTTPException(status_code=422, detail="This video has no usable transcript.")
        return {
            "text": transcript[:MAX_TEXT_CHARS],
            "title": "YouTube transcript",
            "chars": min(len(transcript), MAX_TEXT_CHARS),
        }

    try:
        resp = _requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AgenticNotes/1.0)"},
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not fetch URL: {exc}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Prefer <article> / <main> if present, else the whole body.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    parts = []
    for el in container.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if t and len(t) > 1:
            parts.append(t)

    text = "\n".join(parts).strip()
    if len(text) < 50:
        # fallback: all visible text
        text = soup.get_text("\n", strip=True)

    text = text[:MAX_TEXT_CHARS]
    if not text:
        raise HTTPException(status_code=422, detail="No readable text found at that URL.")

    title = soup.title.get_text(strip=True) if soup.title else url
    return {"text": text, "title": title, "chars": len(text)}


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

@app.post("/api/export/pdf")
async def export_pdf(req: ExportRequest):
    pdf_bytes = notes_to_pdf(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=notes.pdf"},
    )


@app.post("/api/export/markdown")
async def export_markdown(req: ExportRequest):
    md = notes_to_markdown(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(md.encode("utf-8")),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=notes.md"},
    )


@app.post("/api/export/docx")
async def export_docx(req: ExportRequest):
    data = notes_to_docx(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=notes.docx"},
    )


@app.post("/api/export/flashcards-csv")
async def export_flashcards_csv(req: ExportRequest):
    csv_text = flashcards_to_csv(req.flashcards)
    return StreamingResponse(
        BytesIO(csv_text.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=flashcards.csv"},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Serve the built frontend (single-service deployment).
# Mounted LAST so all /api/* routes take precedence. Only activates when a
# build is present, so local API-only development still works unchanged.
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    # Auto-reload only for local dev; disable in production via DEV=0.
    reload = os.getenv("DEV", "1").lower() in ("1", "true", "yes")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
