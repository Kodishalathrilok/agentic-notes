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
import re
import json
import socket
import asyncio
import logging
import ipaddress
from io import BytesIO
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from auth import limiter, auth_required, verify_auth_config
from agent import (
    run_agent,
    generate_quiz,
    generate_flashcards,
    rewrite_notes,
    chat_about_notes_stream,
    edit_selection,
)
from pdf_export import notes_to_pdf, notes_to_markdown, notes_to_docx, flashcards_to_csv
from retriever import active_embedding_backend, page_spans, normalize
from models import (
    IncompleteStreamError,
    get_active_provider,
    OLLAMA_URL,
    available_models,
    default_model,
    gemini_available,
    extract_text_from_image,
    transcribe_audio,
)

# Anchored to backend/, not the cwd — see the note in models.py.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# Surface retrieval logs (index build, per-retrieval metrics, failure fallbacks)
# in the console — observability only, doesn't affect behavior.
_retrieval_logger = logging.getLogger("retrieval")
if not _retrieval_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [retrieval] %(message)s"))
    _retrieval_logger.addHandler(_h)
    _retrieval_logger.setLevel(logging.INFO)
    _retrieval_logger.propagate = False

# Fail fast if this is a production start with authentication switched off.
# Must run before the app accepts a single request.
verify_auth_config()

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

# Per-account DAILY caps on the endpoints that spend model tokens. The short
# sliding windows below stop bursts but reset forever, so one enthusiastic
# visitor could still drain a free tier's daily token quota. Tune via env.
DAILY_GENERATIONS = int(os.getenv("DAILY_GENERATIONS", "10"))
DAILY_EXTRACTS = int(os.getenv("DAILY_EXTRACTS", "40"))
DAILY_REGENS = int(os.getenv("DAILY_REGENS", "40"))
DAILY_CHATS = int(os.getenv("DAILY_CHATS", "60"))

# ---------------------------------------------------------------------------
# Server hardening: dedicated thread pool + size limits + SSRF-safe fetching
# ---------------------------------------------------------------------------

# Dedicated, bounded pool for all blocking work (model calls, PDF parsing,
# OCR). Using the loop's DEFAULT executor for this is dangerous: it's shared
# with the rest of asyncio and capped at min(32, cpus+4) threads, so a handful
# of long generations could starve every other request. Size via env.
WORKER_THREADS = int(os.getenv("WORKER_THREADS", "16"))
EXECUTOR = ThreadPoolExecutor(max_workers=WORKER_THREADS, thread_name_prefix="work")

# Hard cap on how many /api/generate pipelines may run at once. Excess
# requests get a clear 503 instead of silently queueing behind the pool.
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "4"))
_generation_slots = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

# Upload size caps (bytes). Without these, `await file.read()` loads whatever
# the client sends straight into RAM.
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "25")) * 1024 * 1024
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024

# Cap for server-side URL fetches (/api/extract-url).
MAX_URL_FETCH_BYTES = int(os.getenv("MAX_URL_FETCH_MB", "5")) * 1024 * 1024


async def _read_upload(file: UploadFile, max_bytes: int, label: str) -> bytes:
    """Read an upload in chunks, aborting with 413 once it exceeds max_bytes."""
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} is too large (max {max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _host_resolves_public(host: str) -> bool:
    """True only if EVERY address the hostname resolves to is a public IP.

    Blocks loopback (127.x, ::1), private ranges (10.x, 172.16-31.x,
    192.168.x), link-local / cloud metadata (169.254.x, fe80::), and other
    reserved space — so /api/extract-url can't be used to probe this server
    or its internal network (SSRF).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _fetch_url_safely(url: str) -> "object":
    """GET a user-supplied URL with SSRF protection and a download size cap.

    - Only http/https.
    - Every hop (including each redirect target) must resolve to public IPs.
    - Response body is streamed and truncated at MAX_URL_FETCH_BYTES.
    Returns the requests.Response with `.safe_text` attached.
    """
    import requests as _requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AgenticNotes/1.0)"}
    current = url
    for _hop in range(4):  # original request + up to 3 redirects
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(422, "Only http(s) URLs are allowed.")
        if not parsed.hostname or not _host_resolves_public(parsed.hostname):
            raise HTTPException(
                422, "This URL points at a private or unreachable address and can't be fetched."
            )

        resp = _requests.get(
            current, timeout=20, headers=headers, stream=True, allow_redirects=False
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise HTTPException(422, "URL redirected without a destination.")
            current = urljoin(current, location)
            continue

        resp.raise_for_status()
        # Stream the body with a hard byte cap.
        body, total = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > MAX_URL_FETCH_BYTES:
                resp.close()
                break
            body.append(chunk)
        raw = b"".join(body)
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        try:
            resp.safe_text = raw.decode(encoding, errors="replace")
        except LookupError:
            resp.safe_text = raw.decode("utf-8", errors="replace")
        return resp

    raise HTTPException(422, "Too many redirects.")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: str = Field(default="")  # length checked against MAX_TEXT_CHARS below
    mode: str = Field(default="exam", max_length=40)
    tone: str = Field(default="academic", max_length=40)
    length: str = Field(default="medium", max_length=40)
    format: str = Field(default="bullet", max_length=40)
    model: str = Field(default="", max_length=100)
    instructions: str = Field(default="", max_length=5000)
    # Quiz and flashcards are generated on demand from the UI (separate
    # /api/quiz and /api/flashcards calls), not with the notes.
    include_quiz: bool = Field(default=False)
    include_flashcards: bool = Field(default=False)
    # [{page, start, end}] from /api/extract-pdf, so citations can name the
    # page they came from. Absent for pasted text, URLs and transcripts,
    # which have no pages — those simply cite without one.
    page_spans: list = Field(default_factory=list, max_length=5000)


class ChatRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    question: str = Field(default="", max_length=4000)
    history: list = Field(default_factory=list, max_length=24)
    model: str = Field(default="", max_length=100)


class UrlRequest(BaseModel):
    url: str = Field(default="", max_length=2000)


class EditSelectionRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    selection: str = Field(default="", max_length=20000)
    instruction: str = Field(default="", max_length=2000)
    model: str = Field(default="", max_length=100)


class ExportRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    quiz: str = Field(default="", max_length=100000)
    flashcards: str = Field(default="", max_length=100000)


class RegenRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    model: str = Field(default="", max_length=100)


class RewriteRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    direction: str = Field(default="shorter", max_length=20)  # shorter | longer | clarity
    mode: str = Field(default="exam", max_length=40)
    tone: str = Field(default="academic", max_length=40)
    format: str = Field(default="bullet", max_length=40)
    model: str = Field(default="", max_length=100)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": get_active_provider(),
        # Gemini backs image OCR and audio transcription, so this is what
        # tells you whether those two endpoints will work at all.
        "gemini_configured": gemini_available(),
        "ollama_url": OLLAMA_URL,
        "embeddings": active_embedding_backend(),
        # So a misconfigured deploy is visible at a glance instead of only
        # discoverable by noticing that unauthenticated calls succeed.
        "auth_required": auth_required(),
        # The cap /api/generate enforces. Published so the browser can hold a
        # source to the same limit at upload time — it used to carry its own
        # hard-coded 300000 and only discovered the real one by having Generate
        # rejected, after the user had already picked a file and waited.
        "max_text_chars": MAX_TEXT_CHARS,
    }


@app.get("/api/eval-report")
async def eval_report():
    """Serve the latest eval report (eval/report.json) for the dashboard."""
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "eval", "report.json")
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
async def generate(req: GenerateRequest, user=Depends(limiter("generate", 6, 600, daily=DAILY_GENERATIONS))):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"`text` exceeds the {MAX_TEXT_CHARS} character limit.",
        )

    # Reject immediately (don't queue invisibly) when the server is already
    # running its maximum number of pipelines.
    if _generation_slots.locked():
        raise HTTPException(
            status_code=503,
            detail="The server is at capacity right now — please try again in a minute.",
        )

    async def event_generator():
      async with _generation_slots:
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
            include_quiz=req.include_quiz,
            include_flashcards=req.include_flashcards,
            page_spans=req.page_spans,
        )
        sentinel = object()

        def _next():
            return next(gen, sentinel)

        while True:
            event = await loop.run_in_executor(EXECUTOR, _next)
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
async def extract_pdf(file: UploadFile = File(...), user=Depends(limiter("extract", 20, 600, daily=DAILY_EXTRACTS))):
    from pypdf import PdfReader

    raw = await _read_upload(file, MAX_PDF_BYTES, "PDF")
    def _parse() -> tuple:
        reader = PdfReader(BytesIO(raw))
        n_pages = len(reader.pages)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                parts.append("")
        joined = "\n\n".join(c.strip() for c in parts if c.strip()).strip()
        # Page boundaries measured in the same normalized space the chunker
        # uses, so a citation can later be traced back to the page it came
        # from. Computed here because this is the only place page structure
        # still exists — joining throws it away.
        return n_pages, joined, page_spans(parts)

    loop = asyncio.get_event_loop()
    try:
        pages, text, spans = await loop.run_in_executor(EXECUTOR, _parse)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"Could not read PDF: {exc}")

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF (it may be scanned/image-only).",
        )

    # Never hand back more than /api/generate will accept. extract-url and the
    # transcript path already clamp here; this one didn't, so a PDF over the
    # limit extracted fine, reported "N words ready", and then failed at
    # Generate with a raw 422 — the worst possible moment to find out.
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        # Cut on whitespace so the last word survives intact.
        head = text[:MAX_TEXT_CHARS]
        cut = head.rfind(" ")
        text = head[:cut] if cut > MAX_TEXT_CHARS * 0.9 else head
        # Spans are measured in normalized space (see page_spans), so clamp
        # them against the normalized length of what's left. Pages past the
        # cut have to go: a citation must not name a page whose text the
        # model was never given.
        limit = len(normalize(text))
        spans = [
            {**s, "end": min(s["end"], limit)}
            for s in spans
            if s["start"] < limit
        ]

    return {
        "text": text,
        "pages": pages,
        "page_spans": spans,
        "truncated": truncated,
        "max_chars": MAX_TEXT_CHARS,
    }


# ---------------------------------------------------------------------------
# Image OCR and audio transcription (Gemini multimodal)
# ---------------------------------------------------------------------------

@app.post("/api/extract-image")
async def extract_image(file: UploadFile = File(...), user=Depends(limiter("extract", 20, 600, daily=DAILY_EXTRACTS))):
    """OCR a photo of study material (textbook page, slides, handwriting) via Gemini vision."""
    if not gemini_available():
        raise HTTPException(
            status_code=503,
            detail="Image text extraction requires a GEMINI_API_KEY. Set it in backend/.env.",
        )

    raw = await _read_upload(file, MAX_IMAGE_BYTES, "Image")
    mime = file.content_type or "image/jpeg"
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(EXECUTOR, lambda: extract_text_from_image(raw, mime))
    except Exception:  # noqa: BLE001 - message kept generic so the API key never leaks
        raise HTTPException(
            status_code=502,
            detail="Couldn't read text from the image (the vision API may be rate-limited). "
            "Please try again in a moment.",
        )

    text = (text or "").strip()[:MAX_TEXT_CHARS]
    if len(text) < 5:
        raise HTTPException(
            status_code=422, detail="No readable text found in this image.")
    return {"text": text, "chars": len(text)}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), user=Depends(limiter("extract", 20, 600, daily=DAILY_EXTRACTS))):
    """Transcribe a lecture recording or voice memo via Gemini."""
    if not gemini_available():
        raise HTTPException(
            status_code=503,
            detail="Audio transcription requires a GEMINI_API_KEY. Set it in backend/.env.",
        )

    raw = await _read_upload(file, MAX_AUDIO_BYTES, "Audio file")
    mime = file.content_type or "audio/webm"

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(EXECUTOR, lambda: transcribe_audio(raw, mime))
    except Exception:  # noqa: BLE001 - message kept generic so the API key never leaks
        raise HTTPException(
            status_code=502,
            detail="Transcription failed (the audio API may be rate-limited). "
            "Please try again in a moment.",
        )

    text = (text or "").strip()[:MAX_TEXT_CHARS]
    if not text:
        raise HTTPException(
            status_code=422, detail="No speech was found in this recording.")
    return {"text": text}


# ---------------------------------------------------------------------------
# Regenerate individual sections
# ---------------------------------------------------------------------------

@app.post("/api/quiz")
async def regen_quiz(req: RegenRequest, user=Depends(limiter("regen", 20, 600, daily=DAILY_REGENS))):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_event_loop()
    quiz = await loop.run_in_executor(EXECUTOR, lambda: generate_quiz(notes, 5, req.model))
    return {"quiz": quiz}


@app.post("/api/flashcards")
async def regen_flashcards(req: RegenRequest, user=Depends(limiter("regen", 20, 600, daily=DAILY_REGENS))):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_event_loop()
    cards = await loop.run_in_executor(EXECUTOR, lambda: generate_flashcards(notes, 8, req.model))
    return {"flashcards": cards}


@app.post("/api/edit-selection")
async def edit_selection_endpoint(req: EditSelectionRequest, user=Depends(limiter("regen", 20, 600, daily=DAILY_REGENS))):
    notes = (req.notes or "").strip()
    selection = (req.selection or "").strip()
    if not notes or not selection:
        raise HTTPException(
            status_code=422, detail="`notes` and `selection` are required.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        EXECUTOR, lambda: edit_selection(
            notes, selection, req.instruction, req.model)
    )
    return {"notes": result}


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest, user=Depends(limiter("regen", 20, 600, daily=DAILY_REGENS))):
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    if req.direction not in ("shorter", "longer", "clarity"):
        raise HTTPException(status_code=422, detail="Invalid `direction`.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        EXECUTOR,
        lambda: rewrite_notes(notes, req.direction, req.mode,
                              req.tone, req.format, req.model),
    )
    return {"notes": result}


# ---------------------------------------------------------------------------
# Chat with your notes (streaming plain text)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest, user=Depends(limiter("chat", 40, 600, daily=DAILY_CHATS))):
    notes = (req.notes or "").strip()
    question = (req.question or "").strip()
    if not notes:
        raise HTTPException(
            status_code=422, detail="Generate notes first to chat about them.")
    if not question:
        raise HTTPException(status_code=422, detail="`question` is required.")

    async def token_generator():
        loop = asyncio.get_event_loop()
        gen = chat_about_notes_stream(
            notes, question, req.history, model=req.model)
        sentinel = object()

        def _next():
            return next(gen, sentinel)

        while True:
            try:
                piece = await loop.run_in_executor(EXECUTOR, _next)
            except IncompleteStreamError:
                # The partial answer is already on screen; say so rather than
                # let it pass for the whole answer.
                yield "\n\n_[The answer was cut off \u2014 please ask again.]_"
                break
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


# Why a transcript fetch failed, in words a user can act on. Keyed by the
# library's exception class name so we don't parse its (very long) messages.
_YT_REASONS = {
    "TranscriptsDisabled": "This video has captions turned off.",
    "NoTranscriptFound": "This video has no English captions.",
    "NoTranscriptAvailable": "This video has no captions at all.",
    "VideoUnavailable": "This video isn't available — it may be private, deleted, or region-locked.",
    "VideoUnplayable": "YouTube won't play this video, so its captions can't be read.",
    "AgeRestricted": "This video is age-restricted, so its captions can't be read.",
    "IpBlocked": "YouTube is blocking requests from this server's network.",
    "RequestBlocked": "YouTube is blocking requests from this server's network.",
}


def _youtube_failure_reason(exc: Exception) -> str:
    return _YT_REASONS.get(
        type(exc).__name__,
        "Couldn't read this video's captions.",
    )


def _youtube_transcript(video_id: str) -> str:
    """Fetch a video's caption text.

    youtube-transcript-api >= 1.0 replaced the YouTubeTranscriptApi
    .get_transcript() / .list_transcripts() CLASSMETHODS with instance
    .fetch() / .list(). The previous fallback here still called the
    classmethods, so on any failure it raised
    `AttributeError: no attribute 'list_transcripts'` — burying the real
    reason (captions disabled, video unavailable, IP blocked) under a
    meaningless one. Let the library's own exception propagate instead;
    the caller turns it into a message that says what actually happened.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = ["en", "en-US", "en-GB"]
    api = YouTubeTranscriptApi()

    if hasattr(api, "fetch"):  # >= 1.0
        try:
            fetched = api.fetch(video_id, languages=langs)
        except Exception:
            # No English track — fall back to whatever the video does have,
            # but let a hard failure (unavailable/blocked) surface as itself.
            fetched = next(iter(api.list(video_id))).fetch()
        return " ".join(_seg_text(s) for s in fetched if _seg_text(s))

    # Legacy 0.6.x
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    except Exception:
        segments = next(iter(YouTubeTranscriptApi.list_transcripts(video_id))).fetch()
    return " ".join(_seg_text(s) for s in segments if _seg_text(s))


@app.post("/api/extract-url")
async def extract_url(req: UrlRequest, user=Depends(limiter("extract", 20, 600, daily=DAILY_EXTRACTS))):
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
                detail=f"{_youtube_failure_reason(exc)} Try another video, or "
                "paste the text in directly.",
            )
        if len(transcript) < 50:
            raise HTTPException(
                status_code=422, detail="This video has no usable transcript.")
        return {
            "text": transcript[:MAX_TEXT_CHARS],
            "title": "YouTube transcript",
            "chars": min(len(transcript), MAX_TEXT_CHARS),
        }

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(EXECUTOR, lambda: _fetch_url_safely(url))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"Could not fetch URL: {exc}")

    soup = BeautifulSoup(resp.safe_text, "html.parser")
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
        raise HTTPException(
            status_code=422, detail="No readable text found at that URL.")

    title = soup.title.get_text(strip=True) if soup.title else url
    return {"text": text, "title": title, "chars": len(text)}


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

@app.post("/api/export/pdf")
async def export_pdf(req: ExportRequest, user=Depends(limiter("export", 30, 600))):
    pdf_bytes = notes_to_pdf(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=notes.pdf"},
    )


@app.post("/api/export/markdown")
async def export_markdown(req: ExportRequest, user=Depends(limiter("export", 30, 600))):
    md = notes_to_markdown(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(md.encode("utf-8")),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=notes.md"},
    )


@app.post("/api/export/docx")
async def export_docx(req: ExportRequest, user=Depends(limiter("export", 30, 600))):
    data = notes_to_docx(req.notes, req.quiz, req.flashcards)
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=notes.docx"},
    )


@app.post("/api/export/flashcards-csv")
async def export_flashcards_csv(req: ExportRequest, user=Depends(limiter("export", 30, 600))):
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

# Starlette's StaticFiles sets ETag and Last-Modified but never Cache-Control,
# so browsers fall back to HEURISTIC caching and can keep serving an old
# index.html long after a deploy. A client pinned that way goes on running the
# previous build's entry bundle and asks for lazy chunk names that no longer
# exist on the server -- the QuizPanel-*.js / FlashcardPanel-*.js 404s seen in
# production, with the entry bundle "loading fine" because it came from cache.
#
# Vite gives the two kinds of file opposite guarantees, so they get opposite
# policies. Anything it emits into assets/ carries a content hash: the name
# changes whenever the bytes do, so it can be cached forever. Files copied
# through from public/ keep their name across builds, so they must revalidate.
# index.html is the entry point and must never be cached, or none of the rest
# matters -- it is what names every hashed file.
_HASHED_ASSET_RE = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")

CACHE_ENTRY_HTML = "no-cache, no-store, must-revalidate"
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_REVALIDATE = "public, max-age=0, must-revalidate"


def cache_policy(full_path: str) -> str:
    """The Cache-Control for one served file, decided by how it is named."""
    name = os.path.basename(full_path)
    if name == "index.html":
        return CACHE_ENTRY_HTML
    parts = os.path.normpath(full_path).replace("\\", "/").split("/")
    if "assets" in parts and _HASHED_ASSET_RE.search(name):
        return CACHE_IMMUTABLE
    return CACHE_REVALIDATE


class CachedStaticFiles(StaticFiles):
    """StaticFiles that states its caching intent instead of leaving it to the
    browser's heuristics.

    The header is applied after super() so it lands on 304 Not Modified too --
    a revalidation response that omits Cache-Control drops the client straight
    back onto heuristics, which is the behaviour being fixed.
    """

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["cache-control"] = cache_policy(str(full_path))
        return response


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", CachedStaticFiles(directory=STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    # Auto-reload only for local dev; disable in production via DEV=0.
    reload = os.getenv("DEV", "1").lower() in ("1", "true", "yes")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)