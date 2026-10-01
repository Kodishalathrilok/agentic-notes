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
import sys
import signal
import socket
import time
import asyncio
import logging
import threading
import ipaddress
from io import BytesIO
from typing import Any, Literal, Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urljoin
from concurrent.futures import Future, ThreadPoolExecutor

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
    verify_quiz_detailed,
    generate_flashcards,
    rewrite_notes,
    chat_about_notes_stream,
    edit_selection,
    CHAT_HISTORY_TURNS,
    SelectionNotFoundError,
    SelectionAmbiguousError,
    RewriteTooLongError,
)
from pdf_export import notes_to_pdf, notes_to_markdown, notes_to_docx, flashcards_to_csv
from retriever import active_embedding_backend, page_spans, normalize, valid_page_spans
from models import (
    IncompleteStreamError,
    UserFacingError,
    ProvidersUnavailableError,
    get_active_provider,
    available_models,
    default_model,
    gemini_available,
    extract_text_from_image,
    transcribe_audio,
    validate_model_id,
    UnknownModelError,
    log_unexpected_error,
    PipelineCancelled,
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

# Server-side record of unexpected errors (see models.log_unexpected_error):
# clients get a short error id, the detail lands here.
_agentic_logger = logging.getLogger("agentic")
if not _agentic_logger.handlers:
    _ah = logging.StreamHandler()
    _ah.setFormatter(logging.Formatter("%(levelname)s [agentic] %(message)s"))
    _agentic_logger.addHandler(_ah)
    _agentic_logger.setLevel(logging.INFO)
    _agentic_logger.propagate = False

# Fail fast if this is a production start with authentication switched off.
# Must run before the app accepts a single request.
verify_auth_config()

@asynccontextmanager
async def _lifespan(_app):
    restore = _hook_shutdown_signals()
    try:
        yield
    finally:
        for sig, handler in restore.items():
            signal.signal(sig, handler)
        # Shutdown: stop every open pipeline/chat stream and its model calls.
        # See _shutdown_streams.
        await _shutdown_streams()


app = FastAPI(title="Agentic AI Notes Generator", version="1.0.0", lifespan=_lifespan)


class _BodyTooLarge(HTTPException):
    """Raised from inside `receive` once a body passes its route's limit. An
    HTTPException, so FastAPI's body parsing re-raises it untouched (it turns
    other exceptions into a 400) and the app's handler answers 413."""

    def __init__(self, limit: int):
        super().__init__(status_code=413,
                         detail=f"Request body too large (max {limit // 1024} KB).")


# Multipart overhead allowed on top of an upload route's file cap (boundaries,
# part headers, the filename).
_MULTIPART_OVERHEAD = 1024 * 1024


def _body_limit(path: str) -> int:
    """Largest request body `path` accepts. Upload routes get their file cap
    plus multipart overhead; every other route takes JSON (MAX_JSON_BODY_BYTES).
    Looked up per request because the MAX_* constants are defined below."""
    uploads = {
        "/api/extract-pdf": MAX_PDF_BYTES,
        "/api/transcribe": MAX_AUDIO_BYTES,
        "/api/extract-image": MAX_IMAGE_BYTES,
    }
    cap = uploads.get(path.rstrip("/"))
    return cap + _MULTIPART_OVERHEAD if cap is not None else MAX_JSON_BODY_BYTES


class BodySizeLimitMiddleware:
    """Reject oversized request bodies BEFORE they are read into memory.

    Starlette buffers a whole JSON body (and spools a whole multipart body)
    before any handler, dependency or auth check runs, so without this an
    unauthenticated client could make the server hold hundreds of MB.

    - A declared Content-Length over the route's limit is answered 413 at
      once, without reading a byte of the body.
    - A body without Content-Length (chunked) is counted as it streams in and
      aborted with 413 as soon as it passes the limit.
    Pure ASGI (not BaseHTTPMiddleware), so streaming responses (SSE, chat) and
    static files pass through untouched; only request-body messages are
    inspected.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        limit = _body_limit(scope.get("path") or "")
        declared = None
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break
        if declared is not None and declared > limit:
            await self._reject(send, limit)
            return

        received = 0
        started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge(limit)
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            # Normally the app's exception handler has already answered; this
            # covers a body read outside it.
            if started:
                raise
            await self._reject(send, limit)

    @staticmethod
    async def _reject(send, limit: int) -> None:
        body = json.dumps({"detail": _BodyTooLarge(limit).detail}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})


# Added BEFORE CORSMiddleware so CORS wraps it and a 413 still carries the
# CORS headers the browser needs to read it.
app.add_middleware(BodySizeLimitMiddleware)

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

# Largest JSON request body accepted (see BodySizeLimitMiddleware). The
# biggest legitimate bodies: /api/chat (notes 300k chars + the conversation
# so far; ChatPanel answers are capped at 900 tokens, so even 100 exchanges
# stay well under 1M chars), /api/export/* (300k + 100k + 100k chars) and /api/generate
# (MAX_TEXT_CHARS + up to 5000 page spans at ~45 bytes). At up to 3 UTF-8
# bytes per character (all of the BMP) that tops out around 2.3 MB, so the
# default is 2.5 MiB - or 8 bytes per character if MAX_TEXT_CHARS is raised.
MAX_JSON_BODY_BYTES = int(os.getenv(
    "MAX_JSON_BODY_BYTES", str(max(5 * 1024 * 1024 // 2, 8 * MAX_TEXT_CHARS))))

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
#
# Hard cap on how many /api/generate pipelines may run at once. Excess
# requests get a clear 503 instead of silently queueing behind the pool.
# Each pipeline writes up to agent.SECTION_CONCURRENCY sections at once, so the
# server-wide ceiling on concurrent section streams is
# MAX_CONCURRENT_GENERATIONS x SECTION_CONCURRENCY.
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "4"))

# SIZING RULE for EXECUTOR. A generate pipeline holds ONE EXECUTOR thread at a
# time (its current step; the section writers run on the pipeline's own pool
# in agent.py, not here). Every open chat stream holds one, and so does every
# in-flight quiz/flashcards/rewrite/edit/extract call. So the pool must be at
# least MAX_CONCURRENT_GENERATIONS plus headroom for those, or full generation
# capacity alone could queue everyone's chat and extract work. Default:
# generation slots + 12, never below 16. Exports render on EXPORT_EXECUTOR.
_EXECUTOR_HEADROOM = 12
WORKER_THREADS = int(os.getenv(
    "WORKER_THREADS", str(max(16, MAX_CONCURRENT_GENERATIONS + _EXECUTOR_HEADROOM))))
if WORKER_THREADS < MAX_CONCURRENT_GENERATIONS + 2:
    _agentic_logger.warning(
        "WORKER_THREADS=%d leaves little room beyond MAX_CONCURRENT_GENERATIONS=%d; "
        "chat and extract requests may queue behind generations",
        WORKER_THREADS, MAX_CONCURRENT_GENERATIONS)
EXECUTOR = ThreadPoolExecutor(max_workers=WORKER_THREADS, thread_name_prefix="work")

# Exports (PDF/DOCX/Markdown/CSV rendering, CPU-bound for up to seconds on a
# 300k-char document) get their own small pool so a burst of them can only
# queue behind each other, never occupy EXECUTOR and stall generations/chats.
EXPORT_THREADS = max(1, int(os.getenv("EXPORT_THREADS", "2")))
EXPORT_EXECUTOR = ThreadPoolExecutor(max_workers=EXPORT_THREADS, thread_name_prefix="export")

# End-to-end budget for one /api/generate stream. When it runs out the
# pipeline is cancelled and the client gets one final, user-safe error event.
GENERATION_DEADLINE_S = float(os.getenv("GENERATION_DEADLINE_S", "900"))

# How long shutdown waits for open streams to stop before giving up on them.
SHUTDOWN_GRACE_S = 5.0

GENERATION_TIMEOUT_MESSAGE = (
    "Generating your notes took too long and was stopped. Please try again - "
    "a shorter source or length usually finishes sooner."
)


class _SlotCounter:
    """Thread-safe, non-blocking capacity counter.

    Replaces an asyncio.Semaphore whose `locked()` pre-check and later
    `async with` were two separate steps: several requests could pass the
    check together and then queue invisibly inside the stream. try_acquire()
    checks and takes a slot in one step, synchronously in the handler, so a
    request is either admitted with its slot or rejected with a 503.
    """

    def __init__(self, limit: int):
        self.limit = max(0, int(limit))  # 0 rejects everything, as Semaphore(0) did
        self._in_use = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_use >= self.limit:
                return False
            self._in_use += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_use <= 0:
                raise RuntimeError("generation slot released more times than acquired")
            self._in_use -= 1

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use


_generation_slots = _SlotCounter(MAX_CONCURRENT_GENERATIONS)

# Per-account cap on EXPENSIVE requests in flight at once (generate, chat,
# quiz/flashcards/rewrite/edit, the extract endpoints and exports). Each one holds an
# EXECUTOR thread for seconds to minutes, and the rate limits above count
# requests per window, not concurrency: without this, one account could open
# enough parallel calls to occupy every worker thread and stall everyone else.
MAX_INFLIGHT_PER_USER = int(os.getenv("MAX_INFLIGHT_PER_USER", "3"))

INFLIGHT_LIMIT_MESSAGE = (
    "You already have several requests running - please wait for one to "
    "finish and try again."
)


class _UserInflight:
    """Thread-safe per-identity in-flight counter (same shape as _SlotCounter)."""

    def __init__(self, limit: int):
        self.limit = max(0, int(limit))
        self._counts: dict = {}
        self._lock = threading.Lock()

    def try_acquire(self, identity: str) -> bool:
        with self._lock:
            n = self._counts.get(identity, 0)
            if n >= self.limit:
                return False
            self._counts[identity] = n + 1
            return True

    def release(self, identity: str) -> None:
        with self._lock:
            n = self._counts.get(identity, 0)
            if n <= 1:
                self._counts.pop(identity, None)  # no key left behind per identity
            else:
                self._counts[identity] = n - 1

    def in_use(self, identity: str) -> int:
        with self._lock:
            return self._counts.get(identity, 0)


_user_inflight = _UserInflight(MAX_INFLIGHT_PER_USER)


class _UserSlot:
    """One held per-user slot; release() is idempotent. Passed to _StreamRun so
    a stream gives its slot back in the same cleanup that releases the rest."""

    def __init__(self, identity: str):
        self.identity = identity
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            _user_inflight.release(self.identity)


def _take_user_slot(user) -> _UserSlot:
    """Take one of this account's in-flight slots, or raise 429."""
    identity = str(user.get("id") or "") if isinstance(user, dict) else ""
    if not _user_inflight.try_acquire(identity):
        raise HTTPException(status_code=429, detail=INFLIGHT_LIMIT_MESSAGE)
    return _UserSlot(identity)


def _inflight_guard(dep):
    """Wrap a limiter dependency so the request also holds a per-user slot
    until the handler has finished (non-streaming endpoints only - a stream
    hands its slot to _StreamRun instead)."""
    async def _guard(user=Depends(dep)):
        slot = _take_user_slot(user)
        try:
            yield user
        finally:
            slot.release()
    return _guard

_END = object()        # the stepped generator is exhausted (or stopped)
_TERMINAL_EVENTS = {"done", "error", "blocked"}  # run_agent ends after any of these
_TIMED_OUT = object()  # a step did not finish within the time allowed

# Every open generate/chat stream, until its cleanup has finished.
_active_runs: "set[_StreamRun]" = set()


def _consume_outcome(fut: "asyncio.Future") -> None:
    # Mark the outcome as retrieved: when nobody reads a wrapped step (timeout,
    # client gone, cleanup's plain wait), asyncio would otherwise log "Future
    # exception was never retrieved" with a traceback for every failed step.
    if not fut.cancelled():
        fut.exception()


def _wrap_step(fut: Future) -> "asyncio.Future":
    wrapped = asyncio.wrap_future(fut)
    wrapped.add_done_callback(_consume_outcome)
    return wrapped


class _StreamRun:
    """One streaming response driven by a synchronous generator on EXECUTOR.

    Owns the cancel flag, the single in-flight step and (for /api/generate)
    the capacity slot, and guarantees the cleanup below runs exactly once:

        set cancel -> wait for the in-flight step -> gen.close() on EXECUTOR
        -> release the slot -> leave the registry

    Why this is race-free: `step` and `_cleanup` are only read and written
    on the event loop thread, and next_item() checks `_cleanup`, submits the
    step and records it with no await in between. So cleanup either starts
    before a step is submitted (and next_item then submits nothing) or sees
    that step and waits for it. At most one step exists at a time, so
    gen.close() never runs while the generator is executing on another
    thread ("generator already executing"), and the slot is released only
    after close() has returned - i.e. after the pipeline's own `finally`
    blocks, including waiting for its section writers, have run. begin_cleanup()
    is idempotent, so the stream's `finally`, the response's `finally`, and
    shutdown can all call it.
    """

    def __init__(self, kind: str, gen, cancel: threading.Event, slot: Optional[_SlotCounter] = None,
                 user_slot: Optional["_UserSlot"] = None):
        self.kind = kind
        self.gen = gen
        self.cancel = cancel
        self.slot = slot
        self.user_slot = user_slot
        self.step: Optional[Future] = None
        self.finished = False  # the generator ran to its natural end
        self.shutdown_requested = False
        self._cleanup: Optional[asyncio.Task] = None
        # Set together with _cleanup. The stream races each step against it,
        # so a stream stops at once even while a step is blocked in a call
        # that cannot be interrupted; the cleanup task keeps the slot until
        # that step has really finished.
        self._closing = asyncio.Event()

    @property
    def closing(self) -> bool:
        return self._cleanup is not None

    async def next_item(self, timeout: Optional[float] = None) -> Any:
        """Run one step of the generator on EXECUTOR.

        Returns the item; _END when the generator is exhausted or the run is
        closing (also while a step is still in flight - cleanup waits for
        it); or _TIMED_OUT if the step is still running after `timeout`
        seconds.
        """
        if self._cleanup is not None:
            return _END
        fut = EXECUTOR.submit(next, self.gen, _END)
        self.step = fut
        step = _wrap_step(fut)
        closing = asyncio.ensure_future(self._closing.wait())
        try:
            done, _ = await asyncio.wait({step, closing}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
        finally:
            closing.cancel()
        if step not in done:
            return _END if self._closing.is_set() else _TIMED_OUT
        try:
            item = step.result()
        except PipelineCancelled:
            return _END
        except BaseException:
            self.finished = True  # the exception ended the generator
            raise
        if item is _END and not self.cancel.is_set():
            self.finished = True
        return item

    def begin_cleanup(self) -> "asyncio.Task":
        """Start the one-time cleanup (idempotent); returns its task."""
        if self._cleanup is None:
            self.cancel.set()
            self._closing.set()
            if not self.finished:
                _agentic_logger.info(
                    "[%s] stream closed before the work finished; cancelling it", self.kind)
            self._cleanup = asyncio.get_running_loop().create_task(self._run_cleanup())
        return self._cleanup

    async def _run_cleanup(self) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            step = self.step
            if step is not None and not step.done():
                # asyncio.wait never raises for the awaited future, so a step
                # that failed or was cancelled is fine.
                await asyncio.wait({_wrap_step(step)})
            try:
                await loop.run_in_executor(EXECUTOR, self.gen.close)
            except RuntimeError:
                # EXECUTOR unusable (interpreter exiting): close on a plain
                # thread instead of blocking the loop.
                await asyncio.to_thread(self.gen.close)
        except Exception as exc:  # noqa: BLE001
            log_unexpected_error(f"{self.kind} cleanup", exc)
        finally:
            if self.slot is not None:
                self.slot.release()
            if self.user_slot is not None:
                self.user_slot.release()
            _active_runs.discard(self)
            if not self.finished:
                _agentic_logger.info("[%s] cancelled run stopped after %.2fs; resources released",
                                     self.kind, loop.time() - started)


class _RunCleanupMixin:
    """Makes a streaming response always clean up its run.

    The body generator's own `finally` is not enough: if the stream is torn
    down before the generator starts, or while it is suspended at a `yield`
    (the send is cancelled, not the generator), that `finally` only runs when
    the generator is garbage-collected - if ever.
    """

    def __init__(self, content, run: _StreamRun, **kwargs: Any):
        super().__init__(content, **kwargs)  # type: ignore[call-arg]
        self._run = run

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)  # type: ignore[misc]
        finally:
            self._run.begin_cleanup()


class _CleanupEventSourceResponse(_RunCleanupMixin, EventSourceResponse):
    pass


class _CleanupStreamingResponse(_RunCleanupMixin, StreamingResponse):
    pass


SERVER_RESTARTING_MESSAGE = (
    "The server is restarting, so this generation was stopped. Please try "
    "again in a moment."
)


def _stop_streams_for_shutdown() -> None:
    """Cancel every open stream; each then ends at its next step boundary.

    Needed because uvicorn only runs the lifespan shutdown AFTER every open
    connection has closed - and an SSE stream would otherwise keep going for
    up to GENERATION_DEADLINE_S. sse-starlette's own exit hook does not fire
    under the uvicorn CLI (it patches Server.handle_exit after uvicorn has
    already installed the bound method as the signal handler).
    """
    for run in list(_active_runs):
        run.shutdown_requested = True
        run.begin_cleanup()


def _hook_shutdown_signals() -> dict:
    """Chain a stream-stopping step in front of the server's SIGINT/SIGTERM
    handlers. Returns the handlers to restore. Only possible (and only
    needed) when the app runs on the main thread, as under the uvicorn CLI."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    loop = asyncio.get_running_loop()
    restore: dict = {}
    sigs = [signal.SIGINT, signal.SIGTERM]
    if sys.platform == "win32":
        sigs.append(signal.SIGBREAK)  # uvicorn handles it there too
    for sig in sigs:
        prev = signal.getsignal(sig)
        if not callable(prev):
            continue  # no server handler to chain onto

        def _handler(signum, frame, _prev=prev):
            try:
                # A signal handler must not touch asyncio objects directly.
                loop.call_soon_threadsafe(_stop_streams_for_shutdown)
            finally:
                # Whatever happens above (e.g. the loop is already closed),
                # the server must still see its Ctrl+C / SIGTERM.
                _prev(signum, frame)

        signal.signal(sig, _handler)
        restore[sig] = prev
    return restore


async def _shutdown_streams(timeout: Optional[float] = None) -> None:
    """Cancel every open stream and wait (bounded) for their cleanup.

    EXECUTOR is deliberately NOT shut down. uvicorn runs this only after every
    connection has closed, so no request work is left queued on it, and
    shutdown(wait=False) cannot interrupt a running step anyway - the
    interpreter joins the pool's threads at exit either way. Shutting the
    module-global pool down bought nothing and made the app unusable after
    one lifespan cycle (e.g. a second `with TestClient(app)`). The backend
    holds no DB pool or long-lived HTTP client: model and auth calls use
    per-call `requests`."""
    grace = SHUTDOWN_GRACE_S if timeout is None else timeout
    _stop_streams_for_shutdown()
    tasks = [run.begin_cleanup() for run in list(_active_runs)]
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=grace)
        if pending:
            _agentic_logger.warning(
                "[shutdown] %d stream(s) still stopping after %.1fs; exiting anyway",
                len(pending), grace)

# Upload size caps (bytes). Without these, `await file.read()` loads whatever
# the client sends straight into RAM.
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "25")) * 1024 * 1024
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024

# Pages /api/extract-pdf will parse. A 20 MB PDF can hold tens of thousands of
# tiny pages; parsing stops here (and once MAX_TEXT_CHARS + a margin of text
# has been extracted), and the response reports `truncated`. Remaining risk:
# a single pathological page can still be slow to parse - there is no
# per-page CPU/time limit (that needs a subprocess, out of scope here).
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "500"))
_PDF_TEXT_MARGIN = 10_000

# Time limits for one server-side URL fetch (/api/extract-url): per attempt a
# 5s connect and 20s read timeout, and an overall deadline across every
# address and redirect hop, including the body download.
URL_FETCH_DEADLINE_S = float(os.getenv("URL_FETCH_DEADLINE_S", "30"))
_URL_CONNECT_TIMEOUT_S = 5.0
_URL_READ_TIMEOUT_S = 20.0
URL_FETCH_TIMEOUT_MESSAGE = "Could not fetch URL: the site took too long to respond."

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


# NAT64 prefixes (RFC 6052 well-known, RFC 8215 local-use): the low 32 bits
# are an IPv4 address the translator will connect to.
_NAT64_NETS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))
# IPv4-compatible IPv6 (deprecated, ::a.b.c.d) — `is_global` says True for
# ::127.0.0.1, so the whole block is refused outright.
_V4_COMPAT_NET = ipaddress.ip_network("::/96")


def _ip_is_public(ip) -> bool:
    """True only for a globally routable unicast address.

    `is_global` is the allowlist: it already excludes loopback, RFC 1918,
    link-local / cloud metadata (169.254.x), carrier-grade NAT (100.64/10,
    e.g. Alibaba's 100.100.100.200 metadata), 0.0.0.0/8, benchmarking, ULA
    (fc00::/7) and the rest of the IANA special-purpose registry. On top of
    that, IPv6 forms that EMBED an IPv4 address the packet ends up at
    (IPv4-mapped, 6to4, Teredo, NAT64) are judged by that embedded address.
    """
    if ip.version == 6:
        if ip in _V4_COMPAT_NET:
            return False
        embedded = [ip.ipv4_mapped, ip.sixtofour]
        if ip.teredo:
            embedded.append(ip.teredo[1])
        if any(ip in net for net in _NAT64_NETS):
            embedded.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        for v4 in embedded:
            if v4 is not None and not _ip_is_public(v4):
                return False
    return ip.is_global and not ip.is_multicast


def _resolve_public(host: str, port: Optional[int] = None) -> Optional[list]:
    """Resolve `host` once; return its addresses only if EVERY one is public.

    None when it doesn't resolve or any address is private/reserved. The
    caller connects to one of the returned addresses (see _PinnedIPAdapter),
    so a second, attacker-controlled DNS answer (rebinding) is never used.
    """
    try:
        infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, ValueError):
        return None
    addrs: list = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        except ValueError:
            return None
        if not _ip_is_public(ip):
            return None
        if str(ip) not in addrs:  # getaddrinfo repeats an address per socket type
            addrs.append(str(ip))
    return addrs or None


def _host_resolves_public(host: str) -> bool:
    """True only if EVERY address the hostname resolves to is a public IP
    (see _ip_is_public), so /api/extract-url can't be used to probe this
    server or its internal network (SSRF)."""
    return _resolve_public(host) is not None


def _pinned_adapter(host: str, ip: str):
    """A requests transport adapter that connects to `ip` - an address that
    was already vetted - instead of resolving `host` again.

    The request itself still names `host`: the Host header, the TLS SNI and
    the certificate hostname check (assert_hostname) all use the ORIGINAL
    name, so https verification is exactly as strict as a normal request.
    Only the TCP destination is pinned. This closes the DNS-rebinding gap
    between "validate the name" and "urllib3 resolves it again to connect".
    """
    from requests.adapters import HTTPAdapter

    # The pin lives in build_connection_pool_key_attributes (requests >= 2.32).
    # On an older requests that hook is never called and urllib3 would quietly
    # resolve `host` again - so refuse to fetch rather than fetch unpinned.
    if not hasattr(HTTPAdapter, "build_connection_pool_key_attributes"):
        raise RuntimeError("requests >= 2.32 is required for pinned URL fetches")

    class _PinnedIPAdapter(HTTPAdapter):
        def add_headers(self, request, **kwargs):
            # urllib3 would otherwise derive Host from the pool's host - the
            # pinned IP. Name the original host (and port, if the URL has one).
            request.headers["Host"] = urlparse(request.url).netloc

        def build_connection_pool_key_attributes(self, request, verify, cert=None):
            host_params, pool_kwargs = super().build_connection_pool_key_attributes(
                request, verify, cert)
            host_params["host"] = ip
            if host_params.get("scheme") == "https":
                pool_kwargs["server_hostname"] = host
                pool_kwargs["assert_hostname"] = host
            return host_params, pool_kwargs

    return _PinnedIPAdapter()


def _checked_target(url: str):
    """Parse and validate one hop's URL. Returns (parsed, host, port).

    The host is taken from urllib3's parser - the one requests itself uses to
    decide where to connect - never from urllib.parse, whose idea of the host
    differs for inputs like `http://127.0.0.1:8765\\@example.com/` (urlparse
    says example.com, urllib3 connects to 127.0.0.1). Backslashes, userinfo
    and control characters are refused outright: no legitimate article link
    needs them and they are what parser-confusion attacks are built from.
    """
    from urllib3.util import parse_url
    from urllib3.exceptions import LocationParseError

    if "\\" in url or any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in url):
        raise HTTPException(422, "This URL contains characters that aren't allowed.")
    try:
        parsed = parse_url(url)
    except (LocationParseError, ValueError):
        raise HTTPException(422, "This URL couldn't be understood.") from None
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise HTTPException(422, "Only http(s) URLs are allowed.")
    if parsed.auth is not None or "@" in (urlparse(url).netloc or ""):
        raise HTTPException(422, "URLs with a username or password aren't allowed.")
    host = (parsed.host or "").strip("[]").lower()
    # Belt and braces: both parsers must agree on the host (urllib3 returns
    # an internationalised name in its punycode form).
    other = (urlparse(url).hostname or "").lower()
    try:
        other_ascii = other.encode("idna").decode("ascii")
    except UnicodeError:
        other_ascii = other
    if not host or host not in (other, other_ascii):
        raise HTTPException(422, "This URL couldn't be understood.")
    port = parsed.port or (443 if scheme == "https" else 80)
    return parsed, host, port


def _abort_response(resp, aborted: threading.Event) -> None:
    """Deadline watchdog: wake a body read that is blocked on a slow server.

    Read timeouts are per recv(), so a server dripping one byte every few
    seconds never trips them; shutting the socket down ends the read at once.
    """
    aborted.set()
    raw = getattr(resp, "raw", None)
    # The connection's socket - or, when the server answered HTTP/1.0 or
    # "Connection: close", http.client has already dropped conn.sock and only
    # the response's own socket file (fp -> SocketIO -> socket) still holds it.
    socks = [
        getattr(getattr(raw, "_connection", None), "sock", None),
        getattr(getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None), "_sock", None),
    ]
    for sock in socks:
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _fetch_url_safely(url: str) -> "object":
    """GET a user-supplied URL with SSRF protection and a download size cap.

    - Only http/https; no userinfo, backslashes or control characters.
    - Every hop (including each redirect target) is parsed with urllib3's
      parser, resolved ONCE, and every address must be public
      (_ip_is_public). The connection then goes to that vetted address
      (_pinned_adapter), so DNS can't be rebound between check and connect.
    - Environment proxies are ignored: a proxy would resolve the name itself
      and bypass the pinning.
    - Response body is streamed and truncated at MAX_URL_FETCH_BYTES.
    - Time-bounded: 5s connect / 20s read per attempt, and URL_FETCH_DEADLINE_S
      for the whole fetch (all addresses, all hops, the body); past it: 422.
    Returns the requests.Response with `.safe_text` attached.
    """
    import requests as _requests

    deadline = time.monotonic() + URL_FETCH_DEADLINE_S

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise HTTPException(422, URL_FETCH_TIMEOUT_MESSAGE)
        return left

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AgenticNotes/1.0)"}
    current = url
    for _hop in range(4):  # original request + up to 3 redirects
        remaining()
        _parsed, host, port = _checked_target(current)
        addrs = _resolve_public(host, port)
        if not addrs:
            raise HTTPException(
                422, "This URL points at a private or unreachable address and can't be fetched."
            )

        # Try each vetted address in order (like a normal client would try
        # every A/AAAA record); a connection failure moves on to the next.
        # Every attempt is pinned to a vetted address and verifies TLS
        # against the original hostname.
        attempts = addrs[:4]  # bounds the worst case at 4 connect timeouts
        for index, ip in enumerate(attempts):
            session = _requests.Session()
            session.trust_env = False  # no env proxies (see docstring)
            adapter = _pinned_adapter(host, ip)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            left = remaining()
            try:
                resp = session.get(
                    current, headers=headers, stream=True, allow_redirects=False,
                    timeout=(min(_URL_CONNECT_TIMEOUT_S, left), min(_URL_READ_TIMEOUT_S, left)),
                )
                break
            except _requests.ConnectionError:
                session.close()
                if index == len(attempts) - 1:
                    remaining()  # a spent deadline says "took too long"
                    raise
            except _requests.Timeout:
                session.close()
                raise HTTPException(422, URL_FETCH_TIMEOUT_MESSAGE) from None
            except BaseException:
                session.close()
                raise
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            session.close()
            if not location:
                raise HTTPException(422, "URL redirected without a destination.")
            current = urljoin(current, location)
            continue

        aborted = threading.Event()
        watchdog = threading.Timer(remaining(), _abort_response, args=(resp, aborted))
        watchdog.daemon = True
        watchdog.start()
        try:
            resp.raise_for_status()
            # Stream the body with a hard byte cap.
            body, total = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > MAX_URL_FETCH_BYTES:
                    break
                body.append(chunk)
        except (_requests.RequestException, OSError):
            if aborted.is_set():
                raise HTTPException(422, URL_FETCH_TIMEOUT_MESSAGE) from None
            raise
        finally:
            watchdog.cancel()
            resp.close()
            session.close()
        if aborted.is_set():  # the read ended because the watchdog cut it
            raise HTTPException(422, URL_FETCH_TIMEOUT_MESSAGE)
        raw = b"".join(body)
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        try:
            resp.safe_text = raw.decode(encoding, errors="replace")
        except LookupError:
            resp.safe_text = raw.decode("utf-8", errors="replace")
        return resp

    raise HTTPException(422, "Too many redirects.")


def _require_known_model(model: str) -> str:
    """422 unless `model` is empty (use the default) or one /api/models lists.

    Called first thing by every handler that takes a `model`, so a caller
    can't route requests to arbitrary (paid) catalog models.
    """
    try:
        return validate_model_id(model)
    except UnknownModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


# Rate limit + per-account in-flight slot for the non-streaming expensive
# endpoints. One dependency per bucket, shared by the endpoints in it (the
# limiter's state is keyed by bucket name, so this is the same as before).
_EXTRACT_GUARD = _inflight_guard(limiter("extract", 20, 600, daily=DAILY_EXTRACTS))
_REGEN_GUARD = _inflight_guard(limiter("regen", 20, 600, daily=DAILY_REGENS))
_EXPORT_GUARD = _inflight_guard(limiter("export", 30, 600))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PageSpan(BaseModel):
    """One entry of /api/extract-pdf's `page_spans`. Wrong TYPES are a 422 (no
    honest client sends them); structurally inconsistent spans are dropped in
    generate() instead, because stale spans are an honest case (the user
    edited the extracted text before generating)."""
    page: int = Field(ge=1, le=1_000_000)
    start: int = Field(ge=0, le=100_000_000)
    end: int = Field(ge=0, le=100_000_000)


class ChatTurn(BaseModel):
    """One prior chat message, exactly as ChatPanel sends it."""
    role: Literal["user", "assistant"]
    # An answer is capped at 900 output tokens and the prompt only keeps the
    # first 2000 chars of each turn; this bound is generous on purpose.
    content: str = Field(default="", max_length=20000)


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
    page_spans: list[PageSpan] = Field(default_factory=list, max_length=5000)


class ChatRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    question: str = Field(default="", max_length=4000)
    # ChatPanel sends the WHOLE conversation (error bubbles included), so the
    # list has no length cap - any cap would eventually 422 a long chat. Each
    # item is still validated; the JSON body-size limit bounds the payload and
    # the handler keeps only the last CHAT_HISTORY_TURNS.
    history: list[ChatTurn] = Field(default_factory=list)
    model: str = Field(default="", max_length=100)


class UrlRequest(BaseModel):
    url: str = Field(default="", max_length=2000)


class EditSelectionRequest(BaseModel):
    notes: str = Field(default="", max_length=300000)
    selection: str = Field(default="", max_length=20000)
    instruction: str = Field(default="", max_length=2000)
    model: str = Field(default="", max_length=100)
    # Source-line anchors from the rendered notes (NotesOutput's data-line):
    # the highlight is rendered text without markdown/citations, so these
    # say which raw lines it covers. Optional; text matching is the fallback.
    line_start: Optional[int] = Field(default=None, ge=0, le=200000)
    line_end: Optional[int] = Field(default=None, ge=0, le=200000)


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
        # Whether an Ollama endpoint was configured - not the URL itself,
        # which is internal topology an unauthenticated caller has no use for.
        "ollama_configured": bool((os.getenv("OLLAMA_URL") or "").strip()),
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


def _load_eval_report() -> dict:
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "eval", "report.json")
    if not os.path.isfile(path):
        return {"available": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"available": True, "report": json.load(f)}
    except Exception:  # noqa: BLE001
        return {"available": False}


@app.get("/api/eval-report")
async def eval_report():
    """Serve the latest eval report (eval/report.json) for the dashboard."""
    # File I/O + JSON parsing is blocking; keep it off the event loop.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, _load_eval_report)


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
    model = _require_known_model(req.model)
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"`text` exceeds the {MAX_TEXT_CHARS} character limit.",
        )

    # Page spans are only honoured when they are structurally what
    # /api/extract-pdf produces for THIS text; anything else (stale after an
    # edit, or hand-crafted to multiply the chunking work) is dropped and the
    # document is chunked without pages, exactly as for pasted text.
    spans = [s.model_dump() for s in req.page_spans]
    if spans and not valid_page_spans(spans, len(normalize(text))):
        _agentic_logger.warning("[generate] ignoring %d inconsistent page span(s)", len(spans))
        spans = []

    # Reject immediately (don't queue invisibly) when the server is already
    # running its maximum number of pipelines. The check and the acquire are one
    # atomic step; from here on the stream owns the slot and its cleanup
    # releases it.
    user_slot = _take_user_slot(user)
    slots = _generation_slots
    if not slots.try_acquire():
        user_slot.release()
        raise HTTPException(
            status_code=503,
            detail="The server is at capacity right now — please try again in a minute.",
        )
    try:
        cancel = threading.Event()
        gen = run_agent(
            text,
            req.mode,
            req.tone,
            req.length,
            req.format,
            model=model,
            instructions=req.instructions,
            include_quiz=req.include_quiz,
            include_flashcards=req.include_flashcards,
            page_spans=spans,
            cancel=cancel,
        )
        run = _StreamRun("generate", gen, cancel, slot=slots, user_slot=user_slot)
        response = _CleanupEventSourceResponse(_generation_events(run), run=run)
    except BaseException:
        slots.release()
        user_slot.release()
        raise
    _active_runs.add(run)
    return response


async def _generation_events(run: _StreamRun):
    """SSE body: step the synchronous pipeline on EXECUTOR under a deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GENERATION_DEADLINE_S
    try:
        while True:
            remaining = deadline - loop.time()
            event: Any = _TIMED_OUT if remaining <= 0 else await run.next_item(timeout=remaining)
            if event is _TIMED_OUT:
                # Stop the pipeline now (the slot is freed once it has
                # actually stopped), then tell the client why.
                _agentic_logger.info("[generate] deadline of %ss reached; cancelling",
                                     GENERATION_DEADLINE_S)
                run.begin_cleanup()
                yield {"data": json.dumps({"type": "error", "step": "error",
                                           "content": GENERATION_TIMEOUT_MESSAGE,
                                           "data": None})}
                return
            if event is _END:
                if run.shutdown_requested and not run.finished:
                    yield {"data": json.dumps({"type": "error", "step": "error",
                                               "content": SERVER_RESTARTING_MESSAGE,
                                               "data": None})}
                return
            if event.get("type") in _TERMINAL_EVENTS:
                # The pipeline's last word: nothing (e.g. a shutdown notice
                # landing in the pacing sleep below) may follow it.
                run.finished = True
            yield {"data": json.dumps(event)}
            # Pace discrete step events so the frontend can render each one,
            # but stream token deltas as fast as they arrive.
            if event.get("type") != "notes_delta":
                await asyncio.sleep(0.05)
    finally:
        # Client disconnect (CancelledError here), deadline, or normal end.
        run.begin_cleanup()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), user=Depends(_EXTRACT_GUARD)):
    from pypdf import PdfReader

    raw = await _read_upload(file, MAX_PDF_BYTES, "PDF")
    def _parse() -> tuple:
        reader = PdfReader(BytesIO(raw))
        n_pages = len(reader.pages)
        parts, chars, stopped_early = [], 0, False
        for index, page in enumerate(reader.pages):
            # Bound the parsing work: text past MAX_TEXT_CHARS is cut below
            # anyway, so extracting further pages only burns a worker thread.
            if index >= MAX_PDF_PAGES or chars > MAX_TEXT_CHARS + _PDF_TEXT_MARGIN:
                stopped_early = True
                break
            try:
                part = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                part = ""
            parts.append(part)
            chars += len(part.strip()) + 2  # as measured in `joined` below
        joined = "\n\n".join(c.strip() for c in parts if c.strip()).strip()
        # Page boundaries measured in the same normalized space the chunker
        # uses, so a citation can later be traced back to the page it came
        # from. Computed here because this is the only place page structure
        # still exists — joining throws it away.
        return n_pages, joined, page_spans(parts), stopped_early

    loop = asyncio.get_running_loop()
    try:
        pages, text, spans, stopped_early = await loop.run_in_executor(EXECUTOR, _parse)
    except Exception as exc:  # noqa: BLE001
        # The parser's exception text is not for users; log it with an id.
        error_id = log_unexpected_error("extract-pdf", exc)
        raise HTTPException(
            status_code=422,
            detail="Could not read this PDF - it may be damaged, encrypted or "
            f"not a PDF. (error id: {error_id})") from None

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF (it may be scanned/image-only).",
        )

    # Never hand back more than /api/generate will accept. extract-url and the
    # transcript path already clamp here; this one didn't, so a PDF over the
    # limit extracted fine, reported "N words ready", and then failed at
    # Generate with a raw 422 — the worst possible moment to find out.
    # stopped_early: pages past MAX_PDF_PAGES (or past the character budget)
    # were never read, so the text is incomplete even if it is under the cap.
    truncated = stopped_early or len(text) > MAX_TEXT_CHARS
    if len(text) > MAX_TEXT_CHARS:
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
async def extract_image(file: UploadFile = File(...), user=Depends(_EXTRACT_GUARD)):
    """OCR a photo of study material (textbook page, slides, handwriting) via Gemini vision."""
    if not gemini_available():
        raise HTTPException(
            status_code=503,
            detail="Image text extraction requires a GEMINI_API_KEY. Set it in backend/.env.",
        )

    raw = await _read_upload(file, MAX_IMAGE_BYTES, "Image")
    mime = file.content_type or "image/jpeg"
    loop = asyncio.get_running_loop()
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
async def transcribe(file: UploadFile = File(...), user=Depends(_EXTRACT_GUARD)):
    """Transcribe a lecture recording or voice memo via Gemini."""
    if not gemini_available():
        raise HTTPException(
            status_code=503,
            detail="Audio transcription requires a GEMINI_API_KEY. Set it in backend/.env.",
        )

    raw = await _read_upload(file, MAX_AUDIO_BYTES, "Audio file")
    mime = file.content_type or "audio/webm"

    loop = asyncio.get_running_loop()
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
async def regen_quiz(req: RegenRequest, user=Depends(_REGEN_GUARD)):
    model = _require_known_model(req.model)
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_running_loop()

    # This is the quiz users actually get (the pipeline's include_quiz path is
    # off in the UI), so its answer key is checked here, with the same model
    # that wrote it. A verifier failure must not cost the user their quiz: it
    # fails open, and `verification.checked: false` says the key wasn't checked
    # instead of passing an unchecked key off as a checked one.
    def _quiz_job():
        quiz = generate_quiz(notes, 5, model)
        try:
            return verify_quiz_detailed(notes, quiz, model)
        except Exception:  # noqa: BLE001
            return quiz, {"checked": False, "questions": 0, "judged": 0,
                          "corrected": 0, "rejected": 0, "disputed": []}

    quiz, verification = await loop.run_in_executor(EXECUTOR, _quiz_job)
    return {"quiz": quiz, "verification": verification}


@app.post("/api/flashcards")
async def regen_flashcards(req: RegenRequest, user=Depends(_REGEN_GUARD)):
    model = _require_known_model(req.model)
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    loop = asyncio.get_running_loop()
    cards = await loop.run_in_executor(EXECUTOR, lambda: generate_flashcards(notes, 8, model))
    return {"flashcards": cards}


@app.post("/api/edit-selection")
async def edit_selection_endpoint(req: EditSelectionRequest, user=Depends(_REGEN_GUARD)):
    model = _require_known_model(req.model)
    raw = req.notes or ""
    notes = raw.strip()
    # The anchors index the notes as the client rendered them; stripping
    # leading blank lines shifts every line, so shift the anchors with them.
    lead = raw[:len(raw) - len(raw.lstrip())].count("\n")
    line_start = None if req.line_start is None else req.line_start - lead
    line_end = None if req.line_end is None else req.line_end - lead
    selection = (req.selection or "").strip()
    if not notes or not selection:
        raise HTTPException(
            status_code=422, detail="`notes` and `selection` are required.")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            EXECUTOR, lambda: edit_selection(
                notes, selection, req.instruction, model,
                line_start=line_start, line_end=line_end)
        )
    except SelectionNotFoundError:
        raise HTTPException(
            status_code=422,
            detail="Couldn't find the selected text in your notes. "
            "Try selecting it again.")
    except SelectionAmbiguousError:
        raise HTTPException(
            status_code=422,
            detail="The selected text appears more than once in your notes. "
            "Select a longer passage so it's unique.")
    except Exception as exc:  # noqa: BLE001 - provider failure, truncation, empty reply
        # Generic on purpose: exception text can carry provider URLs/keys.
        # A strict call that was cut off lands here too - the notes are left
        # exactly as they were rather than replaced with a partial edit.
        print(f"[edit-selection] failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="The edit couldn't be completed. Nothing was changed - "
            "please try again.")
    return {"notes": result}


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest, user=Depends(_REGEN_GUARD)):
    model = _require_known_model(req.model)
    notes = (req.notes or "").strip()
    if not notes:
        raise HTTPException(status_code=422, detail="`notes` is required.")
    if req.direction not in ("shorter", "longer", "clarity"):
        raise HTTPException(status_code=422, detail="Invalid `direction`.")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            EXECUTOR,
            lambda: rewrite_notes(notes, req.direction, req.mode,
                                  req.tone, req.format, model),
        )
    except RewriteTooLongError:
        raise HTTPException(
            status_code=422,
            detail="These notes are too long to rewrite in one go. "
            "Try rewriting a section at a time.")
    except Exception as exc:  # noqa: BLE001 - provider failure or a truncated part
        # Any failed or cut-off part fails the whole rewrite: partially
        # rewritten notes are never returned. Message kept generic (no keys).
        print(f"[rewrite] failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="The rewrite couldn't be completed. Nothing was changed - "
            "please try again.")
    return {"notes": result}


# ---------------------------------------------------------------------------
# Chat with your notes (streaming plain text)
# ---------------------------------------------------------------------------

CHAT_CUT_OFF_NOTICE = "\n\n_[The answer was cut off \u2014 please ask again.]_"


def _chat_notice(message: str, after_text: bool) -> str:
    """A chat error as plain text, styled like CHAT_CUT_OFF_NOTICE."""
    return ("\n\n" if after_text else "") + f"_[{message}]_"


@app.post("/api/chat")
async def chat(req: ChatRequest, user=Depends(limiter("chat", 40, 600, daily=DAILY_CHATS))):
    model = _require_known_model(req.model)
    notes = (req.notes or "").strip()
    question = (req.question or "").strip()
    if not notes:
        raise HTTPException(
            status_code=422, detail="Generate notes first to chat about them.")
    if not question:
        raise HTTPException(status_code=422, detail="`question` is required.")

    # Same cancel-on-disconnect as /api/generate: closing the stream sets the
    # flag (the provider stream stops at its next delta) and closes the
    # generator once its in-flight step is done. No capacity slot or deadline:
    # a chat answer is one bounded model call.
    history = [t.model_dump() for t in req.history[-CHAT_HISTORY_TURNS:]]
    # From here the stream's cleanup gives the per-user slot back.
    user_slot = _take_user_slot(user)
    cancel = threading.Event()
    try:
        run = _StreamRun("chat", chat_about_notes_stream(notes, question, history, model=model,
                                                         cancel=cancel), cancel, user_slot=user_slot)
    except BaseException:
        user_slot.release()
        raise

    async def token_generator():
        sent = False
        try:
            while True:
                try:
                    piece = await run.next_item()
                except IncompleteStreamError:
                    # The partial answer is already on screen; say so rather than
                    # let it pass for the whole answer.
                    yield CHAT_CUT_OFF_NOTICE
                    break
                except UserFacingError as exc:
                    # e.g. every provider down. The 200 headers are already
                    # sent, so the only way to tell the user is the body; an
                    # escaping exception would just drop the connection.
                    if isinstance(exc, ProvidersUnavailableError):
                        log_unexpected_error("chat", exc)  # per-provider detail
                    yield _chat_notice(exc.user_message, sent)
                    break
                except Exception as exc:  # noqa: BLE001
                    # Raw text can carry provider errors/URLs: log it, show an id.
                    error_id = log_unexpected_error("chat", exc)
                    yield _chat_notice(
                        "Something went wrong while answering. Please try again. "
                        f"(error id: {error_id})", sent)
                    break
                if piece is _END:
                    if run.shutdown_requested and not run.finished:
                        # Stopped by a server restart, not by the client: the
                        # reader is still there and must not take a partial
                        # answer for the whole one.
                        yield CHAT_CUT_OFF_NOTICE
                    break
                sent = True
                yield piece
        finally:
            run.begin_cleanup()

    _active_runs.add(run)
    return _CleanupStreamingResponse(token_generator(), run=run, media_type="text/plain")


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
async def extract_url(req: UrlRequest, user=Depends(_EXTRACT_GUARD)):
    import requests as _requests
    from bs4 import BeautifulSoup

    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="`url` is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # YouTube: fetch the caption transcript (the real content) instead of HTML.
    yt_id = _youtube_id(url)
    loop = asyncio.get_running_loop()
    if yt_id:
        try:
            # Blocking network call — keep it off the event loop (single
            # worker: it would otherwise stall every open SSE stream).
            transcript = (await loop.run_in_executor(
                EXECUTOR, lambda: _youtube_transcript(yt_id))).strip()
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

    try:
        # The fetch enforces URL_FETCH_DEADLINE_S itself; this outer bound
        # only guarantees the response time if DNS resolution itself hangs.
        resp = await asyncio.wait_for(
            loop.run_in_executor(EXECUTOR, lambda: _fetch_url_safely(url)),
            URL_FETCH_DEADLINE_S + 5)
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=422, detail=URL_FETCH_TIMEOUT_MESSAGE) from None
    except _requests.HTTPError as exc:
        # The remote status is ours to report; the exception text is not.
        status = getattr(exc.response, "status_code", None)
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch URL: the page returned HTTP {status}."
            if status else "Could not fetch URL: the page returned an error.") from None
    except _requests.Timeout:
        raise HTTPException(
            status_code=422, detail="Could not fetch URL: the site took too long to respond.") from None
    except _requests.ConnectionError:
        raise HTTPException(
            status_code=422, detail="Could not fetch URL: couldn't connect to the site.") from None
    except Exception as exc:  # noqa: BLE001
        error_id = log_unexpected_error("extract-url", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch URL. (error id: {error_id})") from None

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
# Rendering up to 300k chars with reportlab/python-docx is CPU-bound for a
# noticeable time; run it on EXPORT_EXECUTOR so the single worker's event loop
# (and every open SSE stream) isn't frozen meanwhile, and so a burst of exports
# can't occupy EXECUTOR's threads either.

@app.post("/api/export/pdf")
async def export_pdf(req: ExportRequest, user=Depends(_EXPORT_GUARD)):
    loop = asyncio.get_running_loop()
    pdf_bytes = await loop.run_in_executor(
        EXPORT_EXECUTOR, lambda: notes_to_pdf(req.notes, req.quiz, req.flashcards))
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=notes.pdf"},
    )


@app.post("/api/export/markdown")
async def export_markdown(req: ExportRequest, user=Depends(_EXPORT_GUARD)):
    loop = asyncio.get_running_loop()
    md = await loop.run_in_executor(
        EXPORT_EXECUTOR, lambda: notes_to_markdown(req.notes, req.quiz, req.flashcards))
    return StreamingResponse(
        BytesIO(md.encode("utf-8")),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=notes.md"},
    )


@app.post("/api/export/docx")
async def export_docx(req: ExportRequest, user=Depends(_EXPORT_GUARD)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(
        EXPORT_EXECUTOR, lambda: notes_to_docx(req.notes, req.quiz, req.flashcards))
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=notes.docx"},
    )


@app.post("/api/export/flashcards-csv")
async def export_flashcards_csv(req: ExportRequest, user=Depends(_EXPORT_GUARD)):
    loop = asyncio.get_running_loop()
    csv_text = await loop.run_in_executor(
        EXPORT_EXECUTOR, lambda: flashcards_to_csv(req.flashcards))
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