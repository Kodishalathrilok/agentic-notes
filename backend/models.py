"""
Model router: a provider-agnostic LLM layer with automatic failover.

Providers (failover order starts from the chosen model's provider):
  - nvidia : NVIDIA NIM / build.nvidia.com (OpenAI-compatible; Nemotron etc.)
  - gemini : Google Gemini (large context, generous free tier, multimodal)
  - ollama : local fallback (offline)

If the chosen provider is rate-limited or down, the next available provider is
used automatically — so one provider's rate limit doesn't dead-end a request.

NVIDIA takes precedence when configured: its models are listed first, so
default_model() picks one and _provider_for() routes to it. Gemini is the
hosted fallback and also does the multimodal work (image OCR, audio
transcription) that NVIDIA's chat endpoint can't.

Gemini ids are pinned to a currently-available release, and the router falls
back to Google's moving alias automatically if that id is ever retired. See
_gemini_post for why neither half works on its own.

Public API:
    call_model(prompt, max_tokens, model, temperature, json_mode) -> str
    call_model_stream(prompt, max_tokens, model, temperature)     -> generator[str]
    safe_json(text)        -> dict
    get_active_provider()  -> "nvidia" | "gemini" | "ollama"
    available_models()     -> list[dict]
    default_model()        -> str
    resolve_model(model)   -> str
    extract_text_from_image(bytes, mime) -> str   (Gemini vision OCR)
    transcribe_audio(bytes, mime)        -> str   (Gemini audio)
"""

import os
import re
import json
import time

import requests
from dotenv import load_dotenv

# Anchored to this directory, NOT the process's cwd. Bare load_dotenv() walks
# up from the working directory, so launching from anywhere but backend/ (a
# uvicorn --app-dir invocation, an IDE run config, a service unit) found no
# .env and silently started the app with EVERY provider key missing — the
# health endpoint just reported "ollama" with no error anywhere.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Gemini
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
# Pinned to a currently-available release, with an automatic alias fallback if
# it is ever retired — see _gemini_post. Google already retired
# gemini-2.0-flash and then gemini-2.5-flash out from under this app.
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
AVAILABLE_GEMINI_MODELS = [
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash · Google (large context)", "provider": "gemini"},
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite · Google (cheapest)", "provider": "gemini"},
]
# Where a retired id self-heals to. These track the newest release, which is
# why they are the safety net and not the default.
_GEMINI_FLASH_ALIAS = "gemini-flash-latest"
_GEMINI_LITE_ALIAS = "gemini-flash-lite-latest"
_GEMINI_ROOT = "https://generativelanguage.googleapis.com"
_GEMINI_BASE = f"{_GEMINI_ROOT}/v1beta"
_GEMINI_UPLOAD = f"{_GEMINI_ROOT}/upload/v1beta/files"
# Current Gemini Flash models think before answering, and those thought tokens
# are billed against maxOutputTokens — the same trap NVIDIA's reasoning models
# set (see NVIDIA_REASONING_HEADROOM). Measured: a 30-token budget on the title
# prompt spends 27 tokens thinking and returns an EMPTY string with
# finishReason MAX_TOKENS. Every small budget in agent.py (30 for the title,
# 200 for the gatekeeper) would silently come back blank without this.
GEMINI_REASONING_HEADROOM = int(os.getenv("GEMINI_REASONING_HEADROOM", "1024"))

# NVIDIA (NIM — OpenAI-compatible chat/completions)
#
# The model id is deliberately NOT hard-coded to a list. NVIDIA's catalog ids
# are versioned strings like "nvidia/nemotron-3-ultra-550b-a55b", and the
# Nemotron line renames across releases — pinning a guess in code means a code
# change every time you want a different one. Set NVIDIA_MODEL to the exact id
# from build.nvidia.com; add NVIDIA_MODELS (comma-separated) to offer more than
# one in the model picker.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
# Extra completion-token allowance for models that emit chain-of-thought into
# `reasoning_content` before answering. See _nvidia_body for why this exists.
NVIDIA_REASONING_HEADROOM = int(os.getenv("NVIDIA_REASONING_HEADROOM", "1024"))
# "off" sends the Nemotron "detailed thinking off" system directive, which cuts
# reasoning output substantially. Left on by default — reasoning is only ~11%
# of generated text on notes-sized prompts, and it is presumably why you chose
# a reasoning model.
NVIDIA_THINKING = os.getenv("NVIDIA_THINKING", "on").strip().lower()
DEFAULT_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct").strip()


def _nvidia_model_ids():
    """Configured NVIDIA model ids, default first, de-duplicated in order."""
    ids = [DEFAULT_NVIDIA_MODEL]
    ids += [m.strip() for m in os.getenv("NVIDIA_MODELS", "").split(",") if m.strip()]
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _nvidia_label(model_id: str) -> str:
    """Turn a catalog id into a picker label: "nvidia/nemotron-3-ultra-550b-a55b"
    becomes "Nemotron 3 Ultra 550B A55B"."""
    tail = model_id.split("/")[-1]
    # Size/variant tokens ("70b", "550b", "a55b", "v1") read as codes, not
    # words — capitalizing them gives "550b A55b" instead of "550B A55B".
    return " ".join(
        w.upper() if any(c.isdigit() for c in w) else w.capitalize()
        for w in tail.replace("-", " ").split()
    )


# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()


class RateLimitError(RuntimeError):
    """Kept for compatibility; failover now handles rate limits transparently."""


def _safe(exc) -> str:
    """Strip API keys out of error text before logging (handles ?key= and &key=)."""
    return re.split(r"[?&]key=", str(exc))[0]


# ---------------------------------------------------------------------------
# Shared HTTP: rate-limit / overload retry
# ---------------------------------------------------------------------------

# Both hosted providers are free tiers that answer bursts with 429, and the
# newest Gemini aliases also return 503 ("high demand") under load. The
# map-reduce writer fires many calls in quick succession, so a short wait for
# the per-minute window to refill beats dropping to the next provider.
RATE_LIMIT_RETRIES = int(os.getenv("RATE_LIMIT_RETRIES", "3"))
_RETRYABLE_STATUS = (429, 503)


def _retry_wait(resp, attempt: int) -> float:
    """Seconds to wait before retrying. Honour Retry-After, else back off."""
    headers = getattr(resp, "headers", None) or {}
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra:
        try:
            return min(float(ra) + 0.5, 30.0)
        except (TypeError, ValueError):
            pass
    return min(2.0 * (2 ** attempt), 20.0)


def _post_retrying(label: str, **kwargs):
    """requests.post that retries 429/503 with backoff before giving up.

    Returns the final response either way — the caller still calls
    raise_for_status(), so a persistent limit becomes a normal provider error
    and failover takes over from there.
    """
    resp = None
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        resp = requests.post(**kwargs)
        if resp.status_code not in _RETRYABLE_STATUS or attempt >= RATE_LIMIT_RETRIES:
            return resp
        wait = _retry_wait(resp, attempt)
        status = resp.status_code
        resp.close()
        print(f"[models] {label} returned {status}; retrying in {wait:.0f}s "
              f"(attempt {attempt + 1}/{RATE_LIMIT_RETRIES}).")
        time.sleep(wait)
    return resp


# ---------------------------------------------------------------------------
# Availability / routing
# ---------------------------------------------------------------------------

def _gemini_available() -> bool:
    return bool(GEMINI_API_KEY)


def _nvidia_available() -> bool:
    return bool(NVIDIA_API_KEY)


_ollama_reachable = None
_ollama_checked_at = 0.0
_OLLAMA_PROBE_TTL = 60  # seconds — re-probe periodically instead of caching forever


def _ollama_available() -> bool:
    """Probe Ollama with a short TTL cache. On hosted deploys (no local Ollama)
    this stays False, so a dead localhost fallback never masks the real
    NVIDIA/Gemini error — but if Ollama comes up later, it's picked up within a
    minute instead of never."""
    global _ollama_reachable, _ollama_checked_at
    now = time.time()
    if _ollama_reachable is None or (now - _ollama_checked_at) > _OLLAMA_PROBE_TTL:
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
            _ollama_reachable = True
        except Exception:  # noqa: BLE001
            _ollama_reachable = False
        _ollama_checked_at = now
    return _ollama_reachable


def _provider_ready(prov: str) -> bool:
    return {
        "nvidia": _nvidia_available(),
        "gemini": _gemini_available(),
        "ollama": _ollama_available(),
    }.get(prov, False)


def _provider_for(model_id: str) -> str:
    """Route a model id to its provider.

    The three families have distinct id shapes: Gemini ids start with
    "gemini", Ollama tags carry a ":" ("qwen2.5:3b"), and NVIDIA catalog ids
    are namespaced ("nvidia/…", "meta/…", "mistralai/…"). Configured ids are
    checked first in case a deployment sets NVIDIA_MODEL or OLLAMA_MODEL to
    something that doesn't follow its family's shape. Anything unrecognised
    goes to NVIDIA, the primary provider, and failover covers a wrong guess.
    """
    mid = (model_id or "").strip()
    if mid in _nvidia_model_ids():
        return "nvidia"
    if mid == OLLAMA_MODEL:
        return "ollama"
    if mid.startswith("gemini"):
        return "gemini"
    if ":" in mid and "/" not in mid:
        return "ollama"
    return "nvidia"


def _failover_chain(primary: str):
    return [primary] + [p for p in ("nvidia", "gemini", "ollama") if p != primary]


def available_models():
    models = []
    # NVIDIA first when configured: default_model() takes av[0], so this is
    # what actually makes it the app's primary provider.
    if _nvidia_available():
        models += [
            {"id": mid, "label": f"{_nvidia_label(mid)} · NVIDIA", "provider": "nvidia"}
            for mid in _nvidia_model_ids()
        ]
    if _gemini_available():
        models += AVAILABLE_GEMINI_MODELS
    return models or AVAILABLE_GEMINI_MODELS


def default_model() -> str:
    av = available_models()
    return av[0]["id"] if av else DEFAULT_GEMINI_MODEL


def resolve_model(model):
    if model and model.strip():
        return model.strip()
    return default_model()


def get_active_provider() -> str:
    if _nvidia_available():
        return "nvidia"
    if _gemini_available():
        return "gemini"
    return "ollama"


# A cheaper "helper" model for the mechanical packaging agents (gatekeeper,
# title, quiz, flashcards). It runs on a SEPARATE free-tier daily token quota
# from the main writing model, so these steps don't burn the primary model's
# budget — the writing/critique that actually determines quality keeps the
# strong model.
HELPER_GEMINI_MODEL = os.getenv("HELPER_GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
# Unset by default: NVIDIA's quota model isn't Gemini's, and pointing the
# helper agents at a model that may share one pool would spend the writer's
# budget on packaging steps. Falls back to the main NVIDIA model until you name
# a smaller one — set HELPER_NVIDIA_MODEL to a cheap id to get the split back.
HELPER_NVIDIA_MODEL = os.getenv("HELPER_NVIDIA_MODEL", "").strip()


def helper_model(main_model=None) -> str:
    """Pick a cheap helper model in the same provider family as `main_model`."""
    target = resolve_model(main_model)
    prov = _provider_for(target)
    if prov == "gemini":
        return HELPER_GEMINI_MODEL
    if prov == "nvidia":
        return HELPER_NVIDIA_MODEL or target
    return target


# ---------------------------------------------------------------------------
# Gemini (REST)
# ---------------------------------------------------------------------------

def _gemini_model(model):
    return model if (model and model.startswith("gemini")) else DEFAULT_GEMINI_MODEL


def _gemini_body(prompt, max_tokens, temperature, json_mode):
    # Headroom, not a bigger answer: thought tokens count against this cap, so
    # without it a small budget returns an empty string. See the constant.
    gen = {"temperature": temperature,
           "maxOutputTokens": max_tokens + GEMINI_REASONING_HEADROOM}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    return {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}


def _gemini_alias_for(model_id: str) -> str:
    """The moving alias in the same family as `model_id`."""
    return _GEMINI_LITE_ALIAS if "lite" in (model_id or "") else _GEMINI_FLASH_ALIAS


def _gemini_post(gm, body, label, timeout, stream=False):
    """POST to a Gemini model, retrying on the moving alias if the id is retired.

    Neither half of this works alone. Pinned ids get retired and come back as a
    permanent 404 ("no longer available to new users") — that is what broke
    this app twice. But defaulting to the "-latest" alias is worse in practice:
    the alias tracks the NEWEST release, which is the one under the most load.
    Measured on the same prompt, three calls each: gemini-flash-latest scored
    1/3 success at 90-170s, while the pinned id scored 3/3 at ~5s.

    So: pin for speed and availability, and self-heal to the alias the day the
    pin dies, instead of making every request pay for that day up front.
    """
    suffix = "streamGenerateContent" if stream else "generateContent"
    params = {"key": GEMINI_API_KEY}
    if stream:
        params["alt"] = "sse"

    def go(model_id):
        return _post_retrying(
            f"{label} [{model_id}]",
            url=f"{_GEMINI_BASE}/models/{model_id}:{suffix}",
            params=params,
            json=body,
            stream=stream,
            timeout=timeout,
        )

    resp = go(gm)
    alias = _gemini_alias_for(gm)
    if resp.status_code == 404 and alias != gm:
        print(f"[models] gemini model {gm} is gone (404); falling back to {alias}. "
              f"Set GEMINI_MODEL to a current id to skip this hop.")
        resp.close()
        resp = go(alias)
    return resp


def _gemini_text(data) -> str:
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _call_gemini(prompt, max_tokens, model, temperature, json_mode):
    gm = _gemini_model(model)
    resp = _gemini_post(gm, _gemini_body(prompt, max_tokens, temperature, json_mode),
                        "gemini", 180)
    resp.raise_for_status()
    return _gemini_text(resp.json()).strip()


def _stream_gemini(prompt, max_tokens, model, temperature):
    gm = _gemini_model(model)
    resp = _gemini_post(gm, _gemini_body(prompt, max_tokens, temperature, False),
                        "gemini stream", 300, stream=True)
    with resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            text = _gemini_text(data)
            if text:
                yield text


# ---------------------------------------------------------------------------
# Gemini multimodal — image OCR and audio transcription
# ---------------------------------------------------------------------------

def gemini_available() -> bool:
    return _gemini_available()


def extract_text_from_image(image_bytes: bytes, mime_type: str = "image/jpeg", model=None) -> str:
    """Transcribe study text from an image using Gemini vision."""
    import base64

    gm = _gemini_model(model)  # the Flash aliases are all vision-capable
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Transcribe ALL text from this image of study material (textbook page, "
        "lecture slide, handwritten notes, or diagram labels). Output only the "
        "transcribed text, preserving structure: headings, bullet points, numbered "
        "lists, and equations written as plain text. Do not add commentary."
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0,
                             "maxOutputTokens": 4096 + GEMINI_REASONING_HEADROOM},
    }
    resp = _gemini_post(gm, body, "gemini vision", 180)
    resp.raise_for_status()
    return _gemini_text(resp.json()).strip()


def _gemini_upload_file(data: bytes, mime_type: str, display_name: str = "upload"):
    """Push bytes to the Gemini Files API; returns (uri, name).

    Used instead of inline_data because base64 inflates a payload by a third
    and the inline path caps the WHOLE request at 20 MB — which would have
    forced the audio upload limit down from 25 MB. The Files API takes the raw
    bytes and hands back a uri to reference.
    """
    resp = _post_retrying(
        "gemini files upload",
        url=_GEMINI_UPLOAD,
        params={"key": GEMINI_API_KEY},
        headers={
            "X-Goog-Upload-Protocol": "raw",
            "X-Goog-Upload-Content-Type": mime_type,
            "Content-Type": mime_type,
        },
        data=data,
        timeout=300,
    )
    resp.raise_for_status()
    f = resp.json().get("file") or {}
    return f.get("uri"), f.get("name")


def _gemini_delete_file(name: str) -> None:
    """Best-effort cleanup. Uploads expire after 48h on their own, but user
    audio shouldn't sit in Google's file store any longer than the one call
    that needs it."""
    if not name:
        return
    try:
        requests.delete(f"{_GEMINI_BASE}/{name}", params={"key": GEMINI_API_KEY}, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm", model=None) -> str:
    """Transcribe a lecture recording / voice memo with Gemini.

    Gemini reads the audio container directly, so what the browser's
    MediaRecorder produces (audio/webm) needs no transcode on the way in.
    """
    gm = _gemini_model(model)
    mime = mime_type or "audio/webm"
    uri, name = _gemini_upload_file(audio_bytes, mime)
    if not uri:
        raise RuntimeError("Gemini file upload returned no uri.")
    prompt = (
        "Transcribe the spoken words in this audio recording verbatim. "
        "Output only the transcript as plain text, with punctuation and "
        "paragraph breaks where the speaker pauses. Do not summarise, do not "
        "add speaker labels or timestamps, and do not add commentary. "
        "If the audio contains no discernible speech, output nothing at all."
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"file_data": {"mime_type": mime, "file_uri": uri}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0,
                             "maxOutputTokens": 8192 + GEMINI_REASONING_HEADROOM},
    }
    try:
        resp = _gemini_post(gm, body, "gemini audio", 300)
        resp.raise_for_status()
        return _gemini_text(resp.json()).strip()
    finally:
        _gemini_delete_file(name)


# ---------------------------------------------------------------------------
# NVIDIA NIM (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------

def _nvidia_model(model):
    return model if _provider_for(model) == "nvidia" else DEFAULT_NVIDIA_MODEL


def _nvidia_headers():
    return {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"}


def _nvidia_body(prompt, max_tokens, model, temperature, json_mode, stream):
    messages = []
    if NVIDIA_THINKING in ("off", "false", "0"):
        # Llama-Nemotron reasoning models take this as a system directive.
        messages.append({"role": "system", "content": "detailed thinking off"})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": _nvidia_model(model),
        "messages": messages,
        "temperature": temperature,
        # Reasoning models spend completion tokens on `reasoning_content`
        # BEFORE emitting any `content`, and max_tokens caps the two together.
        # Every caller in agent.py sized its budget for the answer alone, so on
        # the small ones the model could burn the whole allowance thinking and
        # return an empty string — a blank title, or a gate whose JSON never
        # arrives and silently falls back. Headroom is a cap, not a target:
        # raising it costs nothing unless the model actually uses it.
        "max_tokens": max_tokens + NVIDIA_REASONING_HEADROOM,
        "stream": stream,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _nvidia_post(body, stream=False):
    """POST to NIM, retrying once without `response_format` if it's rejected.

    Support for structured output varies model by model across the catalog, and
    an unsupported field comes back as a 400. Retrying without it beats either
    hard-failing (which drops the request to the next provider and silently
    stops using the model you chose) or omitting it always (which costs JSON
    adherence on the models that do support it).
    """
    label = f"nvidia {body.get('model')}"
    kwargs = dict(
        url=f"{NVIDIA_BASE_URL}/chat/completions",
        headers=_nvidia_headers(),
        stream=stream,
        timeout=300 if stream else 180,
    )
    resp = _post_retrying(label, json=body, **kwargs)
    if resp.status_code == 400 and "response_format" in body:
        resp.close()
        retry = {k: v for k, v in body.items() if k != "response_format"}
        resp = _post_retrying(label, json=retry, **kwargs)
    resp.raise_for_status()
    return resp


def _call_nvidia(prompt, max_tokens, model, temperature, json_mode):
    resp = _nvidia_post(_nvidia_body(prompt, max_tokens, model, temperature, json_mode, False))
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    return ((choice.get("message") or {}).get("content") or "").strip()


def _stream_nvidia(prompt, max_tokens, model, temperature):
    with _nvidia_post(
        _nvidia_body(prompt, max_tokens, model, temperature, False, True), stream=True
    ) as resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            try:
                delta = (data["choices"][0].get("delta") or {}).get("content")
            except (KeyError, IndexError):
                delta = None
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _call_ollama(prompt, max_tokens, temperature, json_mode):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def _stream_ollama(prompt, max_tokens, temperature):
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            piece = data.get("response", "")
            if piece:
                yield piece


# ---------------------------------------------------------------------------
# Dispatch + public API
# ---------------------------------------------------------------------------

def _dispatch(prov, prompt, max_tokens, model, temperature, json_mode):
    if prov == "nvidia":
        return _call_nvidia(prompt, max_tokens, model, temperature, json_mode)
    if prov == "gemini":
        return _call_gemini(prompt, max_tokens, model, temperature, json_mode)
    return _call_ollama(prompt, max_tokens, temperature, json_mode)


def _dispatch_stream(prov, prompt, max_tokens, model, temperature):
    if prov == "nvidia":
        yield from _stream_nvidia(prompt, max_tokens, model, temperature)
    elif prov == "gemini":
        yield from _stream_gemini(prompt, max_tokens, model, temperature)
    else:
        yield from _stream_ollama(prompt, max_tokens, temperature)


def _providers_failed(errors) -> RuntimeError:
    """Build an actionable error. Reports each configured provider's real reason
    (hosted ones first — a dead local Ollama fallback shouldn't hide the cause)."""
    if not errors:
        return RuntimeError(
            "No model provider is configured. Set NVIDIA_API_KEY or "
            "GEMINI_API_KEY in the server environment."
        )
    ordered = sorted(errors, key=lambda e: e[0] == "ollama")  # non-ollama first
    detail = " | ".join(f"{prov}: {_safe(exc)}" for prov, exc in ordered)
    return RuntimeError(f"All model providers failed — {detail}")


def call_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
    """Call the chosen provider, failing over to the next available one on error."""
    target = resolve_model(model)
    errors = []
    for prov in _failover_chain(_provider_for(target)):
        if not _provider_ready(prov):
            continue
        try:
            out = _dispatch(prov, prompt, max_tokens, target, temperature, json_mode)
        except Exception as exc:  # noqa: BLE001
            errors.append((prov, exc))
            print(f"[models] {prov} call failed ({_safe(exc)}); trying next provider.")
            continue
        if out and out.strip():
            return out
        # Same failure mode as the streaming path: an empty completion means
        # the provider gave us nothing usable (overloaded, or a reasoning model
        # that spent the whole budget thinking). Callers can't tell that apart
        # from a real answer, so fail over rather than hand back "".
        errors.append((prov, RuntimeError("empty completion")))
        print(f"[models] {prov} returned an empty completion; trying next provider.")
    raise _providers_failed(errors)


def call_model_stream(prompt, max_tokens=1400, model=None, temperature=0.4):
    """Stream from the chosen provider; fail over if it errors before any output."""
    target = resolve_model(model)
    errors = []
    for prov in _failover_chain(_provider_for(target)):
        if not _provider_ready(prov):
            continue
        yielded = False
        try:
            for delta in _dispatch_stream(prov, prompt, max_tokens, target, temperature):
                yielded = True
                yield delta
        except Exception as exc:  # noqa: BLE001
            errors.append((prov, exc))
            if yielded:
                print(f"[models] {prov} stream interrupted ({_safe(exc)}).")
                return
            print(f"[models] {prov} stream failed ({_safe(exc)}); trying next provider.")
            continue
        if yielded:
            return
        # A clean stream that produced NOTHING is a failure, not a success.
        # NVIDIA intermittently answers 200 with an empty SSE body under load,
        # and a reasoning model can also spend its whole token budget on
        # `reasoning_content` and never emit a single `content` delta. Both
        # used to return "" to the caller, which the pipeline then wrote over
        # good notes. Fail over instead.
        errors.append((prov, RuntimeError("stream closed without producing any output")))
        print(f"[models] {prov} stream produced no output; trying next provider.")
    raise _providers_failed(errors)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def safe_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from messy model output."""
    if not text:
        return {}
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except Exception:
        pass

    match = re.search(r"\{.*\}", fenced if fenced else text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return {}
