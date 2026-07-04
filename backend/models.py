"""
Model router: a provider-agnostic LLM layer with automatic failover.

Providers (failover order starts from the chosen model's provider):
  - groq   : fast Llama/Gemma inference (best streaming speed)
  - gemini : Google Gemini (large context, generous free tier, multimodal)
  - ollama : local fallback (offline)

If the chosen provider is rate-limited or down, the next available provider is
used automatically — so a Groq rate limit no longer dead-ends a request.

Public API:
    call_model(prompt, max_tokens, model, temperature, json_mode) -> str
    call_model_stream(prompt, max_tokens, model, temperature)     -> generator[str]
    safe_json(text)        -> dict
    get_active_provider()  -> "groq" | "gemini" | "ollama"
    available_models()     -> list[dict]
    default_model()        -> str
    resolve_model(model)   -> str
"""

import os
import re
import json
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_PLACEHOLDER = "your_groq_api_key_here"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
AVAILABLE_GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B · Groq (fast & strong)", "provider": "groq"},
    {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B · Groq (fastest)", "provider": "groq"},
    {"id": "gemma2-9b-it", "label": "Gemma2 9B · Groq", "provider": "groq"},
]

# Gemini
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
AVAILABLE_GEMINI_MODELS = [
    {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash · Google (1M context)", "provider": "gemini"},
    {"id": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash-Lite · Google (cheapest)", "provider": "gemini"},
]
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()

_groq_client = None


class RateLimitError(RuntimeError):
    """Kept for compatibility; failover now handles rate limits transparently."""


def _safe(exc) -> str:
    """Strip API keys out of error text before logging (handles ?key= and &key=)."""
    return re.split(r"[?&]key=", str(exc))[0]


# ---------------------------------------------------------------------------
# Availability / routing
# ---------------------------------------------------------------------------

def _groq_available() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != GROQ_PLACEHOLDER


def _gemini_available() -> bool:
    return bool(GEMINI_API_KEY)


_ollama_reachable = None


def _ollama_available() -> bool:
    """Probe Ollama once and cache it. On hosted deploys (no local Ollama) this
    is False, so a dead localhost fallback never masks the real Groq/Gemini error."""
    global _ollama_reachable
    if _ollama_reachable is None:
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
            _ollama_reachable = True
        except Exception:  # noqa: BLE001
            _ollama_reachable = False
    return _ollama_reachable


def _provider_ready(prov: str) -> bool:
    return {
        "groq": _groq_available(),
        "gemini": _gemini_available(),
        "ollama": _ollama_available(),
    }.get(prov, False)


def _provider_for(model_id: str) -> str:
    return "gemini" if (model_id or "").startswith("gemini") else "groq"


def _failover_chain(primary: str):
    return [primary] + [p for p in ("groq", "gemini", "ollama") if p != primary]


def available_models():
    models = []
    if _groq_available():
        models += AVAILABLE_GROQ_MODELS
    if _gemini_available():
        models += AVAILABLE_GEMINI_MODELS
    return models or AVAILABLE_GROQ_MODELS


def default_model() -> str:
    av = available_models()
    return av[0]["id"] if av else DEFAULT_GROQ_MODEL


def resolve_model(model):
    if model and model.strip():
        return model.strip()
    return default_model()


def get_active_provider() -> str:
    if _groq_available():
        return "groq"
    if _gemini_available():
        return "gemini"
    return "ollama"


# A cheaper "helper" model for the mechanical packaging agents (gatekeeper,
# title, quiz, flashcards). It runs on a SEPARATE free-tier daily token quota
# from the main writing model, so these steps don't burn the primary model's
# budget — the writing/critique that actually determines quality keeps the
# strong model.
HELPER_GROQ_MODEL = os.getenv("HELPER_GROQ_MODEL", "llama-3.1-8b-instant").strip()
HELPER_GEMINI_MODEL = os.getenv("HELPER_GEMINI_MODEL", "gemini-2.0-flash-lite").strip()


def helper_model(main_model=None) -> str:
    """Pick a cheap helper model in the same provider family as `main_model`."""
    target = resolve_model(main_model)
    if _provider_for(target) == "gemini":
        return HELPER_GEMINI_MODEL
    return HELPER_GROQ_MODEL


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _groq_model(model):
    return model if (model and not model.startswith("gemini")) else DEFAULT_GROQ_MODEL


def _groq_messages(prompt):
    return [
        {
            "role": "system",
            "content": "You are a precise, helpful study-notes assistant. "
            "Follow the user's formatting instructions exactly.",
        },
        {"role": "user", "content": prompt},
    ]


def _retry_after(exc, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited request. Honour Groq's
    Retry-After header when present, else exponential backoff (2, 4, 8s)."""
    for attr in ("response", "body"):
        obj = getattr(exc, attr, None)
        headers = getattr(obj, "headers", None)
        if headers:
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra) + 0.5, 30.0)
                except (TypeError, ValueError):
                    pass
    return min(2.0 * (2 ** attempt), 20.0)


def _is_rate_limit(exc) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc) or \
        "rate limit" in str(exc).lower()


GROQ_RATE_LIMIT_RETRIES = int(os.getenv("GROQ_RATE_LIMIT_RETRIES", "4"))


def _groq_create(client, **kwargs):
    """Create a Groq completion, retrying transient 429 rate limits with backoff.
    The map-reduce path fires many calls fast and can hit free-tier TPM limits;
    a short wait lets the per-minute window refill instead of failing the run."""
    for attempt in range(GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc) and attempt < GROQ_RATE_LIMIT_RETRIES:
                wait = _retry_after(exc, attempt)
                print(f"[models] groq rate-limited; retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{GROQ_RATE_LIMIT_RETRIES}).")
                time.sleep(wait)
                continue
            raise


def _call_groq(prompt, max_tokens, model, temperature, json_mode):
    client = _get_groq_client()
    kwargs = dict(
        model=_groq_model(model),
        messages=_groq_messages(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = _groq_create(client, **kwargs)
    return (completion.choices[0].message.content or "").strip()


def _stream_groq(prompt, max_tokens, model, temperature):
    client = _get_groq_client()
    stream = _groq_create(
        client,
        model=_groq_model(model),
        messages=_groq_messages(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            delta = None
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Gemini (REST)
# ---------------------------------------------------------------------------

def _gemini_model(model):
    return model if (model and model.startswith("gemini")) else DEFAULT_GEMINI_MODEL


def _gemini_body(prompt, max_tokens, temperature, json_mode):
    gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    return {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}


def _gemini_text(data) -> str:
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _call_gemini(prompt, max_tokens, model, temperature, json_mode):
    gm = _gemini_model(model)
    url = f"{_GEMINI_BASE}/models/{gm}:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(url, json=_gemini_body(prompt, max_tokens, temperature, json_mode), timeout=120)
    resp.raise_for_status()
    return _gemini_text(resp.json()).strip()


def _stream_gemini(prompt, max_tokens, model, temperature):
    gm = _gemini_model(model)
    url = f"{_GEMINI_BASE}/models/{gm}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    with requests.post(
        url, json=_gemini_body(prompt, max_tokens, temperature, False), stream=True, timeout=300
    ) as resp:
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
# Gemini vision — OCR / image-to-text
# ---------------------------------------------------------------------------

def gemini_available() -> bool:
    return _gemini_available()


def extract_text_from_image(image_bytes: bytes, mime_type: str = "image/jpeg", model=None) -> str:
    """Transcribe study text from an image using Gemini vision."""
    import base64

    gm = _gemini_model(model)  # gemini-2.0-flash is vision-capable
    url = f"{_GEMINI_BASE}/models/{gm}:generateContent?key={GEMINI_API_KEY}"
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
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }
    resp = requests.post(url, json=body, timeout=120)
    resp.raise_for_status()
    return _gemini_text(resp.json()).strip()


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
    if prov == "groq":
        return _call_groq(prompt, max_tokens, model, temperature, json_mode)
    if prov == "gemini":
        return _call_gemini(prompt, max_tokens, model, temperature, json_mode)
    return _call_ollama(prompt, max_tokens, temperature, json_mode)


def _dispatch_stream(prov, prompt, max_tokens, model, temperature):
    if prov == "groq":
        yield from _stream_groq(prompt, max_tokens, model, temperature)
    elif prov == "gemini":
        yield from _stream_gemini(prompt, max_tokens, model, temperature)
    else:
        yield from _stream_ollama(prompt, max_tokens, temperature)


def _providers_failed(errors) -> RuntimeError:
    """Build an actionable error. Reports each configured provider's real reason
    (Groq/Gemini first — a dead local Ollama fallback shouldn't hide the cause)."""
    if not errors:
        return RuntimeError(
            "No model provider is configured. Set GROQ_API_KEY (or GEMINI_API_KEY) "
            "in the server environment."
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
            return _dispatch(prov, prompt, max_tokens, target, temperature, json_mode)
        except Exception as exc:  # noqa: BLE001
            errors.append((prov, exc))
            print(f"[models] {prov} call failed ({_safe(exc)}); trying next provider.")
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
            return
        except Exception as exc:  # noqa: BLE001
            errors.append((prov, exc))
            if yielded:
                print(f"[models] {prov} stream interrupted ({_safe(exc)}).")
                return
            print(f"[models] {prov} stream failed ({_safe(exc)}); trying next provider.")
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
