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


def _provider_ready(prov: str) -> bool:
    return {
        "groq": _groq_available(),
        "gemini": _gemini_available(),
        "ollama": True,
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
    completion = client.chat.completions.create(**kwargs)
    return (completion.choices[0].message.content or "").strip()


def _stream_groq(prompt, max_tokens, model, temperature):
    client = _get_groq_client()
    stream = client.chat.completions.create(
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


def call_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
    """Call the chosen provider, failing over to the next available one on error."""
    target = resolve_model(model)
    last = None
    for prov in _failover_chain(_provider_for(target)):
        if not _provider_ready(prov):
            continue
        try:
            return _dispatch(prov, prompt, max_tokens, target, temperature, json_mode)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[models] {prov} call failed ({_safe(exc)}); trying next provider.")
    raise RuntimeError(
        f"All model providers failed. Last error: {_safe(last) if last else 'none available'}"
    )


def call_model_stream(prompt, max_tokens=1400, model=None, temperature=0.4):
    """Stream from the chosen provider; fail over if it errors before any output."""
    target = resolve_model(model)
    last = None
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
            last = exc
            if yielded:
                print(f"[models] {prov} stream interrupted ({_safe(exc)}).")
                return
            print(f"[models] {prov} stream failed ({_safe(exc)}); trying next provider.")
    raise RuntimeError(
        f"All streaming providers failed. Last error: {_safe(last) if last else 'none available'}"
    )


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
