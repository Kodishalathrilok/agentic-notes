"""
Model router: Groq (primary, cloud) with Ollama (local) fallback.

Public API:
    call_model(prompt, max_tokens, model)        -> str
    call_model_stream(prompt, max_tokens, model) -> generator[str]  (text deltas)
    safe_json(text)                              -> dict
    get_active_provider()                        -> "groq" | "ollama"
    resolve_model(model)                         -> str
    AVAILABLE_GROQ_MODELS                         -> list[dict]
"""

import os
import re
import json
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_PLACEHOLDER = "your_groq_api_key_here"

# Default Groq model (override with GROQ_MODEL in .env). The older
# llama-3.1-70b-versatile was decommissioned on Groq.
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Curated list surfaced to the UI model picker.
AVAILABLE_GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B — best quality"},
    {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B — fastest"},
    {"id": "gemma2-9b-it", "label": "Gemma2 9B — balanced"},
]

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()

# Rate-limit retry policy
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8

_groq_client = None


class RateLimitError(RuntimeError):
    """Raised when Groq is rate-limited and no fallback succeeded."""


def _groq_available() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != GROQ_PLACEHOLDER


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def resolve_model(model: str | None) -> str:
    """Pick a valid Groq model id, falling back to the default."""
    if model and model.strip():
        return model.strip()
    return DEFAULT_GROQ_MODEL


def get_active_provider() -> str:
    return "groq" if _groq_available() else "ollama"


def _is_rate_limit(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return "RateLimit" in name or "429" in str(exc)


# ---------------------------------------------------------------------------
# Non-streaming calls
# ---------------------------------------------------------------------------

def _groq_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": "You are a precise, helpful study-notes assistant. "
            "Follow the user's formatting instructions exactly.",
        },
        {"role": "user", "content": prompt},
    ]


def _call_groq(prompt: str, max_tokens: int, model: str, temperature: float = 0.4, json_mode: bool = False) -> str:
    client = _get_groq_client()
    kwargs = dict(
        model=model,
        messages=_groq_messages(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    return (completion.choices[0].message.content or "").strip()


def _call_ollama(prompt: str, max_tokens: int, temperature: float = 0.4, json_mode: bool = False) -> str:
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


def call_model(
    prompt: str,
    max_tokens: int = 1024,
    model: str | None = None,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> str:
    """Try Groq (with rate-limit retries) then fall back to Ollama."""
    target = resolve_model(model)

    if _groq_available():
        for attempt in range(_MAX_RETRIES):
            try:
                return _call_groq(prompt, max_tokens, target, temperature, json_mode)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit(exc) and attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                print(f"[models] Groq call failed ({exc}); trying Ollama.")
                break

    try:
        return _call_ollama(prompt, max_tokens, temperature, json_mode)
    except Exception as exc:  # noqa: BLE001
        if _groq_available():
            raise RateLimitError(
                "Groq is rate-limited or unavailable and no local Ollama "
                "fallback responded. Please wait a moment and try again."
            ) from exc
        raise RuntimeError(
            "No model backend available. Set a valid GROQ_API_KEY in .env or "
            f"start Ollama (pull `{OLLAMA_MODEL}`). Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Streaming calls
# ---------------------------------------------------------------------------

def _stream_groq(prompt: str, max_tokens: int, model: str, temperature: float = 0.4):
    client = _get_groq_client()
    stream = client.chat.completions.create(
        model=model,
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


def _stream_ollama(prompt: str, max_tokens: int, temperature: float = 0.4):
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


def call_model_stream(prompt: str, max_tokens: int = 1400, model: str | None = None, temperature: float = 0.4):
    """
    Yield text deltas. Tries Groq first; if Groq fails *before* producing any
    output, falls back to Ollama. (A mid-stream failure simply stops.)
    """
    target = resolve_model(model)

    if _groq_available():
        yielded = False
        try:
            for delta in _stream_groq(prompt, max_tokens, target, temperature):
                yielded = True
                yield delta
            return
        except Exception as exc:  # noqa: BLE001
            if yielded:
                print(f"[models] Groq stream interrupted ({exc}).")
                return
            print(f"[models] Groq stream failed ({exc}); trying Ollama.")

    try:
        for delta in _stream_ollama(prompt, max_tokens, temperature):
            yield delta
    except Exception as exc:  # noqa: BLE001
        if _groq_available():
            raise RateLimitError(
                "Groq is rate-limited or unavailable and Ollama did not respond."
            ) from exc
        raise RuntimeError(
            "No model backend available for streaming. "
            f"Last error: {exc}"
        ) from exc


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
