# Ensures the backend package modules are importable during tests.
"""Test-suite isolation.

models.py, auth.py and main.py call load_dotenv(backend/.env) at import time,
and a developer's backend/.env holds REAL provider keys. Without the block
below the suite's behaviour depended on that file (which providers are
"configured", which model is the default, whether auth is on) and a test that
forgot to fake the network could spend real credits.

Everything here runs at conftest import, i.e. BEFORE any test module imports
the app, which is what makes it effective:

1. Secrets and auth settings are pinned to "" in os.environ. python-dotenv
   never overrides a variable that already exists, so the .env values cannot
   come back, and every reader treats "" as unset (`os.getenv(x) or ...`,
   `.strip()` then truthiness).
2. load_dotenv itself is made a no-op, so non-secret values in .env
   (NVIDIA_MODEL, OLLAMA_URL, MAX_TEXT_CHARS, ...) cannot change routing or
   limits either. Model-name variables are REMOVED rather than blanked: they
   are read with os.getenv(name, default), where "" would replace the code
   default instead of meaning "unset".
3. OLLAMA_URL points at a closed local port, so a developer's running Ollama
   is never probed or used.
4. An autouse fixture fails any test that opens a real outbound HTTP
   connection (requests or httpx). Tests that fake a provider patch higher up
   (models._dispatch, requests.post, ...) and never reach the transport.
"""

import os

_BLANK = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY", "GROQ_API_KEY",
    "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET",
    "ALLOWED_EMAILS", "REQUIRE_AUTH", "ALLOW_ANONYMOUS",
)
_UNSET = (
    "NVIDIA_MODEL", "NVIDIA_MODELS", "HELPER_NVIDIA_MODEL", "GEMINI_MODEL",
    "HELPER_GEMINI_MODEL", "NVIDIA_BASE_URL", "OLLAMA_MODEL", "MAX_TEXT_CHARS",
    "MAX_CONCURRENT_GENERATIONS", "GENERATION_DEADLINE_S",
)
for _name in _BLANK:
    os.environ[_name] = ""
for _name in _UNSET:
    os.environ.pop(_name, None)
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"  # discard port: always refused
os.environ.setdefault("DEV", "1")

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False  # modules do `from dotenv import load_dotenv`

import ipaddress  # noqa: E402
from urllib.parse import urlsplit  # noqa: E402

import pytest  # noqa: E402


def _is_local(url) -> bool:
    host = (urlsplit(str(url)).hostname or "").lower()
    if host in ("localhost", "testserver"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class RealNetworkCallError(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Fail loudly on any real outbound HTTP request."""
    import requests.adapters
    import httpx

    real_send = requests.adapters.HTTPAdapter.send

    def guarded_send(self, request, *a, **k):
        if not _is_local(request.url):
            raise RealNetworkCallError(
                f"test attempted a real HTTP request to {urlsplit(request.url).hostname}; "
                f"fake the provider instead")
        return real_send(self, request, *a, **k)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", guarded_send)

    real_h = httpx.HTTPTransport.handle_request
    real_ah = httpx.AsyncHTTPTransport.handle_async_request

    def guarded_h(self, request):
        if not _is_local(request.url):
            raise RealNetworkCallError(f"test attempted a real HTTP request to {request.url.host}")
        return real_h(self, request)

    async def guarded_ah(self, request):
        if not _is_local(request.url):
            raise RealNetworkCallError(f"test attempted a real HTTP request to {request.url.host}")
        return await real_ah(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded_h)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_ah)
    yield
