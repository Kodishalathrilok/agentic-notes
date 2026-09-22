"""
Server-hardening tests: SSRF guard, URL scheme checks, request size limits,
and the production auth-config gate.
No network access needed — private/loopback addresses are validated locally.
"""

import importlib

import pytest
from fastapi import HTTPException

import auth
import main


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "172.16.0.9", "0.0.0.0"],
)
def test_private_hosts_blocked(host):
    assert main._host_resolves_public(host) is False


def test_public_ip_allowed():
    assert main._host_resolves_public("8.8.8.8") is True


def test_non_http_scheme_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("ftp://example.com/file")
    assert exc.value.status_code == 422


def test_loopback_url_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("http://127.0.0.1:8000/api/health")
    assert exc.value.status_code == 422


def test_metadata_endpoint_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("http://169.254.169.254/latest/meta-data/")
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Request size limits (Pydantic max_length)
# ---------------------------------------------------------------------------

def test_chat_request_rejects_oversized_question():
    with pytest.raises(Exception):
        main.ChatRequest(question="x" * 5000)


def test_chat_request_rejects_oversized_notes():
    with pytest.raises(Exception):
        main.ChatRequest(notes="x" * 300001)


def test_url_request_rejects_oversized_url():
    with pytest.raises(Exception):
        main.UrlRequest(url="https://example.com/" + "a" * 2000)


def test_normal_sizes_accepted():
    main.ChatRequest(notes="some notes", question="what is this?")
    main.UrlRequest(url="https://example.com/article")
    main.RegenRequest(notes="notes body")


# ---------------------------------------------------------------------------
# Production auth gate
#
# Auth used to fail OPEN: no Supabase settings -> REQUIRE_AUTH False -> every
# endpoint served anonymous callers, silently. These pin the loud behaviour.
# ---------------------------------------------------------------------------

def _auth_with(monkeypatch, **env):
    """Reimport auth.py under a specific environment."""
    for key in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET",
                "REQUIRE_AUTH", "ALLOW_ANONYMOUS", "DEV"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # load_dotenv() must not put the developer's own .env back in play.
    monkeypatch.setattr(auth, "load_dotenv", lambda *a, **k: None)
    return importlib.reload(auth)


@pytest.fixture(autouse=True)
def _restore_auth(monkeypatch):
    """Reload auth under the ORIGINAL environment after each test.

    The reload must not depend on fixture teardown order: another autouse
    fixture (conftest's _no_real_network) also requests monkeypatch, so
    monkeypatch may be torn down AFTER this fixture. Undo the test's env
    patches explicitly first, otherwise the reload would read e.g. DEV=0 +
    SUPABASE_URL and leak REQUIRE_AUTH=True into the next test.
    """
    yield
    monkeypatch.undo()
    importlib.reload(auth)


def test_production_without_supabase_refuses_to_start(monkeypatch):
    mod = _auth_with(monkeypatch, DEV="0")
    assert mod.auth_required() is False
    with pytest.raises(RuntimeError, match="Refusing to start"):
        mod.verify_auth_config()


def test_production_with_supabase_starts(monkeypatch):
    mod = _auth_with(
        monkeypatch, DEV="0",
        SUPABASE_URL="https://example.supabase.co", SUPABASE_ANON_KEY="anon-key",
    )
    assert mod.auth_required() is True
    mod.verify_auth_config()


def test_jwt_secret_alone_is_enough(monkeypatch):
    mod = _auth_with(monkeypatch, DEV="0", SUPABASE_JWT_SECRET="s3cret")
    assert mod.auth_required() is True
    mod.verify_auth_config()


def test_anonymous_production_requires_explicit_optin(monkeypatch):
    mod = _auth_with(monkeypatch, DEV="0", ALLOW_ANONYMOUS="true")
    assert mod.auth_required() is False
    mod.verify_auth_config()  # allowed, because it was asked for by name


def test_local_dev_without_supabase_still_works(monkeypatch):
    mod = _auth_with(monkeypatch, DEV="1")
    assert mod.auth_required() is False
    mod.verify_auth_config()


# ---------------------------------------------------------------------------
# Rate limiting — the daily window is what protects a free tier's token quota,
# since the short sliding window resets forever.
# ---------------------------------------------------------------------------

async def _call(dep, ip="1.2.3.4"):
    class _Client:
        host = ip

    class _Req:
        client = _Client()

    return await dep(_Req(), "")


@pytest.fixture
def clean_hits():
    auth._hits.clear()
    yield
    auth._hits.clear()


@pytest.mark.asyncio
async def test_short_window_limit_enforced(clean_hits):
    dep = auth.limiter("t-burst", 2, 600)
    await _call(dep)
    await _call(dep)
    with pytest.raises(HTTPException) as exc:
        await _call(dep)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_daily_cap_enforced_under_the_short_limit(clean_hits):
    """3 allowed per 10 min, but only 2 per day — the daily cap must win."""
    dep = auth.limiter("t-daily", 3, 600, daily=2)
    await _call(dep)
    await _call(dep)
    with pytest.raises(HTTPException) as exc:
        await _call(dep)
    assert exc.value.status_code == 429
    assert "Daily limit" in exc.value.detail


@pytest.mark.asyncio
async def test_rejected_call_does_not_consume_other_windows(clean_hits):
    dep = auth.limiter("t-nodouble", 5, 600, daily=1)
    await _call(dep)
    for _ in range(3):
        with pytest.raises(HTTPException):
            await _call(dep)
    # The short window should have recorded exactly one allowed call.
    assert len(auth._hits[("t-nodouble", 600, "anon:1.2.3.4")]) == 1


@pytest.mark.asyncio
async def test_config_values_are_stripped(monkeypatch):
    """A pasted secret picks up a trailing newline far too easily.

    The anon key is sent as an HTTP header; http.client raises ValueError on
    header values containing newlines, which is not a RequestException — so it
    escaped the handler and every authenticated request 500'd.
    """
    mod = _auth_with(
        monkeypatch,
        DEV="0",
        SUPABASE_URL="https://example.supabase.co/\n",
        SUPABASE_ANON_KEY="anon-key\n",
        SUPABASE_JWT_SECRET="  secret  ",
    )
    assert mod.SUPABASE_URL == "https://example.supabase.co"
    assert mod.SUPABASE_ANON_KEY == "anon-key"
    assert mod.SUPABASE_JWT_SECRET == "secret"
    assert "\n" not in mod.SUPABASE_ANON_KEY


@pytest.mark.asyncio
async def test_malformed_config_reports_503_not_500(monkeypatch):
    """Never leak a bare 500 when the credentials themselves are unusable.

    Depending on where the newline sits, requests raises InvalidHeader (a
    RequestException) or lets http.client raise a plain ValueError. Both must
    surface as a handled 503 — the ValueError path is what escaped in
    production and 500'd every authenticated request.
    """
    mod = _auth_with(
        monkeypatch,
        DEV="0",
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_ANON_KEY="anon-key",
    )
    monkeypatch.setattr(mod, "SUPABASE_ANON_KEY", "bad\nkey")
    with pytest.raises(HTTPException) as exc:
        mod._verify_remote("some-token")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_limits_are_per_identity(clean_hits):
    dep = auth.limiter("t-peruser", 1, 600, daily=1)
    await _call(dep, ip="1.1.1.1")
    await _call(dep, ip="2.2.2.2")  # different caller, own bucket
    with pytest.raises(HTTPException):
        await _call(dep, ip="1.1.1.1")