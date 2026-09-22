"""
Authentication + rate limiting for the API.

Every expensive endpoint depends on `require_user`, which:
  1. Reads the `Authorization: Bearer <token>` header (a Supabase access token
     that the frontend attaches to every API call).
  2. Verifies the token:
       - Locally with the project's JWT secret (SUPABASE_JWT_SECRET, HS256),
         which is fast and needs no network call, OR
       - By asking Supabase itself (GET {SUPABASE_URL}/auth/v1/user), which
         works with any signing configuration. Results are cached briefly.
  3. Optionally enforces an email allowlist (ALLOWED_EMAILS) so only approved
     accounts can use the app even if strangers create Supabase accounts.

If no Supabase configuration is present at all, auth is disabled so local
development without Supabase keeps working exactly as before. Set
REQUIRE_AUTH=true/false to override the automatic choice.

`limiter(...)` returns a FastAPI dependency implementing a simple sliding
window rate limit per user (or per IP when auth is off). In-memory only —
resets on restart, which is fine for a single-process deployment.
"""

import os
import time
import logging
import ipaddress
import threading

import requests as _requests
from fastapi import Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# auth.py is imported before main.py calls load_dotenv(), so load the .env
# here too — otherwise SUPABASE_* vars from backend/.env are invisible and
# auth silently stays disabled. Anchored to backend/, not the cwd — see the
# note in models.py.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

try:
    import jwt as _pyjwt  # PyJWT — only needed when SUPABASE_JWT_SECRET is set
except ImportError:  # pragma: no cover
    _pyjwt = None

def _clean_env(name: str) -> str:
    """Read an env var, stripping surrounding whitespace.

    Pasting a key into a hosting provider's secret box very easily picks up a
    trailing newline. It is invisible in every UI, and it made EVERY
    authenticated request fail with a bare 500: the anon key goes into an HTTP
    header, and http.client rejects header values containing newlines (header
    injection defence) with a ValueError — not a RequestException, so the
    handler below never caught it.
    """
    return (os.getenv(name) or "").strip()


SUPABASE_URL = _clean_env("SUPABASE_URL").rstrip("/")
SUPABASE_ANON_KEY = _clean_env("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = _clean_env("SUPABASE_JWT_SECRET")

# Comma-separated list of emails allowed to use the app. Empty = any
# authenticated Supabase user is allowed.
ALLOWED_EMAILS = {
    e.strip().lower()
    for e in (os.getenv("ALLOWED_EMAILS") or "").split(",")
    if e.strip()
}

_default_require = bool(SUPABASE_JWT_SECRET or (
    SUPABASE_URL and SUPABASE_ANON_KEY))
REQUIRE_AUTH = (os.getenv("REQUIRE_AUTH") or ("true" if _default_require else "false")).lower() in (
    "1",
    "true",
    "yes",
)

# Running with no authentication at all must be asked for by name.
ALLOW_ANONYMOUS = (os.getenv("ALLOW_ANONYMOUS") or "false").lower() in (
    "1", "true", "yes")

# The Dockerfile sets DEV=0, so a container start is always treated as production.
_IS_DEV = (os.getenv("DEV") or "1").lower() in ("1", "true", "yes")


# How many reverse proxies in front of this app APPEND to X-Forwarded-For.
# Only used for the anonymous (auth-off) identity; with auth on, the identity
# is the verified user id. Every proxy appends the address it received the
# connection from, so the entry `hops` places from the RIGHT was written by
# our own outermost proxy and is the real client; everything to its left came
# from the client and is spoofable. Hugging Face Spaces (and Render/Railway/
# Fly) put exactly one such proxy in front of the container, hence the
# default of 1. The proxies' IP ranges are not published, so they cannot be
# allowlisted instead. With NO proxy in front (direct exposure) this MUST be
# 0 - otherwise the client writes the entry that is read - and uvicorn must
# not trust forwarders either: set FORWARDED_ALLOW_IPS=127.0.0.1 (the
# Dockerfile defaults it to '*'). If that is forgotten, client_ip detects
# uvicorn's rewrite and refuses to use the spoofed address.
TRUSTED_PROXY_HOPS = max(0, int(os.getenv("TRUSTED_PROXY_HOPS", "1") or 0))

# Identity used when X-Forwarded-For is present but the configured hop's entry
# is not an address. Deliberately NOT request.client: under uvicorn's
# --proxy-headers --forwarded-allow-ips='*' that is the LEFTMOST entry, which
# the client wrote. A proxy that appends always writes a valid address, so
# only hand-made headers land in this one shared bucket.
_UNPARSABLE_XFF_ID = "xff-unparsable"
_warned_unparsable_xff = False

# Identity when TRUSTED_PROXY_HOPS=0 but uvicorn has ALREADY replaced the peer
# with a client-written X-Forwarded-For entry (it trusts forwarders, i.e.
# FORWARDED_ALLOW_IPS='*', which the Dockerfile sets for the proxied
# platforms). The real peer is gone by then, so no per-client identity is
# trustworthy; these requests share one bucket instead of minting new ones.
_REWRITTEN_PEER_ID = "untrusted-forwarded-peer"
_warned_rewritten_peer = False


def _forwarded_ip(entry: str):
    """Parse one X-Forwarded-For entry: `1.2.3.4`, `1.2.3.4:5678`, `2001:db8::1`
    or `[2001:db8::1]:443`. Returns the normalised address or None."""
    entry = entry.strip()
    if entry.startswith("["):
        end = entry.find("]")
        entry = entry[1:end] if end > 0 else ""
    elif entry.count(":") == 1:  # IPv4 with a port (bare IPv6 has 2+ colons)
        entry = entry.split(":", 1)[0]
    try:
        return str(ipaddress.ip_address(entry))
    except ValueError:
        return None


def client_ip(request) -> str:
    """The caller's address for per-IP limits (see TRUSTED_PROXY_HOPS).

    Reads the raw X-Forwarded-For header itself instead of trusting
    request.client, which uvicorn's --proxy-headers --forwarded-allow-ips='*'
    sets to the LEFTMOST (client-supplied) entry. request.client is only used
    when there is no X-Forwarded-For at all, in which case uvicorn has not
    rewritten it and it is the socket peer.
    """
    global _warned_unparsable_xff, _warned_rewritten_peer
    peer = request.client.host if request.client else "local"
    headers = getattr(request, "headers", None)
    values = headers.getlist("x-forwarded-for") if headers is not None else []
    entries = [e.strip() for v in values for e in v.split(",") if e.strip()]
    if TRUSTED_PROXY_HOPS <= 0:
        # Direct exposure: the socket peer is the identity - unless uvicorn's
        # ProxyHeadersMiddleware rewrote it from X-Forwarded-For. It does so
        # only when the header is present, and always sets the port to 0,
        # which a real TCP peer never has; that combination means the value
        # came from the client.
        if entries and request.client is not None and getattr(request.client, "port", None) == 0:
            if not _warned_rewritten_peer:
                _warned_rewritten_peer = True
                logging.getLogger("agentic").warning(
                    "TRUSTED_PROXY_HOPS=0 but uvicorn trusts X-Forwarded-For "
                    "(FORWARDED_ALLOW_IPS); the client address is spoofable, so such "
                    "requests share one rate-limit identity. On direct exposure set "
                    "FORWARDED_ALLOW_IPS=127.0.0.1.")
            return _REWRITTEN_PEER_ID
        return peer
    if not entries:
        return peer
    parsed = _forwarded_ip(entries[max(0, len(entries) - TRUSTED_PROXY_HOPS)])
    if parsed is None:
        if not _warned_unparsable_xff:
            _warned_unparsable_xff = True
            logging.getLogger("agentic").warning(
                "X-Forwarded-For entry at hop %d is not an IP address; such requests "
                "share one rate-limit identity. Check TRUSTED_PROXY_HOPS.",
                TRUSTED_PROXY_HOPS)
        return _UNPARSABLE_XFF_ID
    return parsed


def auth_required() -> bool:
    """Whether tokens are actually being verified. Surfaced on /api/health."""
    return REQUIRE_AUTH


def verify_auth_config() -> None:
    """Refuse to start an unauthenticated production server.

    This used to fail OPEN: with no Supabase settings REQUIRE_AUTH became
    False, `require_user` waved every caller through as anonymous, and the
    frontend's sign-in gate was decorative — anyone who knew the URL could
    spend model credits. The failure was silent, which is what made it
    dangerous. In production it is now loud and fatal.
    """
    if REQUIRE_AUTH or _IS_DEV or ALLOW_ANONYMOUS:
        return
    raise RuntimeError(
        "Refusing to start: authentication is OFF in a production build.\n"
        "\n"
        "  Every endpoint would serve anonymous callers, so anyone who knows\n"
        "  this URL could spend your model credits. The frontend's sign-in\n"
        "  button would have no effect.\n"
        "\n"
        "  Fix it by setting these on the host (they are SEPARATE from the\n"
        "  VITE_-prefixed variables the frontend is built with):\n"
        "      SUPABASE_URL=https://<project>.supabase.co\n"
        "      SUPABASE_ANON_KEY=<anon key>\n"
        "  or, to verify tokens locally without a network call:\n"
        "      SUPABASE_JWT_SECRET=<JWT secret>\n"
        "\n"
        "  To genuinely run this open to the public, set ALLOW_ANONYMOUS=true."
    )

# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

# token -> (user_dict, expires_at) cache for the network-verification path,
# so a burst of requests doesn't hammer Supabase's /auth/v1/user endpoint.
_user_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # seconds

# Negative cache: token -> time until which it is known to be rejected, so
# replaying the same junk token costs no outbound call. Keyed by the exact
# token, so it can never reject a DIFFERENT token - in particular never a
# valid one (a per-IP failure limit did: one client behind a shared NAT, or a
# spoofed header, could lock everyone else out). Only a definite 401/403 from
# Supabase is cached, never an outage.
_bad_tokens: dict = {}
_BAD_TOKEN_TTL = 60  # seconds
_BAD_TOKEN_MAX = 5000


def _verify_local(token: str):
    """Verify with the project's JWT secret (HS256). Returns user dict."""
    if _pyjwt is None:
        raise HTTPException(
            500, "PyJWT not installed but SUPABASE_JWT_SECRET is set")
    try:
        claims = _pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except Exception:
        raise HTTPException(
            401, "Invalid or expired session — please sign in again.")
    return {"id": claims.get("sub"), "email": (claims.get("email") or "").lower()}


def _cached_user(token: str):
    """Return the cached user for this token, or None if absent/expired."""
    with _cache_lock:
        hit = _user_cache.get(token)
        if hit and hit[1] > time.time():
            return hit[0]
    return None


def _token_exp(token: str):
    """The token's `exp` claim, read WITHOUT verifying the signature - used only
    to stop caching a user past the token's own expiry (Supabase has already
    verified it). None if it isn't a readable JWT or has no numeric exp."""
    if _pyjwt is None:
        return None
    try:
        claims = _pyjwt.decode(token, options={"verify_signature": False})
    except Exception:  # noqa: BLE001
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return exp if isinstance(exp, (int, float)) and not isinstance(exp, bool) else None


def _verify_remote(token: str):
    """Verify by asking Supabase who this token belongs to (cached)."""
    now = time.time()
    cached = _cached_user(token)
    if cached is not None:
        return cached
    with _cache_lock:
        if _bad_tokens.get(token, 0) > now:
            raise HTTPException(
                401, "Invalid or expired session — please sign in again.")

    try:
        r = _requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}",
                     "apikey": SUPABASE_ANON_KEY},
            timeout=8,
        )
    except _requests.RequestException:
        raise HTTPException(
            503, "Could not reach the auth service — try again shortly.")
    except ValueError:
        # A malformed SUPABASE_URL or ANON_KEY (stray newline, control char)
        # never reaches the network — it dies building the request. Report it
        # as the server misconfiguration it is instead of a bare 500.
        raise HTTPException(
            503,
            "The auth service is misconfigured on the server "
            "(check SUPABASE_URL and SUPABASE_ANON_KEY).",
        )
    if r.status_code != 200:
        if r.status_code in (401, 403):
            with _cache_lock:
                if len(_bad_tokens) >= _BAD_TOKEN_MAX:
                    _bad_tokens.clear()
                _bad_tokens[token] = now + _BAD_TOKEN_TTL
        raise HTTPException(
            401, "Invalid or expired session — please sign in again.")

    try:
        body = r.json()
    except ValueError:
        raise HTTPException(
            502, "The auth service returned an unreadable response.")
    user = {"id": body.get("id"), "email": (body.get("email") or "").lower()}
    # Never cache past the token's own expiry: an expired token must stop
    # working when it expires, not up to _CACHE_TTL later.
    expires_at = now + _CACHE_TTL
    exp = _token_exp(token)
    if exp is not None:
        expires_at = min(expires_at, exp)
    if expires_at > now:
        with _cache_lock:
            # Opportunistic cleanup so the cache can't grow unbounded.
            if len(_user_cache) > 2000:
                _user_cache.clear()
            _user_cache[token] = (user, expires_at)
    return user


async def require_user(
    request: Request,
    authorization: str = Header(default=""),
):
    """FastAPI dependency: returns the authenticated user, or raises 401/403.

    When auth is disabled (no Supabase config, local dev), returns an
    anonymous user keyed by client IP so rate limiting still has an identity.
    """
    if not REQUIRE_AUTH:
        return {"id": f"anon:{client_ip(request)}", "email": ""}

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Sign in required.")

    if SUPABASE_JWT_SECRET:
        user = _verify_local(token)
    else:
        # _verify_remote makes a blocking HTTP call (up to 8s). One uvicorn
        # worker serves every SSE stream, so run it off the event loop or a
        # slow Supabase stalls all of them. Cache hits skip the thread hop.
        user = _cached_user(token)
        if user is None:
            user = await run_in_threadpool(_verify_remote, token)

    if ALLOWED_EMAILS and user.get("email") not in ALLOWED_EMAILS:
        raise HTTPException(
            403,
            "This account is not authorized to use the app. Contact the owner for access.",
        )
    return user


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, in-memory)
# ---------------------------------------------------------------------------

_hits: dict = {}  # (bucket, window_sec, identity) -> [timestamps]
_hits_lock = threading.Lock()

# Memory bound for _hits. A key's list is only pruned when that identity calls
# again, so keys of identities that never return would live forever. Once the
# dict passes the threshold, one sweep drops expired stamps from EVERY key and
# deletes the keys left empty. The threshold then moves to twice the surviving
# size, so a population of genuinely active identities can't make every
# request pay for a full sweep - the cost stays amortised O(1) per request.
_HITS_SWEEP_MIN = 5000
_hits_sweep_at = _HITS_SWEEP_MIN


def _sweep_expired_hits(now: float) -> None:
    """Drop expired timestamps across all keys. Caller holds _hits_lock."""
    global _hits_sweep_at
    for key in list(_hits):
        win = key[1]
        live = [t for t in _hits[key] if now - t < win]
        if live:
            _hits[key] = live
        else:
            del _hits[key]
    _hits_sweep_at = max(_HITS_SWEEP_MIN, 2 * len(_hits))

DAY_SEC = 86400


def _retry_message(seconds: int) -> str:
    if seconds >= 3600:
        return f"Daily limit reached — this resets in about {seconds // 3600}h."
    if seconds >= 120:
        return f"Rate limit reached — try again in about {seconds // 60} minutes."
    return f"Rate limit reached — try again in about {seconds}s."


def limiter(bucket: str, limit: int, window_sec: int, daily: int = 0):
    """Return a dependency allowing `limit` calls per `window_sec` per user.

    `daily` adds a second 24-hour window on the same bucket. The short window
    stops bursts; the daily one stops one account from draining the provider's
    daily token quota over the course of a day — the free tiers cap on tokens
    per DAY, which a short sliding window does nothing to protect.
    """
    windows = [(limit, window_sec)]
    if daily:
        windows.append((daily, DAY_SEC))

    async def _dep(request: Request, authorization: str = Header(default="")):
        user = await require_user(request, authorization)
        now = time.time()
        with _hits_lock:
            # Check every window before recording anything, so a call that gets
            # rejected doesn't also count against the other windows.
            pruned = []
            for lim, win in windows:
                key = (bucket, win, user["id"])
                stamps = [t for t in _hits.get(key, []) if now - t < win]
                pruned.append((key, stamps))
                if len(stamps) >= lim:
                    _hits[key] = stamps
                    raise HTTPException(
                        429, _retry_message(int(win - (now - stamps[0])) + 1)
                    )
            for key, stamps in pruned:
                stamps.append(now)
                _hits[key] = stamps
            # Bounded memory — daily windows keep keys alive for 24h, and a
            # key is otherwise only pruned when its identity calls again.
            if len(_hits) > _hits_sweep_at:
                _sweep_expired_hits(now)
        return user

    return _dep
