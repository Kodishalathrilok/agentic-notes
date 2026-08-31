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
import threading

import requests as _requests
from fastapi import Header, HTTPException, Request
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


def _verify_remote(token: str):
    """Verify by asking Supabase who this token belongs to (cached)."""
    now = time.time()
    with _cache_lock:
        hit = _user_cache.get(token)
        if hit and hit[1] > now:
            return hit[0]

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
        raise HTTPException(
            401, "Invalid or expired session — please sign in again.")

    try:
        body = r.json()
    except ValueError:
        raise HTTPException(
            502, "The auth service returned an unreadable response.")
    user = {"id": body.get("id"), "email": (body.get("email") or "").lower()}
    with _cache_lock:
        # Opportunistic cleanup so the cache can't grow unbounded.
        if len(_user_cache) > 2000:
            _user_cache.clear()
        _user_cache[token] = (user, now + _CACHE_TTL)
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
        ip = request.client.host if request.client else "local"
        return {"id": f"anon:{ip}", "email": ""}

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Sign in required.")

    user = _verify_local(
        token) if SUPABASE_JWT_SECRET else _verify_remote(token)

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
            # Opportunistic cleanup — daily windows keep keys alive for 24h, so
            # without this the dict grows with every identity ever seen.
            if len(_hits) > 5000:
                for key in [k for k, v in _hits.items() if not v]:
                    del _hits[key]
        return user

    return _dep
