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
# auth silently stays disabled.
load_dotenv()

try:
    import jwt as _pyjwt  # PyJWT — only needed when SUPABASE_JWT_SECRET is set
except ImportError:  # pragma: no cover
    _pyjwt = None

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or ""
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") or ""

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
    if r.status_code != 200:
        raise HTTPException(
            401, "Invalid or expired session — please sign in again.")

    body = r.json()
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

_hits: dict = {}  # (bucket, identity) -> [timestamps]
_hits_lock = threading.Lock()


def limiter(bucket: str, limit: int, window_sec: int):
    """Return a dependency allowing `limit` calls per `window_sec` per user."""

    async def _dep(request: Request, authorization: str = Header(default="")):
        user = await require_user(request, authorization)
        key = (bucket, user["id"])
        now = time.time()
        with _hits_lock:
            stamps = [t for t in _hits.get(key, []) if now - t < window_sec]
            if len(stamps) >= limit:
                retry = int(window_sec - (now - stamps[0])) + 1
                raise HTTPException(
                    429,
                    f"Rate limit reached — try again in about {retry}s.",
                )
            stamps.append(now)
            _hits[key] = stamps
        return user

    return _dep
