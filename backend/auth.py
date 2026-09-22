"""
Authentication + rate limiting for the API.

Every expensive endpoint depends on `require_user`, which:
  1. Reads the `Authorization: Bearer <token>` header (a Supabase access token
     that the frontend attaches to every API call).
  2. Verifies the token, choosing the path from the token header's `alg`:
       - ES256/RS256 (Supabase asymmetric signing keys) with SUPABASE_URL set:
         locally against the project's public keys from
         {SUPABASE_URL}/auth/v1/.well-known/jwks.json (cached). This result is
         authoritative - a bad signature or claim is a 401, never a fallback.
         Only if the JWKS itself is unusable (unreachable, 5xx, empty key
         set) does it fall back to asking Supabase (below).
       - HS256 (legacy shared-secret projects): locally with
         SUPABASE_JWT_SECRET if set, otherwise by asking Supabase.
       - Not a JWT at all, or any other alg (`none`, HS384, ...): rejected
         outright, with no network call.
       "Asking Supabase" is GET {SUPABASE_URL}/auth/v1/user, which works with
       any signing configuration. It is only tried for tokens whose unverified
       exp/aud/iss could be valid; results are cached briefly, and calls are
       admitted FIFO under a global (AUTH_VERIFY_CONCURRENCY) and a
       per-client (AUTH_VERIFY_PER_CLIENT) cap.
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
import asyncio
import weakref
import concurrent.futures
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
    if SUPABASE_JWT_SECRET and not SUPABASE_URL:
        logging.getLogger("agentic").warning(
            "SUPABASE_JWT_SECRET is set without SUPABASE_URL: only legacy HS256 "
            "tokens can be verified. Tokens signed with asymmetric keys "
            "(ES256/RS256) will be rejected; set SUPABASE_URL (and "
            "SUPABASE_ANON_KEY) to verify them via the project's JWKS.")
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
        if r.status_code in (400, 401, 403, 422):
            # Supabase looked at the token and rejected it.
            raise HTTPException(
                401, "Invalid or expired session — please sign in again.")
        # 429 / 5xx / anything unexpected is Supabase's problem, not the
        # user's session: telling them to sign in again would be wrong (and
        # would not help). Not negatively cached either.
        raise HTTPException(
            503, "The auth service is busy — try again in a moment.")

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


# ---------------------------------------------------------------------------
# Asymmetric keys (ES256/RS256) via the project's JWKS
# ---------------------------------------------------------------------------
#
# Verification order in require_user (fail closed at every step):
#   0. not a JWT (no parsable JOSE header) -> 401 at once. Supabase access
#      tokens are always JWTs, so nothing valid is lost and junk reaches no one.
#   1. header alg ES256/RS256 and SUPABASE_URL set -> verify against the JWKS.
#      Authoritative: signature/claims failure = 401, unknown kid = 401 (after
#      at most one rate-limited refresh). No fallback, so junk ES256/RS256
#      tokens cost zero outbound calls once the JWKS is cached.
#   2. JWKS unusable (unreachable, non-200, unparsable, no usable keys, and no
#      cached copy that has the kid) -> GET /auth/v1/user (Supabase decides).
#      Never "accept the token".
#   3. header alg HS256 (legacy projects) -> SUPABASE_JWT_SECRET if set, else
#      GET /auth/v1/user, as before.
#   4. any other header alg (none, HS384, PS256, ...) -> 401.
#
# Before every remote call (2, 3), cheap UNVERIFIED claim checks (exp in the
# future, aud, iss) refuse tokens that could never be valid without spending
# an outbound call. Remote calls are then admitted FIFO, bounded globally
# (AUTH_VERIFY_CONCURRENCY) and per client address (AUTH_VERIFY_PER_CLIENT),
# so one flooding client can't queue out everybody else.
#
# The JWKS fetch runs on its own single-flight worker thread and never waits
# behind remote verifications. Cached keys keep being served past their TTL
# while a refresh runs or fails (stale-while-revalidate, up to _JWKS_MAX_AGE).
#
# The algorithm used to verify is derived from the KEY (its `alg`, or its
# kty/crv), never from the token header, and only ES256/RS256 are allowed on
# this path - so an HS256 token "signed" with the public key as an HMAC
# secret (algorithm confusion) or an alg=none token can never pass here.

_ASYM_ALGS = ("ES256", "RS256")
_JWKS_TTL = max(30, int(os.getenv("AUTH_JWKS_TTL_S", "600") or 600))
# Minimum gap between JWKS fetch attempts (unknown kid, failure retry,
# revalidation): key rotation still works, but a stream of made-up kids (or
# an outage) cannot turn every request into a JWKS fetch.
_JWKS_REFRESH_MIN = 60
# Hard cap on serving cached keys past the TTL when refreshes keep FAILING.
# While the JWKS is reachable a refresh starts as soon as the TTL passes, so
# a removed/revoked key still stops verifying within about the TTL; this cap
# only matters during a prolonged JWKS outage. 24h rides out any realistic
# Supabase incident - during which /auth/v1/user is most likely down too, so
# dropping the keys would lock every user out - while still bounding how long
# a key revoked during such an outage keeps verifying. Signing keys live for
# months, so a day-old copy is almost always the current one.
_JWKS_MAX_AGE = 24 * 3600
_JWKS_LEEWAY = 10  # seconds of clock skew tolerated on exp/nbf/iat
_JWKS_FETCH_WAIT_S = 8.0  # how long a request waits for an in-flight fetch
_clock = time.monotonic  # patched in tests

# Expected `iss`. Supabase sets {SUPABASE_URL}/auth/v1; a project reached via
# a different hostname (custom domain, self-hosted behind a gateway) can say
# what its tokens actually carry.
AUTH_ISSUER = _clean_env("AUTH_ISSUER")
_warned_issuer: set = set()

_jwks_lock = threading.Lock()  # guards _jwks_state
_jwks_state: dict = {
    "keys": {},          # kid -> (key object, alg)
    "fetched_at": None,  # clock() of the last successful fetch
    "attempt_at": None,  # clock() of the last fetch attempt (any outcome)
    "failed_at": None,   # clock() of the last failed/empty fetch, None after a success
    "inflight": None,    # concurrent Future of the running fetch
}
# One dedicated thread: fetches are single-flight, and never compete with
# remote verifications for AUTH_VERIFY_CONCURRENCY or the shared threadpool.
_jwks_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="jwks")

# Outbound /auth/v1/user calls: FIFO admission, AUTH_VERIFY_CONCURRENCY at a
# time; waiters wait on the event loop (not in a thread) for up to
# AUTH_VERIFY_WAIT_S, then get 503. A client address may have at most
# AUTH_VERIFY_PER_CLIENT verifications queued or running; beyond that it gets
# 503 at once, so a single flooding client can't fill the queue.
AUTH_VERIFY_CONCURRENCY = max(1, int(os.getenv("AUTH_VERIFY_CONCURRENCY", "8") or 8))
AUTH_VERIFY_PER_CLIENT = max(1, int(os.getenv("AUTH_VERIFY_PER_CLIENT", "2") or 2))
AUTH_VERIFY_WAIT_S = 3.0
_slots_by_loop: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_inflight_by_client: dict = {}  # touched only from the event loop

_INVALID = "Invalid or expired session — please sign in again."


class _JwksUnavailable(Exception):
    """The JWKS can't be used right now: fall back to remote verification."""


def _expected_issuer() -> str:
    return AUTH_ISSUER or f"{SUPABASE_URL}/auth/v1"


def _warn_issuer_once(where: str, received) -> None:
    """Log expected vs received issuer the first time each path sees a
    mismatch (a misconfigured issuer otherwise shows up only as 401s). Only
    the issuer is logged, never the token."""
    if where in _warned_issuer:
        return
    _warned_issuer.add(where)
    shown = received if isinstance(received, str) else type(received).__name__
    logging.getLogger("agentic").warning(
        "Token issuer mismatch (%s): expected %r, received %r. If the project "
        "uses a custom domain, set AUTH_ISSUER to the issuer its tokens carry.",
        where, _expected_issuer(), shown[:200])


def _fetch_jwks():
    """GET the project's JWKS. Returns the parsed JSON; raises on any failure."""
    r = _requests.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=5)
    if r.status_code != 200:
        raise _JwksUnavailable(f"JWKS HTTP {r.status_code}")
    return r.json()


def _parse_jwks(data) -> dict:
    """kid -> (key, alg) for every usable ES256/RS256 signing key."""
    keys: dict = {}
    if _pyjwt is None or not isinstance(data, dict):
        return keys
    for jwk in data.get("keys") or []:
        if not isinstance(jwk, dict) or not isinstance(jwk.get("kid"), str):
            continue
        if jwk.get("use", "sig") != "sig":
            continue
        alg = jwk.get("alg")
        if alg is None:
            if jwk.get("kty") == "EC" and jwk.get("crv") == "P-256":
                alg = "ES256"
            elif jwk.get("kty") == "RSA":
                alg = "RS256"
        if alg not in _ASYM_ALGS:
            continue  # HS*/oct keys and anything exotic are never used here
        if (alg == "ES256") != (jwk.get("kty") == "EC" and jwk.get("crv") == "P-256"):
            continue  # ES256 must be an EC P-256 key; RS256 must not be EC
        try:
            key = _pyjwt.PyJWK(jwk, algorithm=alg).key
        except Exception as err:  # noqa: BLE001 - malformed/unsupported key: skip it
            logging.getLogger("agentic").warning("Skipping JWKS key %r: %s", jwk["kid"], err)
            continue
        keys[jwk["kid"]] = (key, alg)
    return keys


def _do_refresh() -> None:
    """Fetch and install the JWKS. Runs on the JWKS worker thread."""
    try:
        keys = _parse_jwks(_fetch_jwks())
    except Exception as err:  # noqa: BLE001 - network, HTTP, JSON: all "unavailable"
        keys = {}
        logging.getLogger("agentic").warning("JWKS fetch failed: %s", err)
    with _jwks_lock:
        if not keys:
            _jwks_state["failed_at"] = _clock()
            raise _JwksUnavailable("no usable keys")
        _jwks_state.update(keys=keys, fetched_at=_clock(), failed_at=None)


def _start_refresh():
    """The running fetch, or a newly started one if the rate limit allows;
    None otherwise. Caller holds _jwks_lock."""
    st = _jwks_state
    fut = st["inflight"]
    if fut is not None and not fut.done():
        return fut
    now = _clock()
    if st["attempt_at"] is not None and now - st["attempt_at"] < _JWKS_REFRESH_MIN:
        return None
    st["attempt_at"] = now
    fut = _jwks_executor.submit(_do_refresh)
    st["inflight"] = fut
    return fut


def _jwks_lookup(kid: str, may_fetch: bool = True):
    """Non-blocking key lookup.

    Returns (key, alg); None for a kid that is unknown even though the key set
    is current (401); or, with may_fetch, a Future of a fetch that might
    supply it. Raises _JwksUnavailable when no usable key set can answer.
    """
    with _jwks_lock:
        st = _jwks_state
        now = _clock()
        age = None if st["fetched_at"] is None else now - st["fetched_at"]
        usable = age is not None and age < _JWKS_MAX_AGE
        if usable and kid in st["keys"]:
            if may_fetch and age is not None and age >= _JWKS_TTL:
                _start_refresh()  # revalidate in the background; serve the cached key now
            return st["keys"][kid]
        if may_fetch:
            fut = _start_refresh()
            if fut is not None:
                return fut
        if not usable or st["failed_at"] is not None:
            raise _JwksUnavailable("no current key set")
        return None


async def _verify_jwks(token: str, header: dict):
    """Verify an ES256/RS256 token against the JWKS. Returns the user dict.

    Raises HTTPException(401) on any token problem and _JwksUnavailable when
    the key set can't be used. Waits (on the loop) only for a fetch that is
    actually needed; a cached key never waits.
    """
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise HTTPException(401, _INVALID)
    found = _jwks_lookup(kid)
    if isinstance(found, concurrent.futures.Future):
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(found)), _JWKS_FETCH_WAIT_S)
        except Exception:  # noqa: BLE001, S110 - failure/timeout: decided by the lookup below
            pass
        found = _jwks_lookup(kid, may_fetch=False)
    if found is None or found[1] != header.get("alg"):
        raise HTTPException(401, _INVALID)
    key, alg = found
    try:
        claims = _pyjwt.decode(
            token,
            key,
            algorithms=[alg],  # from the key, never the header
            audience="authenticated",
            issuer=_expected_issuer(),
            leeway=_JWKS_LEEWAY,
            options={"require": ["exp", "sub", "aud", "iss"]},
        )
    except _pyjwt.InvalidIssuerError:
        # Raised only after the signature verified, so this is a genuine token
        # from the project: most likely AUTH_ISSUER needs setting.
        _warn_issuer_once("jwks", _unverified_claims(token).get("iss"))
        raise HTTPException(401, _INVALID) from None
    except Exception:  # noqa: BLE001 - every decode failure is a bad token
        raise HTTPException(401, _INVALID) from None
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(401, _INVALID)
    return {"id": sub, "email": (claims.get("email") or "").lower()}


def _unverified_header(token: str):
    """The token's JOSE header, or None if it isn't a parsable JWT."""
    if _pyjwt is None:
        return None
    try:
        header = _pyjwt.get_unverified_header(token)
    except Exception:  # noqa: BLE001
        return None
    return header if isinstance(header, dict) else None


def _unverified_claims(token: str) -> dict:
    try:
        claims = _pyjwt.decode(token, options={"verify_signature": False})
    except Exception:  # noqa: BLE001
        return {}
    return claims if isinstance(claims, dict) else {}


def _precheck_for_remote(token: str) -> None:
    """Cheap UNVERIFIED sanity checks before spending an outbound call.

    Nothing here grants access - Supabase still verifies the token - it only
    refuses, for free, tokens Supabase would certainly reject: expired, wrong
    audience, or (when SUPABASE_URL is set) another issuer.
    """
    claims = _unverified_claims(token)
    exp = claims.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)) or exp <= time.time():
        raise HTTPException(401, _INVALID)
    aud = claims.get("aud")
    if not (aud == "authenticated" or (isinstance(aud, list) and "authenticated" in aud)):
        raise HTTPException(401, _INVALID)
    if SUPABASE_URL or AUTH_ISSUER:
        iss = claims.get("iss")
        if iss != _expected_issuer():
            _warn_issuer_once("remote precheck", iss)
            raise HTTPException(401, _INVALID)


def _loop_slots() -> asyncio.Semaphore:
    """The global remote-verification semaphore for the running loop
    (asyncio primitives belong to one loop; production has exactly one)."""
    loop = asyncio.get_running_loop()
    sem = _slots_by_loop.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(AUTH_VERIFY_CONCURRENCY)
        _slots_by_loop[loop] = sem
    return sem


async def _outbound(client: str, fn, *args):
    """Run a blocking verification call off the event loop, admitted FIFO
    within the global and per-client caps. anyio's to_thread is not
    cancellable, so the slot is released only after the thread is done."""
    n = _inflight_by_client.get(client, 0)
    if n >= AUTH_VERIFY_PER_CLIENT:
        raise HTTPException(503, "Too many sign-in checks at once — try again in a moment.")
    _inflight_by_client[client] = n + 1
    try:
        sem = _loop_slots()
        try:
            await asyncio.wait_for(sem.acquire(), AUTH_VERIFY_WAIT_S)
        except asyncio.TimeoutError:
            # asyncio.TimeoutError, not the builtin: they are only the same
            # class from Python 3.11; on 3.10 `except TimeoutError` misses it
            # and the caller got an unhandled error instead of this 503.
            raise HTTPException(
                503, "The auth service is busy — try again in a moment.") from None
        try:
            return await run_in_threadpool(fn, *args)
        finally:
            sem.release()
    finally:
        left = _inflight_by_client.get(client, 1) - 1
        if left > 0:
            _inflight_by_client[client] = left
        else:
            _inflight_by_client.pop(client, None)


async def _verify_via_supabase(token: str, client: str, fallback: bool = False):
    """GET /auth/v1/user for this token (cache, prechecks, admission first)."""
    user = _cached_user(token)
    if user is not None:
        return user
    _precheck_for_remote(token)
    if fallback and not SUPABASE_ANON_KEY:
        # The remote path needs the anon key; without it there is no way to
        # verify, so fail closed (503, not 401: the session may well be fine).
        raise HTTPException(503, "Could not reach the auth service — try again shortly.")
    # _verify_remote makes a blocking HTTP call (up to 8s). One uvicorn worker
    # serves every SSE stream, so it runs off the event loop.
    return await _outbound(client, _verify_remote, token)


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

    # Order documented above _ASYM_ALGS.
    header = _unverified_header(token)
    if header is None:
        raise HTTPException(401, _INVALID)
    alg = header.get("alg")
    client = client_ip(request) if request is not None else "local"
    if SUPABASE_URL and alg in _ASYM_ALGS:
        try:
            user = await _verify_jwks(token, header)
        except _JwksUnavailable:
            user = await _verify_via_supabase(token, client, fallback=True)
    elif alg != "HS256":
        raise HTTPException(401, _INVALID)
    elif SUPABASE_JWT_SECRET:
        user = _verify_local(token)
    else:
        user = await _verify_via_supabase(token, client)

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
