"""Supabase asymmetric signing keys (ES256/RS256) verified via the JWKS.

Real EC P-256 and RSA keys are generated here; the JWKS "endpoint" is
auth._fetch_jwks monkeypatched, and the remote /auth/v1/user endpoint is
auth._requests.get monkeypatched - no network.
"""

import asyncio
import base64
import logging
import threading
import time
import weakref

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import HTTPException

import auth
import main

URL = "https://proj.supabase.co"
ISS = f"{URL}/auth/v1"


def _ec():
    return ec.generate_private_key(ec.SECP256R1())


def _rsa():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


EC_KEY, EC_KEY2, RSA_KEY = _ec(), _ec(), _rsa()
_REAL_FETCH_JWKS = auth._fetch_jwks  # before any fixture replaces it


def test_fetch_jwks_non_200_is_unavailable(monkeypatch):
    monkeypatch.setattr(auth, "SUPABASE_URL", URL)

    class R:
        status_code = 503

    monkeypatch.setattr(auth._requests, "get", lambda url, timeout=None: R())
    with pytest.raises(auth._JwksUnavailable):
        _REAL_FETCH_JWKS()


def _jwk(priv, kid, alg):
    algo = jwt.algorithms.ECAlgorithm if alg == "ES256" else jwt.algorithms.RSAAlgorithm
    d = algo.to_jwk(priv.public_key(), as_dict=True)
    d.update(kid=kid, alg=alg, use="sig")
    return d


def _token(priv, kid="k1", alg="ES256", **over):
    now = int(time.time())
    claims = {"sub": "user-1", "email": "A@B.co", "aud": "authenticated",
              "iss": ISS, "exp": now + 3600, "iat": now, "role": "authenticated"}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, priv, algorithm=alg, headers={"kid": kid})


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Jwks:
    """Fake JWKS endpoint: counts fetches; can fail or change its key set."""

    def __init__(self, keys):
        self.keys = keys
        self.calls = 0
        self.fail = False

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise auth._JwksUnavailable("JWKS HTTP 500")
        return {"keys": list(self.keys)}


class _Remote:
    """Fake GET /auth/v1/user."""

    def __init__(self, status=200):
        self.status = status
        self.calls = 0

    def __call__(self, url, headers=None, timeout=None):
        assert url == f"{URL}/auth/v1/user"
        self.calls += 1
        status = self.status

        class R:
            status_code = status

            @staticmethod
            def json():
                return {"id": "remote-user", "email": "r@x.y"}

        return R()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    monkeypatch.setattr(auth, "SUPABASE_URL", URL)
    monkeypatch.setattr(auth, "SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(auth, "ALLOWED_EMAILS", set())
    monkeypatch.setattr(auth, "_user_cache", {})
    monkeypatch.setattr(auth, "_bad_tokens", {})
    monkeypatch.setattr(auth, "_slots_by_loop", weakref.WeakKeyDictionary())
    monkeypatch.setattr(auth, "_inflight_by_client", {})
    monkeypatch.setattr(auth, "_warned_issuer", set())
    monkeypatch.setattr(auth, "AUTH_ISSUER", "")
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 8)
    monkeypatch.setattr(auth, "AUTH_VERIFY_PER_CLIENT", 2)
    monkeypatch.setattr(auth, "_jwks_state", {
        "keys": {}, "fetched_at": None, "attempt_at": None, "failed_at": None,
        "inflight": None})
    clock = _Clock()
    monkeypatch.setattr(auth, "_clock", clock)
    jwks = _Jwks([_jwk(EC_KEY, "k1", "ES256"), _jwk(RSA_KEY, "r1", "RS256")])
    monkeypatch.setattr(auth, "_fetch_jwks", jwks)
    remote = _Remote()
    monkeypatch.setattr(auth._requests, "get", remote)

    class E:
        pass

    e = E()
    e.clock, e.jwks, e.remote = clock, jwks, remote
    yield e
    _settle()  # never let a background fetch write into the next test's state


def _settle():
    """Wait for a background JWKS revalidation to finish."""
    fut = auth._jwks_state.get("inflight")
    if fut is not None:
        try:
            fut.result(timeout=10)
        except Exception:  # noqa: BLE001
            pass


class _Req:
    def __init__(self, host):
        self.client = type("C", (), {"host": host, "port": 1234})()
        self.headers = None


def _hs(sub="u", **over):
    """A JWT-shaped legacy HS256 token with plausible claims (unverifiable
    here), i.e. one that passes the prechecks and goes to remote verify."""
    claims = {"sub": sub, "aud": "authenticated", "iss": ISS,
              "exp": int(time.time()) + 600}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, "q" * 32, algorithm="HS256")


def _call(token):
    return asyncio.run(auth.require_user(None, f"Bearer {token}"))


def _status(token):
    with pytest.raises(HTTPException) as exc:
        _call(token)
    return exc.value.status_code


def test_valid_es256(env):
    assert _call(_token(EC_KEY)) == {"id": "user-1", "email": "a@b.co"}
    assert env.remote.calls == 0 and env.jwks.calls == 1


def test_valid_rs256(env):
    assert _call(_token(RSA_KEY, kid="r1", alg="RS256"))["id"] == "user-1"
    assert env.remote.calls == 0


def test_jwks_is_cached(env):
    for _ in range(5):
        _call(_token(EC_KEY))
    assert env.jwks.calls == 1
    env.clock.t += auth._JWKS_TTL + 1  # TTL expiry -> background refetch
    _call(_token(EC_KEY))  # served from the stale copy meanwhile
    _settle()
    assert env.jwks.calls == 2
    _call(_token(EC_KEY))
    assert env.jwks.calls == 2


@pytest.mark.parametrize("over", [
    {"exp": int(time.time()) - 120},
    {"aud": "anon"},
    {"iss": "https://evil.supabase.co/auth/v1"},
    {"sub": None},
    {"sub": ""},
])
def test_bad_claims_are_401(env, over):
    assert _status(_token(EC_KEY, **over)) == 401
    assert env.remote.calls == 0  # authoritative: no fallback


def test_tampered_signature_is_401(env):
    h, p, sig = _token(EC_KEY).split(".")
    sig = sig[:-4] + ("AAAA" if sig[-4:] != "AAAA" else "BBBB")
    assert _status(f"{h}.{p}.{sig}") == 401
    # tampered payload with the original signature
    other = _token(EC_KEY, sub="someone-else").split(".")[1]
    assert _status(f"{h}.{other}.{_token(EC_KEY).split('.')[2]}") == 401
    assert env.remote.calls == 0


def test_same_kid_different_key_is_401(env):
    assert _status(_token(EC_KEY2, kid="k1")) == 401
    assert env.remote.calls == 0


def test_alg_confusion_hs256_with_public_key_is_rejected(env):
    _call(_token(EC_KEY))  # warm the JWKS cache
    for priv in (EC_KEY, RSA_KEY):
        pem = priv.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        now = int(time.time())
        claims = {"sub": "attacker", "aud": "authenticated", "iss": ISS, "exp": now + 600}
        # PyJWT refuses to HMAC with a PEM key, so build the token by hand.
        import hashlib
        import hmac
        import json

        def b64(b):
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

        for kid in ("k1", "r1"):
            head = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": kid}).encode())
            body = b64(json.dumps(claims).encode())
            sig = b64(hmac.new(pem, f"{head}.{body}".encode(), hashlib.sha256).digest())
            tok = f"{head}.{body}.{sig}"
            # No secret configured -> legacy remote path: Supabase says no.
            env.remote.status = 401
            assert _status(tok) == 401
    env.remote.status = 200


def test_alg_confusion_with_jwt_secret_set(env, monkeypatch):
    """With SUPABASE_JWT_SECRET set an HS256 token is only ever checked
    against that secret, never against a JWKS public key."""
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "legacy-secret-0123456789abcdef-0123456789")
    pem = EC_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    import hashlib
    import hmac
    import json

    def b64(b):
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    head = b64(json.dumps({"alg": "HS256", "kid": "k1"}).encode())
    body = b64(json.dumps({"sub": "x", "aud": "authenticated", "iss": ISS,
                           "exp": int(time.time()) + 600}).encode())
    sig = b64(hmac.new(pem, f"{head}.{body}".encode(), hashlib.sha256).digest())
    assert _status(f"{head}.{body}.{sig}") == 401
    assert env.jwks.calls == 0


def test_alg_none_and_other_algs_rejected_without_network(env):
    none_tok = jwt.encode({"sub": "x", "aud": "authenticated", "iss": ISS,
                           "exp": int(time.time()) + 600}, None, algorithm="none",
                          headers={"kid": "k1"})
    assert _status(none_tok) == 401
    hs384 = jwt.encode({"sub": "x"}, "k" * 48, algorithm="HS384", headers={"kid": "k1"})
    assert _status(hs384) == 401
    assert env.remote.calls == 0 and env.jwks.calls == 0


def test_unknown_kid_refreshes_once_then_rate_limited(env):
    _call(_token(EC_KEY))
    assert env.jwks.calls == 1
    env.clock.t += auth._JWKS_REFRESH_MIN + 1
    assert _status(_token(EC_KEY2, kid="nope")) == 401
    assert env.jwks.calls == 2  # one refresh
    for i in range(20):
        assert _status(_token(EC_KEY2, kid=f"junk{i}")) == 401
    assert env.jwks.calls == 2  # no further fetches within 60s
    assert env.remote.calls == 0


def test_unknown_kid_on_cold_cache_fetches_once(env):
    assert _status(_token(EC_KEY2, kid="nope")) == 401
    assert _status(_token(EC_KEY2, kid="nope2")) == 401
    assert env.jwks.calls == 1


def test_key_rotation(env):
    _call(_token(EC_KEY))
    env.jwks.keys.append(_jwk(EC_KEY2, "k2", "ES256"))
    env.clock.t += auth._JWKS_REFRESH_MIN + 1
    assert _call(_token(EC_KEY2, kid="k2"))["id"] == "user-1"
    assert env.jwks.calls == 2
    _call(_token(EC_KEY))  # old key still valid until removed
    assert env.jwks.calls == 2


def test_jwks_unreachable_falls_back_to_remote(env):
    env.jwks.fail = True
    tok = _token(EC_KEY)
    assert _call(tok) == {"id": "remote-user", "email": "r@x.y"}
    assert env.remote.calls == 1
    # Within the retry gap the JWKS isn't hammered; remote cache serves repeats.
    _call(tok)
    assert env.jwks.calls == 1 and env.remote.calls == 1
    # After the gap the JWKS is tried again and, once back, is used.
    env.jwks.fail = False
    env.clock.t += auth._JWKS_REFRESH_MIN + 1
    assert _call(_token(EC_KEY))["id"] == "user-1"
    assert env.jwks.calls == 2 and env.remote.calls == 1


def test_empty_jwks_falls_back_to_remote(env):
    env.jwks.keys = []
    assert _call(_token(EC_KEY))["id"] == "remote-user"


def test_jwks_fallback_without_anon_key_fails_closed(env, monkeypatch):
    monkeypatch.setattr(auth, "SUPABASE_ANON_KEY", "")
    env.jwks.fail = True
    assert _status(_token(EC_KEY)) == 503
    assert env.remote.calls == 0


@pytest.mark.parametrize("code,expected", [
    (429, 503), (500, 503), (502, 503), (503, 503), (401, 401), (403, 401)])
def test_remote_status_mapping(env, code, expected):
    env.jwks.fail = True
    env.remote.status = code
    assert _status(_token(EC_KEY)) == expected


def test_remote_5xx_via_legacy_path_is_503(env):
    env.remote.status = 500
    assert _status(_hs("a")) == 503
    env.remote.status = 401
    assert _status(_hs("b")) == 401
    assert env.remote.calls == 2


@pytest.mark.parametrize("token", [
    "not-a-jwt", "junk1", "a.b", "a.b.c", "legit" + "eyJhbGciOiJIUzI1NiJ9.e30.x", "",
    "....", "eyJhbGciOiJIUzI1NiJ9"])
def test_non_jwt_tokens_are_401_without_any_call(env, token):
    if not token:
        return  # empty bearer is "Sign in required" (covered elsewhere)
    assert _status(token) == 401
    assert env.remote.calls == 0 and env.jwks.calls == 0


@pytest.mark.parametrize("over", [
    {"exp": int(time.time()) - 5}, {"exp": None}, {"exp": "9999999999"},
    {"exp": True}, {"aud": "anon"}, {"aud": None},
    {"iss": "https://other.supabase.co/auth/v1"}, {"iss": None}])
def test_hs256_prechecks_refuse_without_remote_call(env, over):
    assert _status(_hs(**over)) == 401
    assert env.remote.calls == 0


def test_hs256_plausible_token_goes_remote(env):
    assert _call(_hs())["id"] == "remote-user"
    assert _call(_hs(aud=["authenticated", "x"]))["id"] == "remote-user"
    assert env.remote.calls == 2


def test_remote_fallback_also_prechecked(env):
    env.jwks.fail = True
    assert _status(_token(EC_KEY, aud="nope")) == 401
    assert env.remote.calls == 0


def test_concurrency_bound(env, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 3)
    monkeypatch.setattr(auth, "AUTH_VERIFY_PER_CLIENT", 100)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def slow_get(url, headers=None, timeout=None):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {"id": "u", "email": "e@x.y"}

        return R()

    monkeypatch.setattr(auth._requests, "get", slow_get)

    async def run():
        return await asyncio.gather(*[
            auth.require_user(None, f"Bearer {_hs(f'u{i}')}") for i in range(20)])

    users = asyncio.run(run())
    assert len(users) == 20
    assert state["max"] <= 3
    assert state["max"] >= 2  # it really did run in parallel


async def _statuses(*coros):
    """Status per call: 200 for a user dict, the code for an HTTPException.
    Anything else is re-raised - it would be a 500 in production, and
    mapping it to 200 once hid a Python-3.10-only admission bug."""
    out = []
    for r in await asyncio.gather(*coros, return_exceptions=True):
        if isinstance(r, HTTPException):
            out.append(r.status_code)
        elif isinstance(r, BaseException):
            raise r
        else:
            assert isinstance(r, dict) and "id" in r
            out.append(200)
    return out


class _Gate:
    """Fake /auth/v1/user whose calls block until released, so tests can
    wait for "provably inside the remote call" instead of sleeping."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.release = threading.Event()
        self._lock = threading.Lock()
        monkeypatch.setattr(auth._requests, "get", self._get)

    def _get(self, url, headers=None, timeout=None):
        with self._lock:
            self.calls.append(headers["Authorization"])
        assert self.release.wait(10), "gate never released"

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {"id": "u", "email": "e@x.y"}

        return R()


async def _until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.002)


def _waiters(sem) -> int:
    return len(getattr(sem, "_waiters", None) or ())


def test_concurrency_excess_waits_then_503(env, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 1)
    monkeypatch.setattr(auth, "AUTH_VERIFY_WAIT_S", 0.1)
    gate = _Gate(monkeypatch)

    async def run():
        first = asyncio.create_task(auth.require_user(_Req("10.0.0.1"), f"Bearer {_hs('a')}"))
        await _until(lambda: len(gate.calls) == 1)  # first holds the only slot
        t0 = time.perf_counter()
        second = await _statuses(auth.require_user(_Req("10.0.0.2"), f"Bearer {_hs('b')}"))
        waited = time.perf_counter() - t0
        gate.release.set()
        return second, waited, await first

    try:
        second, waited, first = asyncio.run(run())
    finally:
        gate.release.set()
    assert second == [503] and first["id"] == "u"
    assert auth.AUTH_VERIFY_WAIT_S <= waited < auth.AUTH_VERIFY_WAIT_S + 2.0
    assert len(gate.calls) == 1  # the rejected caller never reached Supabase


def test_admission_timeout_is_a_503_http_error(env, monkeypatch):
    """The admission wait timing out must surface as HTTPException(503) on
    every supported Python (asyncio.TimeoutError is not the builtin
    TimeoutError before 3.11)."""
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 1)
    monkeypatch.setattr(auth, "AUTH_VERIFY_WAIT_S", 0.05)

    async def run():
        sem = auth._loop_slots()
        await sem.acquire()  # the only slot is taken
        try:
            with pytest.raises(HTTPException) as exc:
                await auth._outbound("c", lambda: pytest.fail("must not run"))
            return exc.value.status_code
        finally:
            sem.release()

    assert asyncio.run(run()) == 503
    assert auth._inflight_by_client == {}


def test_per_client_cap(env, monkeypatch):
    gate = _Gate(monkeypatch)

    async def run():
        flood = [asyncio.create_task(
            auth.require_user(_Req("203.0.113.9"), f"Bearer {_hs(f'x{i}')}")) for i in range(6)]
        other = asyncio.create_task(
            auth.require_user(_Req("198.51.100.1"), f"Bearer {_hs('other')}"))
        # Two flood calls and the other client's call are inside Supabase;
        # the remaining four flood calls were refused without waiting.
        await _until(lambda: len(gate.calls) == 3 and sum(t.done() for t in flood) == 4)
        gate.release.set()
        return await _statuses(*flood, other)

    try:
        st = asyncio.run(run())
    finally:
        gate.release.set()
    assert st[:6].count(200) == 2 and st[:6].count(503) == 4  # flooder capped at 2
    assert st[6] == 200  # another client is unaffected
    assert len(gate.calls) == 3
    assert auth._inflight_by_client == {}


def test_admission_is_fifo(env, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 1)
    monkeypatch.setattr(auth, "AUTH_VERIFY_PER_CLIENT", 100)
    gate = _Gate(monkeypatch)
    toks = [_hs(f"f{i}") for i in range(6)]

    async def run():
        tasks = [asyncio.create_task(auth.require_user(_Req("10.1.0.0"), f"Bearer {toks[0]}"))]
        await _until(lambda: len(gate.calls) == 1)
        sem = auth._loop_slots()
        for i, t in enumerate(toks[1:], start=1):
            tasks.append(asyncio.create_task(auth.require_user(_Req(f"10.1.0.{i}"), f"Bearer {t}")))
            await _until(lambda i=i: _waiters(sem) == i)  # queued, in this order
        gate.release.set()
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run())
    finally:
        gate.release.set()
    assert gate.calls == [f"Bearer {t}" for t in toks]


def test_jwks_fetch_does_not_wait_behind_remote_slots(env, monkeypatch):
    """Every remote slot busy with junk (held until the end); a cold-cache
    ES256 login still verifies, because the JWKS fetch has its own worker.
    Remote calls are released only AFTER the login completed, so it cannot
    have been served by a freed slot."""
    monkeypatch.setattr(auth, "AUTH_VERIFY_CONCURRENCY", 1)
    monkeypatch.setattr(auth, "AUTH_VERIFY_PER_CLIENT", 100)
    monkeypatch.setattr(auth, "AUTH_VERIFY_WAIT_S", 30)
    gate = _Gate(monkeypatch)

    async def run():
        junk = [asyncio.create_task(auth.require_user(_Req("10.9.9.9"), f"Bearer {_hs(f'j{i}')}"))
                for i in range(3)]
        sem = auth._loop_slots()
        await _until(lambda: len(gate.calls) == 1 and _waiters(sem) == 2)
        user = await asyncio.wait_for(
            auth.require_user(_Req("10.0.0.5"), f"Bearer {_token(EC_KEY)}"), 5)
        still_blocked = not gate.release.is_set() and len(gate.calls) == 1
        gate.release.set()
        await asyncio.gather(*junk, return_exceptions=True)
        return user, still_blocked

    try:
        user, still_blocked = asyncio.run(run())
    finally:
        gate.release.set()
    assert user["id"] == "user-1" and still_blocked
    assert env.jwks.calls == 1


def test_stale_keys_served_while_refresh_fails(env):
    _call(_token(EC_KEY))
    env.jwks.fail = True
    env.clock.t += auth._JWKS_TTL + 1
    assert _call(_token(EC_KEY))["id"] == "user-1"  # stale-while-revalidate
    _settle()
    assert env.jwks.calls == 2 and env.remote.calls == 0
    env.clock.t += 3600
    assert _call(_token(EC_KEY))["id"] == "user-1"  # still within the hard max age
    _settle()
    # Past the hard max age the stale keys are dropped: remote decides.
    env.clock.t += auth._JWKS_MAX_AGE
    assert _call(_token(EC_KEY))["id"] == "remote-user"
    _settle()


def test_auth_issuer_override_and_mismatch_logged_once(env, monkeypatch, caplog):
    custom = "https://auth.example.com/auth/v1"
    tok = _token(EC_KEY, iss=custom)
    with caplog.at_level(logging.WARNING, logger="agentic"):
        assert _status(tok) == 401
        assert _status(_token(EC_KEY, iss=custom)) == 401
    msgs = [r.getMessage() for r in caplog.records if "issuer mismatch" in r.getMessage()]
    assert len(msgs) == 1
    assert ISS in msgs[0] and custom in msgs[0]
    assert tok not in msgs[0] and tok.split(".")[2] not in msgs[0]
    monkeypatch.setattr(auth, "AUTH_ISSUER", custom)
    assert _call(tok)["id"] == "user-1"
    assert _status(_token(EC_KEY)) == 401  # default issuer no longer accepted
    # The remote precheck uses the override too.
    assert _call(_hs(iss=custom))["id"] == "remote-user"


def test_secret_without_url_warns_at_startup(monkeypatch, caplog):
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "s" * 32)
    monkeypatch.setattr(auth, "SUPABASE_URL", "")
    with caplog.at_level(logging.WARNING, logger="agentic"):
        auth.verify_auth_config()
    assert any("ES256/RS256" in r.getMessage() for r in caplog.records)
    caplog.clear()
    monkeypatch.setattr(auth, "SUPABASE_URL", URL)
    with caplog.at_level(logging.WARNING, logger="agentic"):
        auth.verify_auth_config()
    assert not any("ES256/RS256" in r.getMessage() for r in caplog.records)


def test_legacy_hs256_path_unchanged(env, monkeypatch):
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "legacy-secret-0123456789abcdef-0123456789")
    tok = jwt.encode({"sub": "u9", "email": "Z@z.z", "aud": "authenticated",
                      "exp": int(time.time()) + 600}, "legacy-secret-0123456789abcdef-0123456789", algorithm="HS256")
    assert _call(tok) == {"id": "u9", "email": "z@z.z"}
    assert env.jwks.calls == 0 and env.remote.calls == 0
    # ES256 tokens still verify via JWKS even with the legacy secret set.
    assert _call(_token(EC_KEY))["id"] == "user-1"


def test_hs256_without_secret_goes_remote(env):
    assert _call(_hs("u9"))["id"] == "remote-user"
    assert env.jwks.calls == 0


def test_real_fetch_jwks_parses_response(env, monkeypatch):
    seen = []

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"keys": [_jwk(EC_KEY, "k1", "ES256"),
                             {"kty": "oct", "k": "c2VjcmV0", "kid": "h1", "alg": "HS256"},
                             {"kty": "EC", "kid": "bad", "alg": "ES256"}]}

    def fake_get(url, timeout=None, **kw):
        seen.append((url, timeout))
        return R()

    monkeypatch.setattr(auth._requests, "get", fake_get)
    keys = auth._parse_jwks(_REAL_FETCH_JWKS())
    assert list(keys) == ["k1"]  # HS256/oct and malformed keys are dropped
    assert seen == [(f"{URL}/auth/v1/.well-known/jwks.json", 5)]


# --- event loop ------------------------------------------------------------

@pytest.mark.asyncio
async def test_jwks_fetch_does_not_block_loop(env, monkeypatch):
    started = {}

    def slow_fetch():
        started["t"] = time.perf_counter()
        time.sleep(1.0)
        return {"keys": [_jwk(EC_KEY, "k1", "ES256")]}

    monkeypatch.setattr(auth, "_fetch_jwks", slow_fetch)
    tok = _token(EC_KEY)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow = asyncio.create_task(client.post(
            "/api/export/markdown", json={"notes": "# Hi"},
            headers={"Authorization": f"Bearer {tok}"}))
        deadline = time.perf_counter() + 5
        while "t" not in started:
            assert not slow.done() and time.perf_counter() < deadline
            await asyncio.sleep(0.005)
        health = await client.get("/api/health")
        latency = time.perf_counter() - started["t"]
        assert health.status_code == 200
        resp = await slow
    assert resp.status_code == 200 and "Hi" in resp.text
    assert latency < 0.35


def test_header_alg_must_match_the_keys_alg(env):
    # RS256 token naming the EC key's kid (and vice versa) is refused.
    assert _status(_token(RSA_KEY, kid="k1", alg="RS256")) == 401
    assert _status(_token(EC_KEY, kid="r1", alg="ES256")) == 401
    assert env.remote.calls == 0
