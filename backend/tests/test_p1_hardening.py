"""
Regression tests for the P1 hardening pass: the model-id allowlist, user-safe
error messages (no raw exception text reaches clients), the rate limiter's
bounded memory, and the non-blocking eval report. No network, no API keys.
"""

import logging
import time

import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

import agent
import auth
import main
import models


@pytest.fixture
def client(monkeypatch):
    # Fresh limiter state so these tests never trip each other's 429s.
    monkeypatch.setattr(auth, "_hits", {})
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# Model id allowlist
# ---------------------------------------------------------------------------

EVIL = "evil/unknown-model"

MODEL_ENDPOINTS = [
    ("/api/generate", {"text": "hello"}),
    ("/api/chat", {"notes": "n", "question": "q"}),
    ("/api/quiz", {"notes": "n"}),
    ("/api/flashcards", {"notes": "n"}),
    ("/api/rewrite", {"notes": "n", "direction": "shorter"}),
    ("/api/edit-selection", {"notes": "n", "selection": "n", "instruction": "x"}),
]


@pytest.mark.parametrize("path,body", MODEL_ENDPOINTS)
def test_unknown_model_rejected_everywhere(client, monkeypatch, path, body):
    # Nothing may reach a provider for an id /api/models never offered.
    def boom(*a, **k):
        raise AssertionError("a model call was made for an unlisted id")

    monkeypatch.setattr(models, "_dispatch", boom)
    monkeypatch.setattr(models, "_dispatch_stream", boom)
    r = client.post(path, json={**body, "model": EVIL})
    assert r.status_code == 422
    assert "unknown model" in r.json()["detail"].lower()


@pytest.mark.parametrize("path,body", MODEL_ENDPOINTS)
def test_empty_model_still_means_default(client, path, body):
    """Empty passes the allowlist; the handler's own checks run next."""
    body = {k: ("" if k in ("notes", "text") else v) for k, v in body.items()}
    r = client.post(path, json={**body, "model": ""})
    assert r.status_code == 422
    assert "unknown model" not in str(r.json()["detail"]).lower()


def test_listed_models_are_accepted():
    for m in models.available_models():
        assert models.validate_model_id(m["id"]) == m["id"]
    assert models.validate_model_id(models.default_model()) == models.default_model()
    assert models.validate_model_id("  ") == ""


def test_configured_nvidia_ids_become_selectable(monkeypatch):
    monkeypatch.setattr(models, "_nvidia_available", lambda: True)
    monkeypatch.setenv("NVIDIA_MODELS", "nvidia/extra-model")
    assert models.validate_model_id("nvidia/extra-model") == "nvidia/extra-model"
    with pytest.raises(models.UnknownModelError):
        models.validate_model_id("meta/some-other-catalog-model")


def test_internal_helper_model_is_not_subject_to_allowlist(monkeypatch):
    """helper_model() can name an id the picker doesn't list; internal calls
    must keep working with it."""
    monkeypatch.setattr(models, "HELPER_GEMINI_MODEL", "gemini-internal-helper")
    helper = models.helper_model("gemini-3.6-flash")
    assert helper == "gemini-internal-helper"
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "gemini")
    monkeypatch.setattr(models, "_dispatch", lambda prov, prompt, mt, model, *a, **k: f"ok:{model}")
    assert models.call_model("hi", model=helper) == "ok:gemini-internal-helper"


# ---------------------------------------------------------------------------
# Error leakage: pipeline
# ---------------------------------------------------------------------------

SECRET = "/srv/app/internal/secret_path.py key=SUPERSECRET"


def _pipeline_error(monkeypatch, exc):
    def gate(*a, **k):
        raise exc

    monkeypatch.setattr(agent, "classify_academic", gate)
    events = list(agent.run_agent("some text", "exam", "academic", "medium", "bullet",
                                  include_quiz=False, include_flashcards=False))
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    return errors[0]["content"]


def test_unexpected_pipeline_error_is_generic_with_id(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR, logger="agentic"):
        msg = _pipeline_error(monkeypatch, KeyError(SECRET))
    assert "secret_path" not in msg and "SUPERSECRET" not in msg
    assert "KeyError" not in msg and "Pipeline error" not in msg
    assert "error id:" in msg
    error_id = msg.rsplit("error id:", 1)[1].strip(" )")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert error_id in logged and "secret_path" in logged, "detail kept server-side"


def test_user_facing_error_message_passes_through(monkeypatch):
    msg = _pipeline_error(monkeypatch, models.UserFacingError("Try a shorter source."))
    assert msg == "Try a shorter source."


def test_all_providers_failed_is_user_safe(monkeypatch, caplog):
    exc = models._providers_failed([
        ("nvidia", Exception("Error code: 429 https://integrate.api.nvidia.com/v1?key=SECRETKEY")),
    ])
    assert isinstance(exc, models.UserFacingError)
    with caplog.at_level(logging.ERROR, logger="agentic"):
        msg = _pipeline_error(monkeypatch, exc)
    assert msg == models.PROVIDERS_UNAVAILABLE_MESSAGE
    assert "nvidia" not in msg.lower() and "429" not in msg
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "429" in logged and "SECRETKEY" not in logged


def test_no_provider_configured_is_user_safe(monkeypatch):
    msg = _pipeline_error(monkeypatch, models._providers_failed([]))
    assert msg == models.PROVIDERS_UNAVAILABLE_MESSAGE
    assert "API_KEY" not in msg


# ---------------------------------------------------------------------------
# Error leakage: extract-url / extract-pdf
# ---------------------------------------------------------------------------

def test_extract_url_unexpected_error_hides_exception_text(client, monkeypatch):
    def fetch(url):
        raise ValueError(SECRET)

    monkeypatch.setattr(main, "_fetch_url_safely", fetch)
    r = client.post("/api/extract-url", json={"url": "https://example.com/a"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "secret_path" not in detail and "SUPERSECRET" not in detail
    assert detail.startswith("Could not fetch URL") and "error id:" in detail


def test_extract_url_reports_remote_http_status(client, monkeypatch):
    def fetch(url):
        resp = requests.Response()
        resp.status_code = 404
        raise requests.HTTPError(f"404 Client Error for url: {SECRET}", response=resp)

    monkeypatch.setattr(main, "_fetch_url_safely", fetch)
    r = client.post("/api/extract-url", json={"url": "https://example.com/a"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "HTTP 404" in detail
    assert "secret_path" not in detail


def test_extract_url_connection_error_is_actionable(client, monkeypatch):
    def fetch(url):
        raise requests.ConnectionError(f"HTTPSConnectionPool {SECRET}")

    monkeypatch.setattr(main, "_fetch_url_safely", fetch)
    r = client.post("/api/extract-url", json={"url": "https://example.com/a"})
    assert r.status_code == 422
    assert "connect" in r.json()["detail"] and "secret_path" not in r.json()["detail"]


def test_extract_pdf_parse_error_hides_exception_text(client):
    r = client.post("/api/extract-pdf",
                    files={"file": ("x.pdf", b"this is not a pdf at all", "application/pdf")})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail.startswith("Could not read this PDF") and "error id:" in detail
    # pypdf's own wording ("EOF marker", "Stream has ended", ...) stays in the log.
    assert "EOF" not in detail and "Stream" not in detail


# ---------------------------------------------------------------------------
# Rate limiter memory
# ---------------------------------------------------------------------------

async def _call(dep, ip):
    class _Client:
        host = ip

    class _Req:
        client = _Client()

    return await dep(_Req(), "")


@pytest.mark.asyncio
async def test_limiter_sweeps_expired_identities(monkeypatch):
    monkeypatch.setattr(auth, "_hits", {})
    monkeypatch.setattr(auth, "_HITS_SWEEP_MIN", 50)
    monkeypatch.setattr(auth, "_hits_sweep_at", 50)
    stale = time.time() - auth.DAY_SEC - 10
    for i in range(200):  # identities that will never call again
        auth._hits[("gen", 600, f"anon:10.0.0.{i}")] = [stale]
        auth._hits[("gen", auth.DAY_SEC, f"anon:10.0.0.{i}")] = [stale]
    assert len(auth._hits) == 400

    dep = auth.limiter("gen", 2, 600, daily=5)
    await _call(dep, "9.9.9.9")
    # Only the live identity's two windows survive the sweep.
    assert set(k[2] for k in auth._hits) == {"anon:9.9.9.9"}
    assert len(auth._hits) == 2

    # And its limits are still enforced after the sweep.
    await _call(dep, "9.9.9.9")
    with pytest.raises(HTTPException) as exc:
        await _call(dep, "9.9.9.9")
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_limiter_sweep_is_not_run_on_every_request(monkeypatch):
    """Many ACTIVE identities keep the dict large; the threshold must move up
    so each request doesn't pay for a full sweep."""
    monkeypatch.setattr(auth, "_hits", {})
    monkeypatch.setattr(auth, "_HITS_SWEEP_MIN", 10)
    monkeypatch.setattr(auth, "_hits_sweep_at", 10)
    sweeps = []
    real = auth._sweep_expired_hits
    monkeypatch.setattr(auth, "_sweep_expired_hits", lambda now: (sweeps.append(1), real(now)))
    dep = auth.limiter("act", 5, 600)
    for i in range(200):
        await _call(dep, f"anon-{i}")
    assert len(auth._hits) == 200  # all still live
    assert len(sweeps) <= 6, f"sweeps should be amortised, ran {len(sweeps)}"


# ---------------------------------------------------------------------------
# Eval report
# ---------------------------------------------------------------------------

def test_eval_report_endpoint_still_answers(client):
    r = client.get("/api/eval-report")
    assert r.status_code == 200
    assert "available" in r.json()
