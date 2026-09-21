"""The app runs as ONE uvicorn worker serving long-lived SSE streams, so a
blocking call made directly on the event loop stalls every other request and
every open stream for its duration.

These tests drive the app through httpx.ASGITransport, which runs it on the
test's own event loop (TestClient would run it on a separate thread and hide
the problem). A deliberately slow blocking call is started, and /api/health
must still answer promptly while it is in flight.
"""

import asyncio
import time

import httpx
import pytest

import auth
import main

SLOW_SEC = 1.0
MAX_HEALTH_SEC = 0.35


@pytest.fixture(autouse=True)
def _clean_rate_limits(monkeypatch):
    # Auth off by default so the backend .env can't change behaviour; the
    # remote-auth test turns it back on explicitly.
    monkeypatch.setattr(auth, "REQUIRE_AUTH", False)
    auth._hits.clear()
    yield
    auth._hits.clear()


class _Probe:
    """A blocking stand-in that records when it started."""

    def __init__(self):
        self.started_at = None

    def block(self):
        self.started_at = time.perf_counter()
        time.sleep(SLOW_SEC)


async def _health_latency_during(probe: _Probe, send_slow):
    """Start the slow request, then time /api/health while it is in flight.

    Returns (slow_response, seconds from the slow call starting until health
    answered). If the loop is blocked, the poll below can't even resume until
    the blocking call has finished, so the figure comes out >= SLOW_SEC.
    """
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow = asyncio.create_task(send_slow(client))
        deadline = time.perf_counter() + 5
        while probe.started_at is None:
            if slow.done() or time.perf_counter() > deadline:
                r = await slow
                pytest.fail(f"slow call never started: {r.status_code} {r.text}")
            await asyncio.sleep(0.005)
        health = await client.get("/api/health")
        latency = time.perf_counter() - probe.started_at
        assert health.status_code == 200
        slow_resp = await slow
    return slow_resp, latency


@pytest.mark.asyncio
async def test_remote_auth_does_not_block_loop(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(auth, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(auth, "SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setattr(auth, "ALLOWED_EMAILS", set())
    monkeypatch.setattr(auth, "_user_cache", {})

    probe = _Probe()

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"id": "u1", "email": "a@b.c"}

    def fake_get(*args, **kwargs):
        probe.block()
        return _Resp()

    monkeypatch.setattr(auth._requests, "get", fake_get)

    resp, latency = await _health_latency_during(
        probe,
        lambda c: c.post(
            "/api/export/markdown",
            json={"notes": "# Hi"},
            headers={"Authorization": "Bearer tok-123"},
        ),
    )
    print(f"\n[remote auth] /api/health latency: {latency:.3f}s")
    assert resp.status_code == 200
    assert "Hi" in resp.text
    assert latency < MAX_HEALTH_SEC


@pytest.mark.asyncio
async def test_youtube_transcript_does_not_block_loop(monkeypatch):
    probe = _Probe()

    def fake_transcript(video_id):
        probe.block()
        return "word " * 30

    monkeypatch.setattr(main, "_youtube_transcript", fake_transcript)

    resp, latency = await _health_latency_during(
        probe,
        lambda c: c.post(
            "/api/extract-url",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        ),
    )
    print(f"\n[youtube] /api/health latency: {latency:.3f}s")
    assert resp.status_code == 200
    assert resp.json()["title"] == "YouTube transcript"
    assert latency < MAX_HEALTH_SEC


@pytest.mark.asyncio
async def test_youtube_failure_mapping_survives_executor(monkeypatch):
    class TranscriptsDisabled(Exception):
        pass

    def fake_transcript(video_id):
        raise TranscriptsDisabled("nope")

    monkeypatch.setattr(main, "_youtube_transcript", fake_transcript)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/extract-url",
            json={"url": "https://youtu.be/dQw4w9WgXcQ"},
        )
    assert r.status_code == 422
    assert "captions turned off" in r.json()["detail"]


@pytest.mark.asyncio
async def test_pdf_export_does_not_block_loop(monkeypatch):
    probe = _Probe()

    def fake_pdf(notes, quiz, flashcards):
        probe.block()
        return b"%PDF-fake"

    monkeypatch.setattr(main, "notes_to_pdf", fake_pdf)

    resp, latency = await _health_latency_during(
        probe,
        lambda c: c.post("/api/export/pdf", json={"notes": "# Hi"}),
    )
    print(f"\n[pdf export] /api/health latency: {latency:.3f}s")
    assert resp.status_code == 200
    assert resp.content == b"%PDF-fake"
    assert resp.headers["content-disposition"] == "attachment; filename=notes.pdf"
    assert latency < MAX_HEALTH_SEC
