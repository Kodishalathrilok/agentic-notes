"""Iteration 2 reliability: cancellation, capacity, deadline, shutdown.

Only the PROVIDER layer is faked (models._dispatch / models._dispatch_stream);
run_agent, call_model(_stream), the section pool and the endpoints are real.

Why some tests drive the ASGI app by hand: httpx's ASGITransport (0.28)
buffers the WHOLE response before returning it, so it cannot read one SSE
event and then disconnect. The hand-rolled driver below feeds the app the
same receive/send messages a server would, including `http.disconnect`. The
end-to-end disconnect test instead runs a real uvicorn server in a thread and
streams from it with httpx.
"""

import asyncio
import gc
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from sse_starlette.sse import AppStatus

import agent
import auth
import main
import models
import retrieval.semantic as sem
from retriever import page_spans

MODEL = "nvidia/some-model"
_BASELINE_THREADS = {t.ident for t in threading.enumerate()}


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------

def _kind(prompt: str) -> str:
    for marker, kind in (("gatekeeper", "gate"), ("PLANNING agent", "plan"),
                         ("CRITIQUE agent", "critique"), ("GROUNDING agent", "grounding"),
                         ("WRITING agent", "write"), ("REVISION agent", "revise")):
        if marker in prompt:
            return kind
    return "other"


class _Gate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()


class FakeProviders:
    """Canned provider answers; any prompt kind can be held on a gate."""

    def __init__(self, deltas=3, delta_delay=0.0):
        self.calls = []
        self.gates = {}
        self.deltas = deltas
        self.delta_delay = delta_delay
        self.streams_started = 0
        self.streams_closed = 0
        self.slow_stream = None  # 1-based index of a stream that lags after its gate
        self.log = []
        self._lock = threading.Lock()

    def gate(self, kind):
        self.gates[kind] = _Gate()
        return self.gates[kind]

    def release_all(self):
        for g in self.gates.values():
            g.release.set()

    def dispatch(self, prov, prompt, max_tokens, model, temperature, json_mode, strict=False):
        kind = _kind(prompt)
        self.calls.append(kind)
        gate = self.gates.get(kind)
        if gate is not None:
            gate.entered.set()
            assert gate.release.wait(10), "test gate never released"
        self.log.append(f"{kind}_returned")
        if kind == "gate":
            return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
        if kind == "plan":
            return ('{"outline":["A","B","C"],"checklist":["x"],'
                    '"difficulty":"easy","suggested_format":"bullet"}')
        if kind == "critique":
            return ('{"score":9,"needs_revision":false,"unsupported_claims":[],'
                    '"missing_topics":[],"issues":[],"strengths":[]}')
        if kind == "grounding":
            return '{"verdicts":[]}'
        return "Title"

    def dispatch_stream(self, prov, prompt, max_tokens, model, temperature):
        kind = _kind(prompt)
        self.calls.append(kind)
        with self._lock:
            self.streams_started += 1
            index = self.streams_started
        try:
            yield "First words. "
            gate = self.gates.get(kind)
            if gate is not None:
                gate.entered.set()
                assert gate.release.wait(10), "test gate never released"
            if index == self.slow_stream:
                time.sleep(0.5)
            for i in range(self.deltas):
                if self.delta_delay:
                    time.sleep(self.delta_delay)
                yield f"more{i} "
        finally:
            # Runs when the stream is closed early, i.e. its HTTP response
            # would be closed by the real parser's `with`.
            with self._lock:
                self.streams_closed += 1


class _LoggingSlots(main._SlotCounter):
    """Records, at every release, whether the run it belongs to had already
    finished its in-flight step and closed its generator."""

    def __init__(self, limit):
        super().__init__(limit)
        self.releases = []

    def release(self):
        states = [(r.step is None or r.step.done(), r.gen.gi_frame is None)
                  for r in list(main._active_runs) if r.slot is self]
        self.releases.append(states)
        super().release()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_AUTH", False)
    monkeypatch.setattr(auth, "_hits", {})
    monkeypatch.setattr(models, "_nvidia_available", lambda: True)
    monkeypatch.setenv("NVIDIA_MODELS", MODEL)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    fp = FakeProviders()
    monkeypatch.setattr(models, "_dispatch", fp.dispatch)
    monkeypatch.setattr(models, "_dispatch_stream", fp.dispatch_stream)
    slots = _LoggingSlots(2)
    monkeypatch.setattr(main, "_generation_slots", slots)
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cc-work")
    monkeypatch.setattr(main, "EXECUTOR", pool)
    # sse-starlette keeps one module-global anyio.Event, bound to the first
    # event loop that waits on it; every pytest-asyncio test has its own loop.
    monkeypatch.setattr(AppStatus, "should_exit", False)
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    fp.slots = slots
    fp.pool = pool
    yield fp
    fp.release_all()
    deadline = time.monotonic() + 5
    while main._active_runs and time.monotonic() < deadline:
        time.sleep(0.02)
    pool.shutdown(wait=True, cancel_futures=True)


def _wait_sync(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


async def _until(pred, timeout=5.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


class AsgiStream:
    """Drive one request through the ASGI app like a server would, with a
    client that can disconnect at any moment."""

    def __init__(self, path, payload, on_chunk=None):
        self.path = path
        self.on_chunk = on_chunk
        self.body = json.dumps(payload).encode()
        self.status = None
        self.chunks = []
        self.got_chunk = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.task = None

    def events(self):
        out = []
        for chunk in self.chunks:
            for line in chunk.decode().splitlines():
                if line.startswith("data:"):
                    out.append(json.loads(line[5:].strip()))
        return out

    async def _run(self):
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": self.body, "more_body": False}
            await self.disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.start":
                self.status = msg["status"]
            elif msg["type"] == "http.response.body" and msg.get("body"):
                self.chunks.append(msg["body"])
                self.got_chunk.set()
                if self.on_chunk is not None:
                    self.on_chunk(msg["body"])

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": self.path,
            "raw_path": self.path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }
        await main.app(scope, receive, send)

    def start(self):
        self.task = asyncio.create_task(self._run())
        return self

    def disconnect(self):
        self.disconnected.set()


def _body(text="source text about photosynthesis"):
    return {"text": text, "model": MODEL}


def _sse_events(text):
    return [json.loads(line[5:].strip()) for line in text.splitlines()
            if line.startswith("data:")]


# ---------------------------------------------------------------------------
# Disconnect mid-stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_cancels_and_releases_slot_only_after_step_finished(fake):
    gate = fake.gate("gate")
    stream = AsgiStream("/api/generate", _body()).start()

    # First event ("Checking topic") arrives, then the pipeline blocks inside
    # the gatekeeper's provider call.
    assert await _until(lambda: stream.chunks and gate.entered.is_set())
    assert len(main._active_runs) == 1
    run = next(iter(main._active_runs))

    stream.disconnect()
    assert await _until(run.cancel.is_set, 2), "disconnect did not set the cancel flag"
    await asyncio.wait_for(stream.task, 5)

    # The in-flight step is still running: the slot must still be held and
    # the generator not yet closed (closing it now would race the step).
    await asyncio.sleep(0.2)
    assert fake.slots.in_use == 1
    assert not run.step.done()
    assert run.gen.gi_frame is not None

    gate.release.set()
    assert await _until(lambda: fake.slots.in_use == 0), "slot never released"
    assert not main._active_runs
    assert run.gen.gi_frame is None, "pipeline generator was not closed"
    # At the moment the slot was released the step had finished and the
    # generator had been closed.
    assert fake.slots.releases == [[(True, True)]]
    # Nothing after the cancelled stage reached a provider or the client.
    assert fake.calls == ["gate"]
    assert "done" not in [e["type"] for e in stream.events()]


@pytest.mark.asyncio
async def test_slot_is_free_again_after_cancelled_stream(fake):
    fake.slots.limit = 1
    gate = fake.gate("gate")
    stream = AsgiStream("/api/generate", _body()).start()
    assert await _until(gate.entered.is_set)
    stream.disconnect()
    await asyncio.wait_for(stream.task, 5)
    gate.release.set()
    assert await _until(lambda: fake.slots.in_use == 0)

    del fake.gates["gate"]
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/generate", json=_body())
    assert r.status_code == 200
    assert _sse_events(r.text)[-1]["type"] == "done"
    assert await _until(lambda: fake.slots.in_use == 0)


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------

def _long_doc(n=14, chars=1400):
    pages = [f"Section {i}. " + " ".join(f"topic{i} detail{j}" for j in range(chars // 20))
             for i in range(1, n + 1)]
    return "\n\n".join(pages), page_spans(pages)


def test_cancelled_section_writers_start_no_new_windows_and_never_hang(fake, monkeypatch):
    monkeypatch.setattr(agent, "SECTION_CONCURRENCY", 2)
    # The second writer is still busy when the first one stops: closing the
    # pipeline must wait for it rather than leave it running.
    fake.slow_stream = 2
    gate = fake.gate("write")
    text, spans = _long_doc()
    cancel = threading.Event()
    gen = agent.run_agent(text, "exam", "academic", "medium", "bullet", model=MODEL,
                          include_quiz=False, include_flashcards=False,
                          page_spans=spans, cancel=cancel)
    events = []
    first_delta = threading.Event()

    def consume():
        for ev in gen:
            events.append(ev)
            if ev["type"] == "notes_delta":
                first_delta.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    assert first_delta.wait(5)
    assert gate.entered.wait(5)
    assert _wait_sync(lambda: fake.streams_started == 2)
    parts = next(e for e in events if "Covering the whole document" in e["content"])
    n_windows = int(parts["content"].split(" in ")[1].split(" ")[0])
    assert n_windows > 2, "test needs more windows than concurrent writers"

    cancel.set()
    gate.release.set()
    t.join(5)
    assert not t.is_alive(), "pipeline hung on a section queue after cancel"
    assert gen.gi_frame is None

    # Only the writers already running when cancel was set ever started, and
    # both had stopped (provider streams closed) by the time the pipeline did.
    assert fake.calls.count("write") == 2
    assert fake.streams_closed == fake.streams_started == 2
    types = [e["type"] for e in events]
    assert "done" not in types and "error" not in types
    assert not any("could not be completed" in e["content"] for e in events)
    # The pool was shut down with wait=True: no writer thread is left running.
    assert not [th for th in threading.enumerate()
                if th.name.startswith("section") and th.is_alive()]


def test_cancel_before_start_yields_nothing_and_calls_no_provider(fake):
    cancel = threading.Event()
    cancel.set()
    events = list(agent.run_agent("text", "exam", "academic", "medium", "bullet",
                                  model=MODEL, cancel=cancel))
    assert events == []
    assert fake.calls == []


def test_unset_cancel_changes_nothing_on_a_multi_window_run(fake):
    # An unset flag must leave the event stream byte-for-byte what it is with
    # no flag at all - sectioned path, section pool and all.
    text, spans = _long_doc()

    def run(**kw):
        return json.dumps(list(agent.run_agent(
            text, "exam", "academic", "medium", "bullet", model=MODEL,
            include_quiz=False, include_flashcards=False, page_spans=spans, **kw)),
            sort_keys=True, default=str)

    without = run()
    with_flag = run(cancel=threading.Event())
    events = json.loads(without)
    assert sum("Writing section" in e["content"] for e in events) > 2, "not multi-window"
    assert events[-1]["type"] == "done"
    assert with_flag == without


# ---------------------------------------------------------------------------
# call_model_stream / call_model
# ---------------------------------------------------------------------------

def test_pipeline_cancelled_is_not_an_exception():
    # Every broad `except Exception` in the pipeline must let it through.
    assert issubclass(models.PipelineCancelled, BaseException)
    assert not issubclass(models.PipelineCancelled, Exception)


def test_call_model_stream_stops_closes_provider_and_does_not_fail_over(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ("nvidia", "gemini"))
    called, closed = [], []

    def stream(prov, prompt, max_tokens, model, temperature):
        called.append(prov)
        try:
            for i in range(5):
                yield f"d{i} "
        finally:
            closed.append(prov)

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    cancel = threading.Event()
    gen = models.call_model_stream("x", model=MODEL, cancel=cancel)
    assert next(gen) == "d0 "
    cancel.set()
    with pytest.raises(models.PipelineCancelled):
        next(gen)
    assert closed == ["nvidia"], "provider stream (and its HTTP response) not closed"
    assert called == ["nvidia"], "a cancelled call must not fail over"


def test_call_model_stream_cancelled_before_start_calls_nothing(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: True)
    called = []
    monkeypatch.setattr(models, "_dispatch_stream",
                        lambda *a, **k: called.append(1) or iter(["x"]))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(models.PipelineCancelled):
        list(models.call_model_stream("x", model=MODEL, cancel=cancel))
    assert called == []


def test_call_model_honours_ambient_cancel_scope(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: True)
    called = []
    monkeypatch.setattr(models, "_dispatch", lambda *a, **k: called.append(1) or "ok")
    cancel = threading.Event()
    cancel.set()
    with models.cancel_scope(cancel):
        with pytest.raises(models.PipelineCancelled):
            models.call_model("x", model=MODEL)
    assert called == []
    # The scope ends with the block: the same thread is unaffected afterwards.
    assert models.call_model("x", model=MODEL) == "ok"


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_request_gets_503_immediately_and_slot_frees_after_completion(fake):
    fake.slots.limit = 1
    gate = fake.gate("gate")
    first = AsgiStream("/api/generate", _body()).start()
    assert await _until(gate.entered.is_set)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.monotonic()
        r = await client.post("/api/generate", json=_body())
        assert r.status_code == 503
        assert time.monotonic() - t0 < 1.0, "rejection must be immediate, not queued"
        assert "capacity" in r.json()["detail"]

        gate.release.set()
        await asyncio.wait_for(first.task, 10)
        assert first.events()[-1]["type"] == "done"
        assert await _until(lambda: fake.slots.in_use == 0)

        del fake.gates["gate"]
        r = await client.post("/api/generate", json=_body())
    assert r.status_code == 200
    assert _sse_events(r.text)[-1]["type"] == "done"


def test_slot_counter_is_atomic_under_threads():
    slots = main._SlotCounter(3)
    got = []
    barrier = threading.Barrier(20)

    def grab():
        barrier.wait()
        got.append(slots.try_acquire())

    threads = [threading.Thread(target=grab) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got.count(True) == 3 and slots.in_use == 3
    for _ in range(3):
        slots.release()
    with pytest.raises(RuntimeError):
        slots.release()


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deadline_sends_one_safe_error_and_releases_slot(fake, monkeypatch):
    monkeypatch.setattr(main, "GENERATION_DEADLINE_S", 0.3)
    gate = fake.gate("gate")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.monotonic()
        r = await asyncio.wait_for(client.post("/api/generate", json=_body()), 5)
    assert time.monotonic() - t0 < 3
    events = _sse_events(r.text)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and events[-1] is errors[0]
    assert errors[0]["content"] == main.GENERATION_TIMEOUT_MESSAGE
    assert "done" not in [e["type"] for e in events]

    runs = list(main._active_runs)
    assert len(runs) == 1 and runs[0].cancel.is_set()
    assert fake.slots.in_use == 1, "slot released while the step was still running"
    gate.release.set()
    assert await _until(lambda: fake.slots.in_use == 0 and not main._active_runs)
    assert fake.calls == ["gate"]


# ---------------------------------------------------------------------------
# Shutdown (lifespan)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_shutdown_cancels_streams_and_keeps_executor_usable(fake):
    gate = fake.gate("gate")
    stream = AsgiStream("/api/generate", _body()).start()
    assert await _until(gate.entered.is_set)
    run = next(iter(main._active_runs))

    threading.Timer(0.2, gate.release.set).start()
    async with main.app.router.lifespan_context(main.app):
        pass
    assert run.cancel.is_set()
    assert not main._active_runs and fake.slots.in_use == 0
    assert fake.pool.submit(lambda: 42).result(timeout=5) == 42
    await asyncio.wait_for(stream.task, 5)
    assert fake.calls == ["gate"]


@pytest.mark.asyncio
async def test_lifespan_shutdown_is_bounded_when_a_step_will_not_stop(fake, monkeypatch):
    monkeypatch.setattr(main, "SHUTDOWN_GRACE_S", 0.2)
    gate = fake.gate("gate")
    stream = AsgiStream("/api/generate", _body()).start()
    assert await _until(gate.entered.is_set)
    run = next(iter(main._active_runs))

    t0 = time.monotonic()
    async with main.app.router.lifespan_context(main.app):
        pass
    assert time.monotonic() - t0 < 1.5
    assert run.cancel.is_set()
    # The stream itself did not wait for the blocked step.
    await asyncio.wait_for(stream.task, 1)
    assert fake.slots.in_use == 1, "slot released while the step was still running"

    # The straggler still cleans up once its step ends and gives its slot back.
    gate.release.set()
    await asyncio.wait_for(run.begin_cleanup(), 5)
    assert fake.slots.in_use == 0 and not main._active_runs


@pytest.mark.asyncio
async def test_shutdown_signal_ends_open_streams_before_lifespan_shutdown(fake):
    # uvicorn runs the lifespan shutdown only after every connection closed,
    # so an open SSE stream must be stopped by the signal itself.
    import signal

    fake.deltas = 400
    fake.delta_delay = 0.02
    server_handler_calls = []
    original = signal.signal(signal.SIGTERM, lambda s, f: server_handler_calls.append(s))
    try:
        async with main.app.router.lifespan_context(main.app):
            stream = AsgiStream("/api/generate", _body()).start()
            assert await _until(lambda: any(e["type"] == "notes_delta"
                                            for e in stream.events()))
            signal.raise_signal(signal.SIGTERM)
            assert server_handler_calls == [signal.SIGTERM], "server handler not chained"
            await asyncio.wait_for(stream.task, 5)  # stream ended by itself
            events = stream.events()
            assert events[-1] == {"type": "error", "step": "error",
                                  "content": main.SERVER_RESTARTING_MESSAGE, "data": None}
            assert "done" not in [e["type"] for e in events]
            assert await _until(lambda: fake.slots.in_use == 0 and not main._active_runs)
            assert fake.streams_closed == fake.streams_started == 1
        # The lifespan put the server's handler back.
        assert signal.getsignal(signal.SIGTERM) is not original
        signal.raise_signal(signal.SIGTERM)
        assert server_handler_calls == [signal.SIGTERM, signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, original)


@pytest.mark.asyncio
async def test_shutdown_with_blocked_step_ends_stream_at_once_but_holds_slot(fake):
    # A step blocked in a call that cannot be interrupted must not keep the
    # connection open (uvicorn waits for it) - but the slot stays taken until
    # that step really ends.
    gate = fake.gate("gate")
    stream = AsgiStream("/api/generate", _body()).start()
    assert await _until(gate.entered.is_set)
    run = next(iter(main._active_runs))

    t0 = time.monotonic()
    main._stop_streams_for_shutdown()
    await asyncio.wait_for(stream.task, 1)
    assert time.monotonic() - t0 < 0.5
    assert stream.events()[-1]["content"] == main.SERVER_RESTARTING_MESSAGE

    await asyncio.sleep(0.2)
    assert fake.slots.in_use == 1 and not run.step.done()
    gate.release.set()
    assert await _until(lambda: fake.slots.in_use == 0 and not main._active_runs)
    assert fake.slots.releases == [[(True, True)]]
    assert fake.calls == ["gate"]


@pytest.mark.asyncio
async def test_restart_notice_never_follows_done(fake):
    # A shutdown landing in the pacing sleep right after `done` must not add
    # an error to a finished run.
    def on_chunk(body):
        if b'"type": "done"' in body:
            main._stop_streams_for_shutdown()

    stream = AsgiStream("/api/generate", _body(), on_chunk=on_chunk).start()
    await asyncio.wait_for(stream.task, 10)
    types = [e["type"] for e in stream.events()]
    assert types[-1] == "done" and "error" not in types
    assert await _until(lambda: fake.slots.in_use == 0 and not main._active_runs)


@pytest.mark.asyncio
async def test_signal_handler_always_calls_server_handler(fake, monkeypatch):
    import signal

    calls = []
    original = signal.signal(signal.SIGTERM, lambda s, f: calls.append(s))
    try:
        async with main.app.router.lifespan_context(main.app):
            loop = asyncio.get_running_loop()

            def broken(*a, **k):
                raise RuntimeError("Event loop is closed")

            monkeypatch.setattr(loop, "call_soon_threadsafe", broken)
            with pytest.raises(RuntimeError):
                signal.raise_signal(signal.SIGTERM)
            monkeypatch.undo()
            assert calls == [signal.SIGTERM], "server never saw the signal"
    finally:
        signal.signal(signal.SIGTERM, original)


def test_two_sequential_lifespans_keep_the_executor(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "REQUIRE_AUTH", False)
    monkeypatch.setattr(auth, "_hits", {})
    for _ in range(2):
        with TestClient(main.app) as client:
            # /api/eval-report runs on EXECUTOR.
            assert client.get("/api/eval-report").status_code == 200


@pytest.mark.asyncio
async def test_no_never_retrieved_future_errors(fake, caplog):
    caplog.set_level(logging.ERROR, logger="asyncio")

    # A pipeline step that raises: read once by the stream...
    def boom():
        yield "a"
        raise ValueError("pipeline bug")

    run = main._StreamRun("generate", boom(), threading.Event())
    main._active_runs.add(run)
    assert await run.next_item() == "a"
    with pytest.raises(ValueError):
        await run.next_item()
    await run.begin_cleanup()

    # ...a step that is still running when the stream times out and whose
    # outcome is only waited on by cleanup...
    gate = threading.Event()

    def slow_boom():
        gate.wait(5)
        raise ValueError("late bug")
        yield  # pragma: no cover

    run = main._StreamRun("generate", slow_boom(), threading.Event())
    main._active_runs.add(run)
    assert await run.next_item(timeout=0.05) is main._TIMED_OUT
    task = run.begin_cleanup()
    gate.set()
    await task

    # ...and a chat stream the client drops (PipelineCancelled in the step).
    fake.deltas = 200
    fake.delta_delay = 0.02
    stream = AsgiStream("/api/chat", {"notes": "n", "question": "q", "model": MODEL}).start()
    assert await _until(lambda: len(stream.chunks) >= 2)
    stream.disconnect()
    await asyncio.wait_for(stream.task, 5)
    assert await _until(lambda: not main._active_runs)

    del run, task
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0.05)
    assert "never retrieved" not in caplog.text, caplog.text


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_disconnect_closes_provider_stream(fake):
    gate = fake.gate("other")  # the chat prompt has no agent marker
    stream = AsgiStream("/api/chat", {"notes": "n", "question": "q", "model": MODEL}).start()
    assert await _until(lambda: stream.chunks and gate.entered.is_set())
    run = next(iter(main._active_runs))
    stream.disconnect()
    await asyncio.wait_for(stream.task, 5)
    assert run.cancel.is_set()
    gate.release.set()
    assert await _until(lambda: not main._active_runs)
    assert fake.streams_closed == 1
    assert b"".join(stream.chunks) == b"First words. ", "deltas sent after cancel"


@pytest.mark.asyncio
async def test_chat_stopped_by_shutdown_ends_with_cut_off_notice(fake):
    gate = fake.gate("other")
    stream = AsgiStream("/api/chat", {"notes": "n", "question": "q", "model": MODEL}).start()
    assert await _until(lambda: stream.chunks and gate.entered.is_set())
    main._stop_streams_for_shutdown()
    await asyncio.wait_for(stream.task, 1)  # does not wait for the blocked step
    assert b"".join(stream.chunks).decode() == "First words. " + main.CHAT_CUT_OFF_NOTICE
    gate.release.set()
    assert await _until(lambda: not main._active_runs)
    assert fake.streams_closed == 1


# ---------------------------------------------------------------------------
# End to end: real server, real socket, httpx streaming client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_server_client_close_cancels_run(fake):
    import uvicorn

    fake.deltas = 400
    fake.delta_delay = 0.02
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port,
                                           lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert await _until(lambda: server.started, 5)
        seen = []
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            async with client.stream("POST", f"http://127.0.0.1:{port}/api/generate",
                                     json=_body()) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        seen.append(json.loads(line[5:]))
                        if seen[-1]["type"] == "notes_delta":
                            runs = list(main._active_runs)
                            break
        assert len(runs) == 1
        run = runs[0]
        # Client gone: within 5s the run is cancelled, its provider stream
        # closed, and its slot back.
        assert await _until(lambda: run.cancel.is_set() and fake.slots.in_use == 0
                            and not main._active_runs, 5)
        assert fake.streams_closed == fake.streams_started == 1
        assert fake.calls.count("critique") == 0, "pipeline went on after disconnect"
    finally:
        server.should_exit = True
        thread.join(5)


# ---------------------------------------------------------------------------
# Leak check (keep last in this module)
# ---------------------------------------------------------------------------

def test_zz_no_leaked_runs_or_threads():
    assert not main._active_runs
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        stray = [t for t in threading.enumerate()
                 if t.ident not in _BASELINE_THREADS and t.is_alive() and not t.daemon
                 and (t.name.startswith(("section", "cc-work")))]
        if not stray:
            break
        time.sleep(0.05)
    assert not stray, [t.name for t in stray]
