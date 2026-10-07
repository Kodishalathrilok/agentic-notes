"""A fallback that is known to be down must not be paid for again and again.

Measured on a 55-page document: ten window calls got nothing from NVIDIA. Each
then waited 2 + 4 + 8 seconds on a Gemini that was answering 429 / 503 to every
request, and then tried a local Ollama whose model was not installed - about 14
seconds of waiting per call for an answer that was known in advance.

NVIDIA already cools down after a 503. Gemini did not, and Ollama counted as
available whenever the daemon answered at all.
"""
import time

import pytest
import requests

import models

NVIDIA = "nvidia/some-model"


class Response:
    def __init__(self, status, payload=None):
        self.status_code, self.headers, self._payload = status, {}, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(())


@pytest.fixture
def world(monkeypatch):
    """NVIDIA fails before any output; Gemini goes through its real request
    path and answers with whatever status the test queues (default 429)."""
    state = {"gemini_posts": 0, "nvidia_calls": 0, "sleeps": [], "status": 429}

    def post(**kwargs):
        state["gemini_posts"] += 1
        return Response(state["status"], {"candidates": [
            {"content": {"parts": [{"text": "gemini answer"}]}, "finishReason": "STOP"}]})

    def nvidia(*a, **k):
        state["nvidia_calls"] += 1
        raise RuntimeError("stream closed without producing any output")

    def nvidia_stream(*a, **k):
        state["nvidia_calls"] += 1
        raise RuntimeError("stream closed without producing any output")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(models.requests, "post", post)
    monkeypatch.setattr(models.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(models, "_call_nvidia", nvidia)
    monkeypatch.setattr(models, "_stream_nvidia", nvidia_stream)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ("nvidia", "gemini"))
    monkeypatch.setattr(models, "_nvidia_cooldown", {})
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 3)
    return state


def _fails(call):
    with pytest.raises(models.ProvidersUnavailableError):
        call()


def test_gemini_out_of_retries_is_not_waited_on_again(world):
    _fails(lambda: models.call_model("p", model=NVIDIA))
    assert world["gemini_posts"] == 4, "the first failure still gets every retry"
    assert world["sleeps"] == [2.0, 4.0, 8.0]

    _fails(lambda: models.call_model("p", model=NVIDIA))
    assert world["nvidia_calls"] == 2, "the chosen provider is still tried"
    assert world["gemini_posts"] == 4, "a Gemini known to be limited was asked again"
    assert world["sleeps"] == [2.0, 4.0, 8.0], "another 14 seconds were spent waiting"


def test_streams_skip_a_limited_gemini_too(world):
    _fails(lambda: list(models.call_model_stream("p", model=NVIDIA)))
    posts = world["gemini_posts"]
    _fails(lambda: list(models.call_model_stream("p", model=NVIDIA)))
    assert world["gemini_posts"] == posts


def test_a_503_from_gemini_cools_it_down_as_well(world):
    world["status"] = 503
    _fails(lambda: models.call_model("p", model=NVIDIA))
    _fails(lambda: models.call_model("p", model=NVIDIA))
    assert world["gemini_posts"] == 4


def test_gemini_is_tried_again_once_the_cooldown_is_over(world, monkeypatch):
    _fails(lambda: models.call_model("p", model=NVIDIA))
    world["status"] = 200
    real = time.monotonic
    monkeypatch.setattr(models.time, "monotonic", lambda: real() + 61)
    assert models.call_model("p", model=NVIDIA) == "gemini answer"


def test_a_gemini_answer_ends_its_cooldown(world):
    models._start_cooldown("gemini:gemini-3.5-flash-lite")
    world["status"] = 200
    monkeypatch_ready = lambda p: p == "gemini"  # noqa: E731 - the only provider
    models._provider_ready, saved = monkeypatch_ready, models._provider_ready
    try:
        assert models.call_model("p", model="gemini-3.5-flash-lite") == "gemini answer"
    finally:
        models._provider_ready = saved
    assert not models._cooling_down("gemini:gemini-3.5-flash-lite")


def test_the_cooldown_is_per_gemini_model(world):
    """The helper model has its own quota; one limited model says nothing about it."""
    _fails(lambda: models.call_model("p", model=NVIDIA))  # cools the default Gemini model
    world["status"] = 200
    posts = world["gemini_posts"]
    assert models.call_model("p", model="gemini-3.5-flash-lite") == "gemini answer"
    assert world["gemini_posts"] == posts + 1


def test_a_cooling_gemini_is_still_tried_when_nothing_else_can_answer(world, monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "gemini")
    model = "gemini-3.6-flash"
    models._start_cooldown(f"gemini:{model}")
    world["status"] = 200
    assert models.call_model("p", model=model) == "gemini answer"


def test_with_every_provider_cooling_the_chosen_one_still_gets_the_call(world):
    """Two cooldowns must never add up to refusing a call without one attempt."""
    models._start_cooldown(NVIDIA)
    models._start_cooldown(f"gemini:{models._gemini_model(NVIDIA)}")
    _fails(lambda: models.call_model("p", model=NVIDIA))
    assert world["nvidia_calls"] == 1
    assert world["gemini_posts"] == 0


# ---------------------------------------------------------------------------
# Ollama: a daemon without the model is not a provider
# ---------------------------------------------------------------------------

@pytest.fixture
def ollama(monkeypatch):
    def serve(names):
        monkeypatch.setattr(models, "_ollama_reachable", None)
        monkeypatch.setattr(models.requests, "get", lambda *a, **k: Response(
            200, {"models": [{"name": n} for n in names]}))
    monkeypatch.setattr(models, "OLLAMA_MODEL", "qwen2.5:3b")
    return serve


def test_ollama_without_the_configured_model_is_not_available(ollama):
    ollama(["llama3:latest", "mistral:7b"])
    assert models._ollama_available() is False


def test_ollama_with_the_configured_model_is_available(ollama):
    ollama(["llama3:latest", "qwen2.5:3b"])
    assert models._ollama_available() is True


def test_an_untagged_model_name_matches_its_latest_tag(ollama, monkeypatch):
    monkeypatch.setattr(models, "OLLAMA_MODEL", "llama3")
    ollama(["llama3:latest"])
    assert models._ollama_available() is True


def test_an_unreachable_ollama_is_still_not_available(monkeypatch):
    def refuse(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(models, "_ollama_reachable", None)
    monkeypatch.setattr(models.requests, "get", refuse)
    assert models._ollama_available() is False
