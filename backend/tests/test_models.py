"""Router tests: rate-limit retry and error surfacing (no network)."""

import models


def test_failed_providers_message_puts_real_cause_first():
    """A dead-Ollama fallback must not hide the real Groq/Gemini error."""
    errors = [
        ("groq", Exception("Error code: 429 - rate limit reached")),
        ("ollama", Exception("Connection refused (localhost:11434)")),
    ]
    msg = str(models._providers_failed(errors))
    # Groq's real reason appears, and before the ollama noise.
    assert "groq" in msg and "429" in msg
    assert msg.index("groq") < msg.index("ollama")


def test_failed_providers_message_when_nothing_configured():
    msg = str(models._providers_failed([]))
    assert "GROQ_API_KEY" in msg


def test_is_rate_limit_detection():
    assert models._is_rate_limit(Exception("Error code: 429"))
    assert models._is_rate_limit(Exception("Rate limit reached for model"))
    assert not models._is_rate_limit(Exception("401 invalid api key"))


def test_groq_create_retries_then_succeeds(monkeypatch):
    """A transient 429 is retried (with backoff) instead of failing the run."""
    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr(models.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(models, "GROQ_RATE_LIMIT_RETRIES", 3)

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise Exception("Error code: 429 - rate limit reached")
                    return "ok"

    out = models._groq_create(Client(), model="x")
    assert out == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # waited before each retry


def test_groq_create_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setattr(models, "GROQ_RATE_LIMIT_RETRIES", 2)

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise Exception("Error code: 429")

    try:
        models._groq_create(Client(), model="x")
        assert False, "should have raised"
    except Exception as exc:  # noqa: BLE001
        assert "429" in str(exc)


def test_non_rate_limit_error_not_retried(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls["n"] += 1
                    raise Exception("401 invalid api key")

    try:
        models._groq_create(Client(), model="x")
        assert False
    except Exception:  # noqa: BLE001
        pass
    assert calls["n"] == 1  # auth errors fail fast, no retry
