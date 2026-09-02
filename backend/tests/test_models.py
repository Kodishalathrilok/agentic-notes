"""Router tests: model routing, rate-limit retry and error surfacing (no network)."""

import models


class FakeResponse:
    """Minimal stand-in for requests.Response — status, headers, close()."""

    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def close(self):
        self.closed = True


def _fake_post(statuses, headers=None, calls=None):
    """Return a requests.post replacement that yields `statuses` in order."""
    seq = list(statuses)

    def post(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return FakeResponse(seq.pop(0) if seq else 200, headers)

    return post


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------

def test_failed_providers_message_puts_real_cause_first():
    """A dead-Ollama fallback must not hide the real NVIDIA/Gemini error."""
    errors = [
        ("gemini", Exception("Error code: 429 - rate limit reached")),
        ("ollama", Exception("Connection refused (localhost:11434)")),
    ]
    msg = str(models._providers_failed(errors))
    # Gemini's real reason appears, and before the ollama noise.
    assert "gemini" in msg and "429" in msg
    assert msg.index("gemini") < msg.index("ollama")


def test_failed_providers_message_when_nothing_configured():
    msg = str(models._providers_failed([]))
    assert "NVIDIA_API_KEY" in msg and "GEMINI_API_KEY" in msg


def test_error_text_strips_api_key():
    leaked = Exception("404 for url: https://example.com/v1/models/x?key=SECRET123")
    assert "SECRET123" not in models._safe(leaked)


# ---------------------------------------------------------------------------
# Rate-limit / overload retry
# ---------------------------------------------------------------------------

def test_post_retrying_retries_then_succeeds(monkeypatch):
    """A transient 429 is retried (with backoff) instead of failing the run."""
    sleeps = []
    monkeypatch.setattr(models.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 3)
    monkeypatch.setattr(models.requests, "post", _fake_post([429, 429, 200]))

    resp = models._post_retrying("test", url="http://x")
    assert resp.status_code == 200
    assert len(sleeps) == 2  # waited before each retry


def test_post_retrying_retries_503_overloaded(monkeypatch):
    """Gemini answers "high demand" with 503, which is just as transient."""
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 2)
    monkeypatch.setattr(models.requests, "post", _fake_post([503, 200]))

    assert models._post_retrying("test", url="http://x").status_code == 200


def test_post_retrying_gives_up_and_returns_last_response(monkeypatch):
    """Exhausted retries hand the failing response back so raise_for_status()
    turns it into a normal provider error and failover takes over."""
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 2)
    monkeypatch.setattr(models.requests, "post", _fake_post([429, 429, 429]))

    assert models._post_retrying("test", url="http://x").status_code == 429


def test_post_retrying_does_not_retry_auth_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setattr(models.requests, "post", _fake_post([401], calls=calls))

    assert models._post_retrying("test", url="http://x").status_code == 401
    assert len(calls) == 1  # auth errors fail fast, no retry


def test_retry_wait_honours_retry_after_header():
    assert models._retry_wait(FakeResponse(429, {"retry-after": "4"}), 0) == 4.5
    # Junk header falls back to backoff instead of blowing up.
    assert models._retry_wait(FakeResponse(429, {"retry-after": "soon"}), 0) == 2.0
    # No header: exponential backoff, capped.
    assert models._retry_wait(FakeResponse(429), 2) == 8.0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_provider_routing_by_id_shape():
    assert models._provider_for("gemini-3.6-flash") == "gemini"
    assert models._provider_for("gemini-flash-latest") == "gemini"
    assert models._provider_for("nvidia/nemotron-3-ultra-550b-a55b") == "nvidia"
    assert models._provider_for("meta/llama-3.3-70b-instruct") == "nvidia"
    # Ollama tags carry a ":" and no namespace.
    assert models._provider_for("qwen2.5:3b") == "ollama"
    assert models._provider_for(models.OLLAMA_MODEL) == "ollama"


def test_helper_model_matches_provider_family():
    # Gemini main -> Gemini helper.
    assert models.helper_model(models.DEFAULT_GEMINI_MODEL) == models.HELPER_GEMINI_MODEL
    # A cheap helper must not equal the strong writing model.
    assert models.HELPER_GEMINI_MODEL != models.DEFAULT_GEMINI_MODEL


def test_helper_model_falls_back_to_main_nvidia_model(monkeypatch):
    """With no cheap NVIDIA id configured the helper reuses the main model
    rather than crossing into another provider's quota."""
    nid = "nvidia/nemotron-3-ultra-550b-a55b"
    monkeypatch.setattr(models, "HELPER_NVIDIA_MODEL", "")
    assert models.helper_model(nid) == nid
    monkeypatch.setattr(models, "HELPER_NVIDIA_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    assert models.helper_model(nid) == "nvidia/nemotron-3-nano-30b-a3b"


def test_failover_chain_covers_every_provider():
    chain = models._failover_chain("gemini")
    assert chain[0] == "gemini"
    assert set(chain) == {"nvidia", "gemini", "ollama"}
    assert len(chain) == 3


def test_gemini_body_adds_reasoning_headroom():
    """Thought tokens count against maxOutputTokens, so a 30-token title
    budget must not be sent to Gemini as literally 30."""
    body = models._gemini_body("hi", 30, 0.1, False)
    assert body["generationConfig"]["maxOutputTokens"] == 30 + models.GEMINI_REASONING_HEADROOM


# ---------------------------------------------------------------------------
# Gemini: retired-id self-heal
# ---------------------------------------------------------------------------

def test_gemini_alias_matches_model_family():
    assert models._gemini_alias_for("gemini-3.6-flash") == models._GEMINI_FLASH_ALIAS
    assert models._gemini_alias_for("gemini-3.5-flash-lite") == models._GEMINI_LITE_ALIAS


def test_gemini_post_retries_alias_when_pinned_id_is_retired(monkeypatch):
    """A retired pin (404) must self-heal onto the moving alias, not fail the run."""
    urls = []

    def post(**kwargs):
        urls.append(kwargs["url"])
        # First id is gone; the alias answers.
        return FakeResponse(404 if len(urls) == 1 else 200)

    monkeypatch.setattr(models.requests, "post", post)
    resp = models._gemini_post("gemini-3.6-flash", {}, "test", 30)

    assert resp.status_code == 200
    assert len(urls) == 2
    assert "gemini-3.6-flash" in urls[0]
    assert models._GEMINI_FLASH_ALIAS in urls[1]


def test_gemini_post_does_not_loop_when_the_alias_itself_is_the_model(monkeypatch):
    """Asking for the alias and getting a 404 must not retry the same id."""
    urls = []

    def post(**kwargs):
        urls.append(kwargs["url"])
        return FakeResponse(404)

    monkeypatch.setattr(models.requests, "post", post)
    resp = models._gemini_post(models._GEMINI_FLASH_ALIAS, {}, "test", 30)

    assert resp.status_code == 404
    assert len(urls) == 1


def test_gemini_post_uses_sse_endpoint_when_streaming(monkeypatch):
    seen = {}

    def post(**kwargs):
        seen.update(kwargs)
        return FakeResponse(200)

    monkeypatch.setattr(models.requests, "post", post)
    models._gemini_post("gemini-3.6-flash", {}, "test", 30, stream=True)

    assert seen["url"].endswith(":streamGenerateContent")
    assert seen["params"]["alt"] == "sse"
    assert seen["stream"] is True


# ---------------------------------------------------------------------------
# Grounding: does the cited evidence actually support the claim?
# ---------------------------------------------------------------------------

def test_claim_lines_only_picks_up_cited_lines():
    import agent
    notes = "## Heading\n- supported point [3]\n- another [1][2]\nplain prose line\n"
    got = agent._claim_lines(notes)
    assert [ids for _, _, ids in got] == [[3], [1, 2]]
    assert [i for i, _, _ in got] == [1, 2]


def test_unsupported_claim_is_removed_and_supported_kept(monkeypatch):
    """The failure this exists for: a claim whose citation resolves to a real
    page that does not actually say it."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("unsupported", ""), 2: ("supported", "")},
    )
    notes = "- invented roles [7]\n- a real result [7]"
    out, stats = agent.verify_claim_support(notes, {7: {"text": "x", "page": 7}})
    assert "invented roles" not in out
    assert "a real result" in out
    assert stats["removed"] == 1 and stats["unjudged"] == 0


def test_partial_claim_is_rewritten_not_deleted(monkeypatch):
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("partial", "- debate improved results [7]")},
    )
    out, stats = agent.verify_claim_support(
        "- debate uses four named roles and improved results [7]",
        {7: {"text": "x", "page": 7}},
    )
    assert "four named roles" not in out and "improved results [7]" in out
    assert stats["rewritten"] == 1


def test_grounding_fails_open_when_the_model_errors(monkeypatch):
    """A broken check must never silently gut the notes."""
    import agent

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent, "_verify_batch", boom)
    notes = "- a claim [7]\n- another [7]"
    out, stats = agent.verify_claim_support(notes, {7: {"text": "x", "page": 7}})
    assert out == notes
    assert stats["unjudged"] == 2 and stats["removed"] == 0


def test_partial_rewrite_without_a_citation_is_ignored(monkeypatch):
    """A 'fix' that drops the citation would strip provenance, so it is refused."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("partial", "no citation here")},
    )
    notes = "- a claim [7]"
    out, _ = agent.verify_claim_support(notes, {7: {"text": "x", "page": 7}})
    assert out == notes
