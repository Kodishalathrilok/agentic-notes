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
#
# Selection is deliberately INDEPENDENT of citation markers. A claim used to be
# checked only if it already carried "[n]", so an invented uncited line was
# examined by nothing at all.

CHUNKS = {7: {"text": "Lists support append and pop.", "page": 7}}


def test_claim_selection_does_not_depend_on_citations():
    import agent
    notes = (
        "## Heading\n"
        "- a supported statement about lists and their methods [3]\n"
        "- an uncited statement asserting something substantive here\n"
        "plain prose that also asserts something substantive here\n"
    )
    got = agent._claim_lines(notes)
    assert [i for i, _, _ in got] == [1, 2, 3]
    assert [ids for _, _, ids in got] == [[3], [], []]


def test_headings_and_labels_are_not_claims():
    """Structure carries no assertion — checking it wastes budget, and deleting
    it would damage the notes."""
    import agent
    notes = (
        "# Unit 2\n"
        "**Unit 2: Data Structures**\n"
        "---\n"
        "• **Lists**\n"
        "Sets:\n"
        "- this line actually asserts a fact about the source material\n"
    )
    assert [i for i, _, _ in agent._claim_lines(notes)] == [5]


def test_citation_bearing_heading_is_still_a_heading():
    """A section title that happens to carry a citation must not be treated as
    a normal factual claim."""
    import agent
    got = agent._claim_lines("**Unit 3: Functions** [4]\n")
    assert got == []


def test_unsupported_uncited_claim_is_removed(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "_verify_batch",
                        lambda items, cm, model=None: {1: ("unsupported", "")})
    notes = "- Python f-strings use a colon to control decimal precision"
    out, stats = agent.verify_claim_support(notes, CHUNKS)
    assert "f-strings" not in out
    assert stats["removed"] == 1 and stats["unjudged"] == 0


def test_supported_uncited_claim_is_kept(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "_verify_batch",
                        lambda items, cm, model=None: {1: ("supported", "")})
    notes = "- Lists support the append and pop operations described here"
    out, stats = agent.verify_claim_support(notes, CHUNKS)
    assert out == notes and stats["supported"] == 1


def test_supported_cited_claim_is_kept(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "_verify_batch",
                        lambda items, cm, model=None: {1: ("supported", "")})
    notes = "- Lists support the append and pop operations described here [7]"
    out, _ = agent.verify_claim_support(notes, CHUNKS)
    assert out == notes


def test_mixed_supported_and_unsupported(monkeypatch):
    """The unsupported half goes; the supported half stays untouched."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("supported", ""), 2: ("unsupported", "")},
    )
    notes = (
        "- Lists support the append and pop operations described here [7]\n"
        "- Stacks are LIFO structures implemented with those same methods\n"
    )
    out, stats = agent.verify_claim_support(notes, CHUNKS)
    assert "append and pop" in out and "LIFO" not in out
    assert stats["removed"] == 1 and stats["supported"] == 1


def test_partial_claim_is_rewritten_not_deleted(monkeypatch):
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("partial", "- debate improved the results [7]")},
    )
    out, stats = agent.verify_claim_support(
        "- debate uses four named roles and improved the results [7]", CHUNKS)
    assert "four named roles" not in out and "improved the results [7]" in out
    assert stats["rewritten"] == 1


def test_planted_fabrication_regression(monkeypatch):
    """The original failure: a real page cited for a claim it never makes."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("unsupported", ""), 2: ("supported", "")},
    )
    notes = (
        "- Roles: proposer, critic, verifier and summarizer divide the labour [7]\n"
        "- Lists support the append and pop operations described here [7]\n"
    )
    out, _ = agent.verify_claim_support(notes, CHUNKS)
    assert "proposer" not in out and "append and pop" in out


def test_grounding_fails_open_when_the_model_errors(monkeypatch):
    """A broken check must never silently gut the notes."""
    import agent

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent, "_verify_batch", boom)
    notes = ("- a substantive claim about the source document [7]\n"
             "- another substantive claim about the source document [7]")
    out, stats = agent.verify_claim_support(notes, CHUNKS)
    assert out == notes
    assert stats["unjudged"] == 2 and stats["removed"] == 0


def test_partial_rewrite_without_a_citation_is_allowed_for_uncited_claims(monkeypatch):
    """A repair must keep a citation the claim HAD; an uncited claim never had
    one, so a citation-free repair is legitimate there."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("partial", "- lists support append and pop")},
    )
    notes = "- lists support append, pop and every other sequence protocol method"
    out, stats = agent.verify_claim_support(notes, CHUNKS)
    assert "every other sequence protocol" not in out
    assert stats["rewritten"] == 1


def test_partial_rewrite_dropping_an_existing_citation_is_refused(monkeypatch):
    """Losing provenance is worse than leaving the claim as written."""
    import agent
    monkeypatch.setattr(
        agent, "_verify_batch",
        lambda items, cm, model=None: {1: ("partial", "no citation in this repair")},
    )
    notes = "- lists support append and pop operations as described [7]"
    out, _ = agent.verify_claim_support(notes, CHUNKS)
    assert out == notes


def test_full_source_block_only_when_a_claim_is_uncited():
    import agent
    cited = [(0, "- a claim with a citation attached here [7]", [7])]
    uncited = [(0, "- a claim with no citation attached at all", [])]
    assert agent._full_source_block(cited, CHUNKS) == ""
    assert "FULL SOURCE" in agent._full_source_block(uncited, CHUNKS)


# ---------------------------------------------------------------------------
# Provider observability: which provider ACTUALLY served the request
# ---------------------------------------------------------------------------

def _serving(monkeypatch, ready, behaviour):
    """Record on_serve while faking provider readiness and per-provider results."""
    import models
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ready)

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode):
        out = behaviour[prov]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(models, "_dispatch", dispatch)
    seen = {}
    models.call_model("x", model="nvidia/some-model",
                      on_serve=lambda p, m, fb, r: seen.update(
                          provider=p, model=m, fallback_used=fb, reason=r))
    return seen


def test_nvidia_success_reports_nvidia_without_fallback(monkeypatch):
    seen = _serving(monkeypatch, {"nvidia", "gemini"},
                    {"nvidia": "real notes", "gemini": "should not be used"})
    assert seen["provider"] == "nvidia"
    assert seen["fallback_used"] is False
    assert seen["reason"] == ""


def test_nvidia_failure_reports_the_provider_that_actually_served(monkeypatch):
    seen = _serving(monkeypatch, {"nvidia", "gemini"},
                    {"nvidia": RuntimeError("503 Service Unavailable"),
                     "gemini": "fallback notes"})
    assert seen["provider"] == "gemini"
    assert seen["fallback_used"] is True
    assert seen["reason"] == "nvidia_overloaded"


def test_empty_nvidia_completion_counts_as_fallback(monkeypatch):
    """An empty answer is a failure, and the recorded provider must be the one
    that produced real text."""
    seen = _serving(monkeypatch, {"nvidia", "gemini"},
                    {"nvidia": "   ", "gemini": "fallback notes"})
    assert seen["provider"] == "gemini" and seen["fallback_used"] is True
    assert seen["reason"] == "nvidia_empty_response"


def test_fallback_reason_never_leaks_a_secret():
    """Provider errors embed the request URL, which carries ?key=... — the slug
    is built from the exception TYPE, never its message."""
    import models
    leaky = RuntimeError("404 for url: https://host/v1/models/m?key=SUPERSECRET123")
    slug = models._fallback_reason([("nvidia", leaky)])
    assert "SUPERSECRET123" not in slug
    assert slug == "nvidia_model_unavailable"


def test_stream_reports_provider_on_first_delta(monkeypatch):
    import models
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")

    def stream(prov, prompt, max_tokens, model, temperature):
        yield "a"
        yield "b"

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    seen = []
    out = "".join(models.call_model_stream(
        "x", model="nvidia/some-model",
        on_serve=lambda p, m, fb, r: seen.append((p, fb))))
    assert out == "ab"
    assert seen == [("nvidia", False)], "on_serve must fire exactly once, on the first delta"


def test_on_serve_is_optional_so_existing_callers_are_unaffected(monkeypatch):
    import models
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    monkeypatch.setattr(models, "_dispatch", lambda *a, **k: "notes")
    assert models.call_model("x", model="nvidia/some-model") == "notes"
