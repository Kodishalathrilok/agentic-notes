"""The NVIDIA request: thinking that can really be switched off, and an empty
reply that is asked for again.

Measured on Nemotron 3 with one real window of a 55-page document (cap 1,100
tokens + 1,024 headroom):

  Ultra, thinking on        91.1 s  first word after 78.5 s  18 citations
  Ultra, "detailed thinking off" system message (the old switch)
                            91.1 s  9,054 chars of thinking, 83 chars of answer
  Ultra, the fields below   16.6-19.6 s  first word after 1.7 s  23-24 citations
  Super, thinking on        23.8 s  no answer at all (the cap went on thinking)
  Super, the fields below    3.9-6.4 s  a full cited answer

The critique and the grounding check returned nothing readable with thinking
on, and valid JSON with it off. So the old switch did nothing, and "on" was
costing most of every call. Separately, 5 of 11 calls to Ultra came back as a
200 with an empty body inside about a second.
"""
import json
import os
import subprocess
import sys

import pytest
import requests

import models

MODEL = "nvidia/some-model"
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# thinking off
# ---------------------------------------------------------------------------

def _body(monkeypatch, thinking, json_mode=False, stream=False):
    monkeypatch.setattr(models, "NVIDIA_THINKING", thinking)
    return models._nvidia_body("prompt", 100, MODEL, 0.4, json_mode, stream)


@pytest.mark.parametrize("value", ["off", "false", "0"])
def test_off_sends_the_fields_that_were_measured_to_work(monkeypatch, value):
    body = _body(monkeypatch, value)
    assert body["reasoning_effort"] == "none"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "prompt"}], (
        "the measured request carried no system message")


def test_on_leaves_the_request_alone(monkeypatch):
    body = _body(monkeypatch, "on")
    assert "reasoning_effort" not in body and "chat_template_kwargs" not in body
    assert body["messages"] == [{"role": "user", "content": "prompt"}]


def test_thinking_is_off_unless_someone_asks_for_it():
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_THINKING"}
    out = subprocess.run([sys.executable, "-c", "import models; print(models.NVIDIA_THINKING)"],
                         cwd=BACKEND, env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "off", out.stderr[-300:]


class _Resp:
    def __init__(self, status=200, payload=None, lines=()):
        self.status_code, self.headers = status, {}
        self._payload, self._lines = payload or {}, list(lines)

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
        return iter(ln.encode("utf-8") for ln in self._lines)


OK = _Resp(200, {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})


@pytest.fixture
def posts(monkeypatch):
    """Real _nvidia_post; requests.post answers 400 while `rejects` names a
    field that is still in the body."""
    sent, state = [], {"rejects": ()}

    def post(**kwargs):
        sent.append(kwargs["json"])
        bad = any(k in kwargs["json"] for k in state["rejects"])
        return _Resp(400) if bad else OK

    monkeypatch.setattr(models.requests, "post", post)
    monkeypatch.setattr(models, "_nvidia_cooldown", {})
    state["sent"] = sent
    return state


def test_a_model_that_rejects_the_thinking_fields_is_asked_again_without_them(
        posts, monkeypatch):
    posts["rejects"] = ("reasoning_effort", "chat_template_kwargs")
    body = _body(monkeypatch, "off", json_mode=True)
    assert models._nvidia_post(body).json() == OK.json()
    assert len(posts["sent"]) == 2
    retry = posts["sent"][1]
    assert "reasoning_effort" not in retry and "chat_template_kwargs" not in retry
    assert retry["response_format"] == {"type": "json_object"}, "only the rejected fields go"
    assert retry["messages"] == body["messages"] and retry["max_tokens"] == body["max_tokens"]


def test_a_model_that_rejects_json_mode_as_well_still_gets_an_answer(posts, monkeypatch):
    posts["rejects"] = ("reasoning_effort", "chat_template_kwargs", "response_format")
    assert models._nvidia_post(_body(monkeypatch, "off", json_mode=True)).json() == OK.json()
    assert len(posts["sent"]) == 3
    assert "response_format" not in posts["sent"][2]


def test_json_mode_alone_is_still_retried_once_as_before(posts, monkeypatch):
    posts["rejects"] = ("response_format",)
    assert models._nvidia_post(_body(monkeypatch, "on", json_mode=True)).json() == OK.json()
    assert len(posts["sent"]) == 2


def test_a_request_the_model_accepts_is_sent_once(posts, monkeypatch):
    models._nvidia_post(_body(monkeypatch, "off", json_mode=True))
    assert len(posts["sent"]) == 1


# ---------------------------------------------------------------------------
# an empty 200 is asked for again
# ---------------------------------------------------------------------------

def _sse(*deltas, finish="stop"):
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": d}}]}) for d in deltas]
    last = {"choices": [{"delta": {}}]}
    if finish:
        last["choices"][0]["finish_reason"] = finish
        lines.append("data: " + json.dumps(last))
    return _Resp(200, lines=lines + ["data: [DONE]"])


EMPTY_STREAM = _Resp(200, lines=["data: [DONE]"])
EMPTY_JSON = _Resp(200, {"choices": []})


@pytest.fixture
def replies(monkeypatch):
    """_nvidia_post hands back the queued responses in order."""
    queue, seen = [], []

    def post(body, stream=False):
        seen.append(body)
        return queue.pop(0)

    monkeypatch.setattr(models, "_nvidia_post", post)
    monkeypatch.setattr(models, "NVIDIA_EMPTY_RETRIES", 2)
    return queue, seen


def test_an_empty_stream_is_asked_for_again(replies):
    queue, seen = replies
    queue += [EMPTY_STREAM, _sse("Part ", "one")]
    assert "".join(models._stream_nvidia("p", 100, MODEL, 0.4)) == "Part one"
    assert len(seen) == 2


def test_a_stream_that_stays_empty_gives_up_after_the_retries(replies):
    queue, seen = replies
    queue += [EMPTY_STREAM, EMPTY_STREAM, EMPTY_STREAM, _sse("never reached")]
    assert list(models._stream_nvidia("p", 100, MODEL, 0.4)) == []
    assert len(seen) == 3, "one request and two retries"


def test_a_stream_with_text_is_never_asked_for_twice(replies):
    queue, seen = replies
    queue += [_sse("Part ", "one", finish=None), _sse("again")]
    assert "".join(models._stream_nvidia("p", 100, MODEL, 0.4)) == "Part one"
    assert len(seen) == 1, "text already sent to the reader would be duplicated"


def test_a_stream_cut_at_the_token_cap_is_not_an_empty_reply(replies):
    """No text because the budget went on thinking: the provider said so, and
    asking again with the same budget would get the same answer."""
    queue, seen = replies
    queue += [_sse(finish="length"), _sse("again")]
    with pytest.raises(models.IncompleteStreamError):
        list(models._stream_nvidia("p", 100, MODEL, 0.4))
    assert len(seen) == 1


def test_an_empty_completion_is_asked_for_again(replies):
    queue, seen = replies
    queue += [EMPTY_JSON, OK]
    assert models._call_nvidia("p", 100, MODEL, 0.4, False) == "answer"
    assert len(seen) == 2


def test_a_completion_that_stays_empty_is_returned_empty_for_failover(replies):
    queue, seen = replies
    queue += [EMPTY_JSON, EMPTY_JSON, EMPTY_JSON, OK]
    assert models._call_nvidia("p", 100, MODEL, 0.4, False) == ""
    assert len(seen) == 3


def test_a_completion_with_a_finish_reason_is_not_asked_for_again(replies):
    queue, seen = replies
    queue += [_Resp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}),
              OK]
    assert models._call_nvidia("p", 100, MODEL, 0.4, False) == ""
    assert len(seen) == 1
