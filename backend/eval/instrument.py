"""Observe a pipeline run from the outside, without changing pipeline code.

Three recorders, each installed for the duration of a `with` block:

- UsageMeter counts characters in and out of every provider attempt
  (models._dispatch / models._dispatch_stream), by role and by served model.
  Providers' own token counts never leave models.py, so tokens here are an
  ESTIMATE (characters / 4) and are labelled as one everywhere they appear.
- TimingCapture collects the "[timing]" JSON lines run_agent already logs per
  stage, filtered to one run's gen_id.
- GroundingCapture records the stats verify_claim_support returns, by
  wrapping the module attribute run_agent calls it through.

They patch module attributes, so they observe whichever `agent` / `models`
modules are loaded - including an older commit's (claim_eval --pipeline-dir).
"""

import json
import logging
import math
import threading

CHARS_PER_TOKEN = 4  # the estimate's only parameter; reported alongside it


def estimate_tokens(chars: int) -> int:
    return int(math.ceil(max(0, chars) / CHARS_PER_TOKEN))


class UsageMeter:
    """Character counts per provider attempt, attributed to the current role.

    Roles ("pipeline", "judge") are set by the harness. They never overlap in
    time: the judge runs only after the pipeline generator is exhausted.
    """

    def __init__(self, models_module):
        self._models = models_module
        self._lock = threading.Lock()
        self.role = "pipeline"
        self.usage = {}
        self._orig = None

    def _add(self, prov, target, prompt_chars, output_chars):
        try:
            served = self._models._served_model(prov, target)
        except Exception:  # noqa: BLE001 - accounting must never break a call
            served = str(target)
        with self._lock:
            role = self.usage.setdefault(self.role, {"calls": 0, "prompt_chars": 0,
                                                     "output_chars": 0, "by_model": {}})
            role["calls"] += 1
            role["prompt_chars"] += prompt_chars
            role["output_chars"] += output_chars
            m = role["by_model"].setdefault(served, {"calls": 0, "prompt_chars": 0,
                                                     "output_chars": 0})
            m["calls"] += 1
            m["prompt_chars"] += prompt_chars
            m["output_chars"] += output_chars

    def __enter__(self):
        models = self._models
        self._orig = (models._dispatch, models._dispatch_stream)
        orig_call, orig_stream = self._orig
        meter = self

        def _dispatch(prov, prompt, max_tokens, model, temperature, json_mode, *a, **kw):
            out = ""
            try:
                out = orig_call(prov, prompt, max_tokens, model, temperature, json_mode,
                                *a, **kw)
                return out
            finally:
                meter._add(prov, model, len(prompt or ""), len(out or ""))

        def _dispatch_stream(prov, prompt, max_tokens, model, temperature, *a, **kw):
            produced = 0
            try:
                for delta in orig_stream(prov, prompt, max_tokens, model, temperature,
                                         *a, **kw):
                    produced += len(delta or "")
                    yield delta
            finally:
                meter._add(prov, model, len(prompt or ""), produced)

        models._dispatch, models._dispatch_stream = _dispatch, _dispatch_stream
        return self

    def __exit__(self, *exc):
        self._models._dispatch, self._models._dispatch_stream = self._orig
        return False

    def snapshot(self, role):
        """Totals for one role, with the token estimate attached."""
        r = self.usage.get(role) or {"calls": 0, "prompt_chars": 0, "output_chars": 0,
                                     "by_model": {}}
        return {
            "calls": r["calls"],
            "prompt_chars": r["prompt_chars"],
            "output_chars": r["output_chars"],
            "est_input_tokens": estimate_tokens(r["prompt_chars"]),
            "est_output_tokens": estimate_tokens(r["output_chars"]),
            "by_model": {k: dict(v) for k, v in r["by_model"].items()},
        }

    def reset(self):
        with self._lock:
            self.usage = {}


def estimate_cost(snapshot, prices):
    """USD from per-model prices, or None when any model used has no price.

    `prices` maps a served model id to {"input_per_mtok", "output_per_mtok"}
    in USD per million tokens. A partial sum would understate the cost and
    look like a real number, so a single unpriced model makes the cost None
    and names the model instead.
    """
    total, unpriced = 0.0, []
    for model_id, m in (snapshot.get("by_model") or {}).items():
        p = (prices or {}).get(model_id)
        if not p:
            unpriced.append(model_id)
            continue
        total += (estimate_tokens(m["prompt_chars"]) * float(p["input_per_mtok"])
                  + estimate_tokens(m["output_chars"]) * float(p["output_per_mtok"])) / 1e6
    if unpriced:
        return None, sorted(unpriced)
    return round(total, 6), []


class TimingCapture(logging.Handler):
    """Collect run_agent's "[timing]" stage lines for one gen_id."""

    def __init__(self, gen_id, logger_name="agentic"):
        super().__init__(level=logging.INFO)
        self.gen_id = gen_id
        self.stages = []
        self._logger = logging.getLogger(logger_name)
        self._prev_level = None

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if not msg.startswith("[timing] "):
            return
        try:
            payload = json.loads(msg[len("[timing] "):])
        except (TypeError, ValueError):
            return
        if payload.get("gen_id") == self.gen_id:
            self.stages.append(payload)

    def __enter__(self):
        self._prev_level = self._logger.level
        # The pipeline logs timings at INFO; without this they are dropped
        # before any handler sees them when the root logger is at WARNING.
        if self._logger.getEffectiveLevel() > logging.INFO:
            self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev_level)
        return False


class GroundingCapture:
    """Record every verify_claim_support result during a run."""

    def __init__(self, agent_module):
        self._agent = agent_module
        self._orig = None
        self.calls = []

    def __enter__(self):
        self._orig = self._agent.verify_claim_support
        orig, calls = self._orig, self.calls

        def verify_claim_support(*args, **kwargs):
            notes, stats = orig(*args, **kwargs)
            calls.append(dict(stats or {}))
            return notes, stats

        self._agent.verify_claim_support = verify_claim_support
        return self

    def __exit__(self, *exc):
        self._agent.verify_claim_support = self._orig
        return False

    def totals(self):
        out = {}
        for stats in self.calls:
            for k, v in stats.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = out.get(k, 0) + v
        return out
