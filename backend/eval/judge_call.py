"""The judge's one way to reach a model: its own provider, no failover.

call_model() fails over to the next provider on any error. For the pipeline
that is the point; for the judge it would be a silent breach of the
different-family rule - an NVIDIA judge that hits a 503 would be answered by
Gemini, the writer's family, and the score would look like any other. So a
judge call goes to its own provider only (models._dispatch, which keeps the
provider's own 429/503 retries) and fails loudly otherwise. A failed judge
call means the sample was not measured; callers drop it, never score it.
"""

import models


class JudgeUnavailableError(RuntimeError):
    """The judge's provider is not configured or did not answer."""


def call_judge(prompt, model, max_tokens=1024, temperature=0.0, json_mode=True):
    if not (model or "").strip():
        raise JudgeUnavailableError("no judge model given")
    prov = models._provider_for(model)
    if not models._provider_ready(prov):
        raise JudgeUnavailableError(f"judge provider {prov!r} is not configured")
    try:
        out = models._dispatch(prov, prompt, max_tokens, model, temperature, json_mode)
    except Exception as exc:  # noqa: BLE001
        raise JudgeUnavailableError(f"judge call failed: {type(exc).__name__}") from None
    if not (out or "").strip():
        raise JudgeUnavailableError("judge returned an empty completion")
    return out
