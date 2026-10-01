"""Model families, so the eval can refuse to let a model grade its own work.

A judge from the writer's family shares its blind spots and its style
preferences: it tends to accept the claims that family is prone to make. The
eval therefore requires a judge from a DIFFERENT family, checked twice: when
the run starts (configured ids), and after it (the models that actually
served, since failover can move the writer to another provider mid-run).

Families follow the base model, not the host: NVIDIA serves Llama, Mistral
and its own Nemotron models, and a Nemotron fine-tune of Llama is Llama.
"""

# Checked in order; the first key found in the lower-cased id wins. "llama"
# precedes "nemotron" because llama-3.1-nemotron-* is a Llama fine-tune.
_FAMILY_KEYS = (
    ("gemini", "google"),
    ("gemma", "google"),
    ("llama", "meta-llama"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
    ("nemotron", "nvidia-nemotron"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("phi-", "microsoft-phi"),
    ("gpt", "openai"),
    ("claude", "anthropic"),
)


def model_family(model_id: str) -> str:
    mid = (model_id or "").strip().lower()
    if not mid:
        return "unknown"
    for key, family in _FAMILY_KEYS:
        if key in mid:
            return family
    # Unknown shapes: the catalog namespace ("vendor/model") is the best
    # available guess, else the id itself, so two unknown ids only collide
    # when they really are the same vendor or model.
    return mid.split("/", 1)[0] if "/" in mid else mid


class SameFamilyError(ValueError):
    """The judge is from the same model family as the writer."""


def require_different_families(writer_model: str, judge_model: str) -> None:
    if not (judge_model or "").strip():
        raise SameFamilyError("A judge model is required (--judge-model); it may not "
                              "default to the writer model.")
    writer, judge = model_family(writer_model), model_family(judge_model)
    if writer == judge:
        raise SameFamilyError(
            f"Judge {judge_model!r} and writer {writer_model!r} are both in the "
            f"{writer!r} family. Pick a judge from a different family.")
