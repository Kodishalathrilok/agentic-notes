"""Lightweight tokenizer for BM25 — no heavy NLP dependencies.

- lowercases
- drops punctuation
- keeps technical terms intact: hyphen/dot/underscore-joined tokens stay whole
  (e.g. `gpt-4`, `sha-256`, `numpy.array`, `big_o`)
"""

import re

# One or more alphanumeric runs, optionally joined by . _ - (technical terms).
_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str):
    return _TOKEN.findall((text or "").lower())
