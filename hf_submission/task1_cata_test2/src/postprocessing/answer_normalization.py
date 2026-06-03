"""Minimal answer cleanup for Medico 2026 Task 1 submissions.

Full-test rescoring showed that semantic or qtype-specific normalization can
reduce the official text metrics. Keep this module intentionally conservative:
only remove obvious formatting noise and never rewrite the medical meaning.
"""

from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")
_LEADING_ANSWER_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)+", re.IGNORECASE)


def compact_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def normalize_prediction(prediction: str, question: str = "") -> str:
    """Return a minimally cleaned prediction without semantic rewrites."""

    del question  # The final Task 1 normalizer is intentionally question-agnostic.
    text = compact_spaces(prediction)
    text = _LEADING_ANSWER_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return compact_spaces(text).strip()


def infer_question_type_for_normalization(question: str) -> str:
    """Compatibility shim for older imports; not used for rewriting."""

    q = str(question or "").lower()
    if any(marker in q for marker in ["how many", "number of", "count"]):
        return "numerical_count"
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "has ", "have ")):
        return "yes_no"
    return "other"


__all__ = ["normalize_prediction", "infer_question_type_for_normalization"]
