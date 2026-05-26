"""Conservative answer normalization for Medico 2026 VQA submissions.

The normalizer intentionally avoids using ground-truth labels. It only uses the
model prediction and, when useful, the question text. Rules should therefore be
safe for hidden-set inference and official submission.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional


_LEADING_NOISE_RE = re.compile(r"^(?:[\s\.\-_:;]+|answer\s*[:\-]\s*)+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

_PREFIX_PATTERNS = [
    re.compile(r"^based on (?:the|this) image\s*,?\s*", re.IGNORECASE),
    re.compile(r"^in (?:the|this) image\s*,?\s*", re.IGNORECASE),
    re.compile(r"^(?:the|this) image (?:shows|demonstrates|depicts)\s+", re.IGNORECASE),
    re.compile(r"^there (?:appears to be|is|are)\s+", re.IGNORECASE),
]

_SIMPLE_REPLACEMENTS = [
    # Keep this list intentionally small. Smoke re-score showed that broad
    # synonym rewrites (artefact->artifact, single->one, surgical instruments
    # -> instruments, colonoscopic examination->colonoscopy) can reduce BLEU/
    # ROUGE/F1 because references often use the original wording.
    (re.compile(r"\bz\s*-\s*line\b", re.IGNORECASE), "z-line"),
]


_NUMBER_WORDS = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
}


_QTYPE_PATTERNS = {
    "numerical_count": re.compile(r"\b(how many|number of|count|counts)\b", re.IGNORECASE),
    "color_related": re.compile(r"\b(colou?r|red|pink|white|black|green|yellow|brown|orange)\b", re.IGNORECASE),
    "location_related": re.compile(r"\b(where|location|located|region|regions|left|right|upper|lower|center|centre|central|middle)\b", re.IGNORECASE),
    "procedure_related": re.compile(r"\b(what procedure|procedure|colonoscopy|gastroscopy|endoscopy)\b", re.IGNORECASE),
    "instrument_related": re.compile(r"\b(instrument|instruments|foreign bod|device|tube)\b", re.IGNORECASE),
    "finding_related": re.compile(r"\b(polyp|polyps|abnormal|abnormality|lesion|finding|ulcerative|colitis|inflammation)\b", re.IGNORECASE),
}


def compact_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def infer_question_type_for_normalization(question: str) -> str:
    q = str(question or "").replace("<image>", " ").strip().lower()
    raw_q = str(question or "").lower()
    if sum(marker in raw_q for marker in [",", " and ", " or ", " with "]) >= 2:
        return "multi_attribute"
    for qtype, pattern in _QTYPE_PATTERNS.items():
        if pattern.search(q):
            return qtype
    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", q):
        return "yes_no"
    if re.search(r"\b(which|what type|what kind|choice|option)\b", q):
        return "single_choice"
    return "other"


def _strip_noise(text: str) -> str:
    text = compact_spaces(text)
    text = _LEADING_NOISE_RE.sub("", text)
    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub("", text)
    # Remove repeated dotted artifacts left by some generations.
    text = re.sub(r"(?:^|\s)(?:\.\s*){2,}", " ", text)
    return compact_spaces(text)


def _apply_simple_replacements(text: str) -> str:
    for pattern, replacement in _SIMPLE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return compact_spaces(text)


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])(?=\S)", r"\1 ", text)
    return compact_spaces(text).strip(" ;:")


def _normalize_count_phrasing(text: str, question: str) -> str:
    qtype = infer_question_type_for_normalization(question)
    if qtype not in {"numerical_count", "multi_attribute"}:
        return text
    # Only remove weak approximation before explicit digits. Do not rewrite
    # "single" to "one" because references use both forms and smoke testing
    # showed broad count synonym rewrites can hurt language metrics.
    text = re.sub(r"\b(?:approximately|about)\s+(\d+)\b", r"\1", text, flags=re.IGNORECASE)
    return compact_spaces(text)


def _normalize_no_phrases(text: str) -> str:
    # Disabled for Task 1 scoring: phrase rewrites such as
    # "No text observed" -> "No visible text observed" improve some rows but
    # hurt many references that already match the shorter wording.
    return compact_spaces(text)


def _extract_options(question: str) -> List[str]:
    q = str(question or "")
    # Conservative extraction for explicit option syntax only.
    patterns = [
        r"(?:options?|choices?)\s*[:\-]\s*(.+)$",
        r"choose from\s*[:\-]?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        tail = match.group(1)
        parts = re.split(r"\s*(?:\||/|;|,|\bor\b)\s*", tail)
        options = [compact_spaces(p.strip(" .()[]")) for p in parts]
        return [opt for opt in options if 1 <= len(opt.split()) <= 8]
    return []


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _maybe_map_to_option(text: str, question: str) -> str:
    if infer_question_type_for_normalization(question) != "single_choice":
        return text
    options = _extract_options(question)
    if not options:
        return text
    scored = [(option, _similarity(text, option)) for option in options]
    best_option, best_score = max(scored, key=lambda item: item[1])
    return best_option if best_score >= 0.78 else text


def normalize_prediction(prediction: str, question: str = "") -> str:
    """Normalize a generated answer without using ground truth."""

    text = _strip_noise(prediction)
    if not text:
        return ""

    text = _apply_simple_replacements(text)
    text = _normalize_count_phrasing(text, question)
    text = _normalize_no_phrases(text)
    text = _maybe_map_to_option(text, question)
    text = _normalize_punctuation(text)

    # Preserve natural sentence casing when model already starts uppercase, but
    # keep lowercase labels lowercase. This avoids unnecessary metric churn.
    return text


__all__ = ["normalize_prediction", "infer_question_type_for_normalization"]
