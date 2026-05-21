import math
import re
import string
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(_PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    matcher = SequenceMatcher(a=pred_tokens, b=gold_tokens)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    beta = precision / (recall + 1e-12)
    return ((1 + beta * beta) * precision * recall) / (recall + beta * beta * precision + 1e-12)


def bleu_n(prediction: str, ground_truth: str, n: int = 1) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) < n or len(gold_tokens) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    gold_ngrams = Counter(tuple(gold_tokens[i : i + n]) for i in range(len(gold_tokens) - n + 1))
    overlap = sum((pred_ngrams & gold_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    brevity = 1.0 if len(pred_tokens) > len(gold_tokens) else math.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1))
    return brevity * precision


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    text = normalize_answer(text)
    return any(term in text for term in terms)


def _extract_yes_no(text: str):
    norm = normalize_answer(text)
    if re.search(r"\b(no|none|not|without|absent|negative)\b", norm):
        return "no"
    if re.search(r"\b(yes|present|visible|identified|seen|detected|evidence)\b", norm):
        return "yes"
    return None


def _extract_count(text: str):
    norm = normalize_answer(text)
    word_to_num = {
        "zero": 0,
        "none": 0,
        "no": 0,
        "one": 1,
        "single": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for word, value in word_to_num.items():
        if re.search(rf"\b{word}\b", norm):
            return value
    match = re.search(r"\b\d+\b", norm)
    return int(match.group(0)) if match else None


def extract_clinical_concepts(text: str) -> Dict[str, object]:
    return {
        "yes_no": _extract_yes_no(text),
        "count": _extract_count(text),
        "polyp": _contains_any(text, ["polyp", "polyps"]),
        "abnormality": _contains_any(text, ["abnormal", "abnormality", "lesion", "lesions", "finding", "findings", "inflammation", "ulcerative", "colitis", "oesophagitis", "esophagitis"]),
        "instrument": _contains_any(text, ["instrument", "instruments", "device", "devices", "tube", "foreign body", "foreign bodies", "surgical"]),
        "text": _contains_any(text, ["text", "label", "annotation", "caption"]),
        "landmark": _contains_any(text, ["zline", "z line", "landmark", "anatomical"]),
        "procedure": _contains_any(text, ["colonoscopy", "colonoscopic", "gastroscopy", "gastroscopic", "endoscopy", "endoscopic", "biopsy"]),
        "size": _contains_any(text, ["mm", "size", "small", "large", "greater", "less"]),
        "color": _contains_any(text, ["red", "pink", "white", "black", "green", "yellow", "brown", "orange"]),
        "location": _contains_any(text, ["central", "center", "centre", "upper", "lower", "left", "right", "middle", "region", "regions"]),
    }


def infer_question_type(question: str, ground_truth: str = "") -> str:
    """Infer coarse Kvasir-VQA question category from text.

    This is intentionally rule-based so it works without extra labels.
    It supports the challenge categories plus a multi-attribute bucket for
    questions containing several clinical clauses.
    """
    q = normalize_answer(question.replace("<image>", " "))
    clause_markers = [",", " and ", " or ", " with "]
    raw_q = str(question or "").lower()
    if sum(marker in raw_q for marker in clause_markers) >= 2:
        return "multi_attribute"
    if re.search(r"\b(how many|number of|count|counts)\b", q):
        return "numerical_count"
    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", q):
        return "yes_no"
    if re.search(r"\b(color|colour|red|pink|white|black|green|yellow|brown|orange)\b", q):
        return "color_related"
    if re.search(r"\b(where|location|located|region|regions|left|right|upper|lower|center|centre|central|middle)\b", q):
        return "location_related"
    if re.search(r"\b(which|what type|what kind|choice|option)\b", q):
        return "single_choice"
    if re.search(r"\b(what procedure|procedure|colonoscopy|gastroscopy|endoscopy)\b", q):
        return "procedure_related"
    if re.search(r"\b(instrument|instruments|foreign bod|device|tube)\b", q):
        return "instrument_related"
    if re.search(r"\b(polyp|polyps|abnormal|abnormality|lesion|finding|ulcerative|colitis)\b", q):
        return "finding_related"
    return "other"


def clinical_concept_scores(prediction: str, ground_truth: str, question: str = "") -> Dict[str, float]:
    pred = extract_clinical_concepts(prediction)
    gold = extract_clinical_concepts(ground_truth)
    scores = {}
    for key, gold_value in gold.items():
        pred_value = pred.get(key)
        if isinstance(gold_value, bool):
            scores[f"concept_{key}_acc"] = float(pred_value == gold_value)
        elif gold_value is not None:
            scores[f"concept_{key}_acc"] = float(pred_value == gold_value)
    return scores


def question_type_scores(prediction: str, ground_truth: str, question: str = "") -> Dict[str, float]:
    qtype = infer_question_type(question, ground_truth)
    pred = extract_clinical_concepts(prediction)
    gold = extract_clinical_concepts(ground_truth)
    scores = {
        f"qtype_{qtype}_count": 1.0,
        f"qtype_{qtype}_exact_match": exact_match(prediction, ground_truth),
        f"qtype_{qtype}_token_f1": token_f1(prediction, ground_truth),
    }

    if qtype == "yes_no" and gold["yes_no"] is not None:
        scores[f"qtype_{qtype}_answer_acc"] = float(pred["yes_no"] == gold["yes_no"])
    elif qtype == "numerical_count" and gold["count"] is not None:
        pred_count = pred["count"]
        gold_count = gold["count"]
        scores[f"qtype_{qtype}_answer_acc"] = float(pred_count == gold_count)
        scores[f"qtype_{qtype}_count_exact_acc"] = float(pred_count == gold_count)
        scores[f"qtype_{qtype}_count_tolerance1_acc"] = float(pred_count is not None and abs(pred_count - gold_count) <= 1)
    elif qtype == "color_related":
        scores[f"qtype_{qtype}_answer_acc"] = float(pred["color"] == gold["color"])
    elif qtype == "location_related":
        scores[f"qtype_{qtype}_answer_acc"] = float(pred["location"] == gold["location"])
    elif qtype == "procedure_related":
        scores[f"qtype_{qtype}_answer_acc"] = float(pred["procedure"] == gold["procedure"])
    elif qtype == "instrument_related":
        scores[f"qtype_{qtype}_answer_acc"] = float(pred["instrument"] == gold["instrument"])
    elif qtype == "finding_related":
        scores[f"qtype_{qtype}_answer_acc"] = float(
            pred["polyp"] == gold["polyp"] and pred["abnormality"] == gold["abnormality"]
        )
    elif qtype == "multi_attribute":
        keys = ["yes_no", "count", "polyp", "abnormality", "instrument", "text", "procedure", "color", "location"]
        comparable = [key for key in keys if not isinstance(gold[key], bool) or gold[key] is True]
        if comparable:
            scores[f"qtype_{qtype}_answer_acc"] = sum(float(pred[key] == gold[key]) for key in comparable) / len(comparable)

    return scores


def hallucination_flags(prediction: str, question: str = "") -> Dict[str, float]:
    norm_pred = normalize_answer(prediction)
    norm_question = normalize_answer(question)
    out_of_domain = ["eye", "eyes", "lung", "brain", "hall", "portal", "study found"]
    return {
        "empty_answer": float(len(norm_pred) == 0),
        "question_copy": float(bool(norm_question) and norm_question in norm_pred),
        "too_long": float(len(norm_pred.split()) > 40),
        "out_of_domain": float(any(term in norm_pred for term in out_of_domain)),
    }


def score_prediction(prediction: str, ground_truth: str, question: str = "") -> Dict[str, float]:
    scores = {
        "exact_match": exact_match(prediction, ground_truth),
        "token_f1": token_f1(prediction, ground_truth),
        "rouge_l": rouge_l(prediction, ground_truth),
        "bleu_1": bleu_n(prediction, ground_truth, n=1),
        "bleu_4": bleu_n(prediction, ground_truth, n=4),
    }
    scores.update(clinical_concept_scores(prediction, ground_truth, question))
    scores.update(question_type_scores(prediction, ground_truth, question))
    scores.update(hallucination_flags(prediction, question))
    return scores


def aggregate_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    aggregated = {}
    for key in keys:
        if key.endswith("_count"):
            aggregated[key] = sum(float(row.get(key, 0.0)) for row in rows)
        else:
            values = [float(row[key]) for row in rows if key in row]
            aggregated[key] = sum(values) / max(len(values), 1)
    return aggregated
