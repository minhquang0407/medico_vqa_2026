"""Recompute Task 2 confidence with Bayesian-style log-odds aggregation.

This post-processes an existing Task 2 JSONL file. It reads each row's
`visual_explanation` evidence JSON and, when available, the internal `debug`
probe answers to estimate a reliability confidence via log-odds evidence
aggregation.

The score is not a calibrated clinical probability. It is a Bayesian-style
reliability estimate over visual evidence, artifact burden, and self-probe
agreement.

Usage:
    python apply_bayesian_confidence_task2.py \
      --input submission_task2.jsonl \
      --output submission_task2_bayes.jsonl

If the input still contains `debug`, the script can use probe agreement. The
output omits `debug` by default for clean submission. Add `--include-debug true`
to keep it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

NEGATIVE_TERMS = (
    "no ", "not ", "none", "without", "absent", "no evidence",
    "not identified", "not observed", "not detected", "no significant",
)
POSITIVE_TERMS = (
    "evidence", "visible", "present", "observed", "identified", "detected",
    "lesion", "polyp", "abnormal", "instrument", "text", "erythema",
    "inflammation", "ulcer", "tube",
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def soft_cap(x: float, scale: float) -> float:
    """Map non-negative evidence to [0, 1) with smooth saturation."""
    x = max(0.0, float(x))
    if scale <= 0:
        return 0.0
    return 1.0 - math.exp(-x / scale)


def is_negative(text: str) -> bool:
    t = " " + str(text).lower() + " "
    return any(term in t for term in NEGATIVE_TERMS)


def has_positive_evidence(text: str) -> bool:
    t = str(text).lower()
    return any(term in t for term in POSITIVE_TERMS)


def probe_agreement(answer: str, probe_answers: Iterable[str]) -> Tuple[float, bool, int, int]:
    """Return agreement in [0,1], conflict flag, positive count, negative count."""
    probes = [str(x).strip() for x in probe_answers if str(x).strip()]
    if not probes:
        return 0.50, False, 0, 0

    ans_neg = is_negative(answer)
    pos_count = sum(1 for p in probes if has_positive_evidence(p) and not is_negative(p))
    neg_count = sum(1 for p in probes if is_negative(p))

    if ans_neg:
        agree = neg_count / max(1, len(probes))
        conflict = pos_count > 0
    else:
        agree = pos_count / max(1, len(probes))
        conflict = neg_count >= 2
    return clamp01(agree), conflict, pos_count, neg_count


def resolve_data_path(base_dir: Path, row: Dict[str, Any], wanted_type: str) -> Path | None:
    for item in row.get("visual_explanation", []) or []:
        if item.get("type") == wanted_type and item.get("data"):
            path = Path(str(item["data"]))
            if not path.is_absolute():
                path = base_dir / path
            return path
    return None


def load_evidence(base_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    evidence_path = resolve_data_path(base_dir, row, "evidence_json")
    if evidence_path and evidence_path.exists():
        with evidence_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def bayesian_style_confidence(row: Dict[str, Any], evidence: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Compute Bayesian-style log-odds reliability.

    The prior starts mildly uncertain. Evidence terms are transformed to bounded
    likelihood increments in log-odds space. Weights are fixed expert priors, not
    learned calibration parameters.
    """
    answer = str(row.get("answer", ""))
    debug = row.get("debug") or {}
    probe_answers = debug.get("probe_answers", []) if isinstance(debug, dict) else []
    agreement, conflict, pos_count, neg_count = probe_agreement(answer, probe_answers)

    prior_strength = soft_cap(float(evidence.get("prior_mean", 0.0)), scale=0.12)
    prior_peak = soft_cap(float(evidence.get("prior_max", 0.0)), scale=0.35)
    topo_strength = soft_cap(float(evidence.get("topo_saliency_mean", 0.0)), scale=0.18)
    topo_peak = soft_cap(float(evidence.get("topo_saliency_max", 0.0)), scale=0.65)
    focality = soft_cap(float(evidence.get("heatmap_concentration", 0.0)), scale=0.06)
    heat_strength = soft_cap(float(evidence.get("heatmap_mean", 0.0)), scale=0.20)
    artifact = soft_cap(float(evidence.get("specular_fraction", 0.0)), scale=0.06)

    # Start from a modest reliability prior rather than 0.5 because the primary
    # answer comes from the selected CATA-Final model.
    log_odds = logit(0.58)

    # Visual likelihood increments.
    log_odds += 0.70 * prior_strength
    log_odds += 0.35 * prior_peak
    log_odds += 0.60 * topo_strength
    log_odds += 0.25 * topo_peak
    log_odds += 0.55 * focality
    log_odds += 0.30 * heat_strength

    # Self-probe agreement behaves like an independent weak likelihood source.
    if probe_answers:
        log_odds += 0.90 * (agreement - 0.50) * 2.0
        if conflict:
            log_odds -= 0.95
    else:
        log_odds -= 0.10

    # Negative answers are slightly less visually anchored unless probes agree.
    if is_negative(answer):
        log_odds -= 0.20 * (1.0 - agreement)

    # Artifact burden reduces reliability.
    log_odds -= 0.70 * artifact

    confidence = sigmoid(log_odds)
    confidence = max(0.05, min(0.95, confidence))

    diagnostics = {
        "bayes_style_log_odds": round(log_odds, 6),
        "visual_prior_strength": round(prior_strength, 6),
        "visual_prior_peak": round(prior_peak, 6),
        "topo_strength": round(topo_strength, 6),
        "topo_peak": round(topo_peak, 6),
        "heatmap_focality": round(focality, 6),
        "heatmap_strength": round(heat_strength, 6),
        "artifact_burden": round(artifact, 6),
        "probe_agreement": round(agreement, 6),
        "probe_conflict": bool(conflict),
        "probe_positive_count": int(pos_count),
        "probe_negative_count": int(neg_count),
    }
    return round(float(confidence), 4), diagnostics


def update_explanation(text: str, old_score: Any, new_score: float) -> str:
    if not isinstance(text, str):
        return text
    # Add one concise sentence near the reliability-score discussion. Avoid
    # duplicating it if the script is run multiple times.
    marker = "Bayesian-style log-odds evidence aggregation"
    if marker in text:
        return text
    sentence = (
        f" {marker} estimates the confidence score from visual prior strength, "
        "topological saliency, heatmap focality, artifact burden, and self-probe agreement; "
        "it is not a calibrated clinical probability."
    )
    anchor = "This explanation is intended for clinician review"
    if anchor in text:
        return text.replace(anchor, sentence.strip() + " " + anchor, 1)
    return text.rstrip() + sentence


def process_file(input_path: Path, output_path: Path, include_debug: bool, include_bayes_debug: bool) -> Tuple[int, int]:
    base_dir = input_path.parent
    rows = 0
    changed = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            evidence = load_evidence(base_dir, row)
            old_score = row.get("confidence_score")
            new_score, diagnostics = bayesian_style_confidence(row, evidence)
            row["confidence_score"] = new_score
            row["textual_explanation"] = update_explanation(row.get("textual_explanation", ""), old_score, new_score)

            if include_bayes_debug:
                row["bayesian_confidence_debug"] = diagnostics
            else:
                row.pop("bayesian_confidence_debug", None)

            if not include_debug:
                row.pop("debug", None)

            if old_score != new_score:
                changed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Bayesian-style confidence to Task 2 JSONL.")
    parser.add_argument("--input", default="submission_task2.jsonl", help="Input Task 2 JSONL.")
    parser.add_argument("--output", default="submission_task2_bayes.jsonl", help="Output Task 2 JSONL.")
    parser.add_argument(
        "--include-debug",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Keep original internal debug field. Default false for clean submission.",
    )
    parser.add_argument(
        "--include-bayes-debug",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Include Bayesian confidence diagnostics. Default false for clean submission.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    rows, changed = process_file(
        input_path=input_path,
        output_path=output_path,
        include_debug=args.include_debug,
        include_bayes_debug=args.include_bayes_debug,
    )
    print(f"✅ Wrote Bayesian-style confidence JSONL: {output_path}")
    print(f"Rows processed: {rows}")
    print(f"Rows with changed confidence: {changed}")
    print("Note: confidence_score is a Bayesian-style reliability estimate, not a calibrated clinical probability.")


if __name__ == "__main__":
    main()
