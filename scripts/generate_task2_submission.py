"""Generate MediaEval Medico 2026 Task 2 explanation submission.

This script is intended to run locally/Colab with GPU. It creates:
  hf_submission/task2/submission_task2.jsonl
  hf_submission/task2/visuals/*.png
  hf_submission/task2/visuals/*.json

It reuses the Task 1 submission model/helpers to keep answers identical to the
FINAL Task 1 pipeline, then enriches explanations with targeted self-probing,
structural/TDA evidence, OT attention, and Bayes-gated confidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Image as HfImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJECT_ROOT / "hf_submission" / "task1"
TASK2_DIR = PROJECT_ROOT / "hf_submission" / "task2"

# Keep the project root first so `src.inference` resolves to the full project
# package. The Task 1 submission directory is only needed for importing
# `submission_task1.py`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TASK1_DIR) not in sys.path:
    sys.path.append(str(TASK1_DIR))

from src.inference.bayesian_gate import BayesianExplanationGate
from src.inference.visualizer import normalize_map, overlay_heatmap, quadrant_from_heatmap

import submission_task1 as task1


COLOR_WORDS = {"red", "pink", "white", "yellow", "black", "green", "brown", "pale", "dark"}
LOCATION_WORDS = {"left", "right", "upper", "lower", "central", "center", "proximal", "distal", "middle"}
PRESENCE_POSITIVE = {"yes", "present", "visible", "identified", "observed", "seen", "there is", "noted"}
PRESENCE_NEGATIVE = {"no", "absent", "not visible", "not seen", "none", "without"}


def _load_task2_subset(max_samples: int | None = None):
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    val_set_task2 = (
        ds.filter(lambda x: x["complexity"] == 1)
        .shuffle(seed=42)
        .select(range(1500))
        .add_column("val_id", list(range(1500)))
        .remove_columns(["complexity", "answer", "original", "question_class"])
        .cast_column("image", HfImage())
    )
    if max_samples is not None:
        val_set_task2 = val_set_task2.select(range(min(max_samples, len(val_set_task2))))
    return val_set_task2


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _cosine_map_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    av = normalize_map(a).reshape(-1)
    bv = normalize_map(b).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return 0.5
    return float(np.clip(np.dot(av, bv) / denom, 0.0, 1.0))


def _concentration(heatmap: np.ndarray, top_ratio: float = 0.10) -> float:
    flat = normalize_map(heatmap).reshape(-1)
    total = float(flat.sum())
    if total <= 1e-8:
        return 0.0
    k = max(1, int(math.ceil(len(flat) * top_ratio)))
    return float(np.sort(flat)[::-1][:k].sum() / total)


def _tokenize(text: str) -> List[str]:
    return [tok.strip(".,;:!?()[]{}\"'").lower() for tok in str(text or "").split() if tok.strip()]


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lower = str(text or "").lower()
    return any(term in lower for term in terms)


def _answer_specificity(answer: str) -> float:
    text = str(answer or "").strip().lower()
    if not text:
        return 0.0
    generic = {
        "yes", "no", "none", "normal", "abnormal", "not visible", "visible",
        "present", "absent", "unknown", "unclear",
    }
    tokens = _tokenize(text)
    if text in generic:
        return 0.25
    return float(np.clip(0.35 + 0.08 * min(len(tokens), 8), 0.35, 0.95))


def _generic_answer_penalty(answer: str) -> float:
    specificity = _answer_specificity(answer)
    return float(np.clip(1.0 - specificity, 0.0, 1.0))


def _question_answer_compatibility(question: str, answer: str) -> float:
    q = str(question or "").lower()
    a = str(answer or "").lower()
    if not a.strip():
        return 0.0
    score = 0.60
    if any(k in q for k in ["how many", "number", "count"]):
        number_words = {"zero", "one", "two", "three", "four", "five", "no", "single", "multiple"}
        score = 0.75 if any(ch.isdigit() for ch in a) or any(w in a.split() for w in number_words) else 0.45
    elif any(k in q for k in ["where", "location", "located"]):
        score = 0.75 if any(w in a for w in LOCATION_WORDS) else 0.50
    elif any(k in q for k in ["color", "colour"]):
        score = 0.75 if any(w in a for w in COLOR_WORDS) else 0.45
    elif q.strip().startswith(("is ", "are ", "does ", "do ", "have ", "has ")):
        yn_words = {"yes", "no", "present", "absent", "visible", "identified", "observed", "not"}
        score = 0.72 if any(w in a for w in yn_words) else 0.55
    return float(np.clip(score, 0.0, 1.0))


def _infer_question_family(question: str, answer: str = "") -> str:
    q = str(question or "").lower()
    a = str(answer or "").lower()
    if _contains_any(q, ["color", "colour"]):
        return "color"
    if _contains_any(q, ["where", "location", "located"]):
        return "location"
    if _contains_any(q, ["how many", "number", "count"]):
        return "count"
    if _contains_any(q + " " + a, ["instrument", "tool", "forceps", "tube", "scope"]):
        return "instrument"
    if _contains_any(q, ["size", "large", "small", "diameter"]):
        return "size"
    if _contains_any(q + " " + a, ["polyp", "lesion", "abnormality"]):
        return "polyp_or_lesion_attribute"
    if _contains_any(q, ["text", "label", "caption", "writing", "letter", "word"]):
        return "visible_text"
    if _contains_any(q, ["finding", "present", "visible", "presence", "is there", "are there"]):
        return "finding_presence"
    return "generic"


def _build_probe_questions(question: str, answer: str, family: str, max_probes: int = 3) -> List[str]:
    banks: Dict[str, List[str]] = {
        "color": [
            "What is the dominant color of the relevant abnormal region?",
            "Is visible redness present in the mucosa?",
            "Are there pale, white, yellow, or dark regions supporting the answer?",
        ],
        "location": [
            "Where is the most relevant visual evidence located in the image?",
            "Is the finding central, left, right, upper, or lower?",
            "Is the evidence focal or diffuse?",
        ],
        "count": [
            "How many distinct relevant regions are visible?",
            "Is there one dominant region or multiple separated regions?",
            "Are any weak secondary regions visible?",
        ],
        "instrument": [
            "Is an instrument visible in the image?",
            "Where is the instrument-like structure located?",
            "What visual cues support instrument presence or absence?",
        ],
        "finding_presence": [
            "Is there a visible abnormal mucosal region?",
            "What visual feature supports presence or absence of the finding?",
            "Is the evidence focal or diffuse?",
            "Are there color or texture changes near the highlighted region?",
            "Could artifacts or poor visibility affect this judgment?",
        ],
        "polyp_or_lesion_attribute": [
            "What lesion-like morphology is visible?",
            "Is the lesion-like area raised, flat, focal, or diffuse?",
            "What color or texture cues support the answer?",
        ],
        "size": [
            "Does the finding appear small, moderate, or large relative to the frame?",
            "Is the relevant region focal or spread across the image?",
            "Where is the region used to judge size located?",
        ],
        "visible_text": [
            "Is visible text or an overlay label present in the image?",
            "Where is the visible text or label located in the image?",
            "Could the visible text be an overlay rather than an anatomical finding?",
        ],
        "generic": [
            "What visual evidence is most relevant to answering the question?",
            "Is the supporting evidence focal or diffuse?",
            "Could image artifacts affect the answer?",
        ],
    }
    probes = banks.get(family, banks["generic"])
    # For high-risk presence questions, allow more probes when requested.
    if family == "finding_presence":
        return probes[: max(3, min(max_probes, len(probes)))]
    return probes[: min(max_probes, len(probes))]


def _run_probe_answers(image: Image.Image, probe_questions: Sequence[str], max_new_tokens: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for probe in probe_questions:
        answer = task1.predict_one(image, probe, max_new_tokens=max_new_tokens)
        results.append({"question": probe, "answer": answer})
    return results


def _score_probe_consistency(answer: str, probe_answers: Sequence[Dict[str, str]], family: str) -> Dict[str, float | str]:
    main = str(answer or "").lower()
    probe_text = " ".join(str(item.get("answer", "")) for item in probe_answers).lower()
    if not probe_answers:
        return {
            "probe_support": 0.5,
            "probe_consistency": 0.5,
            "probe_conflict_penalty": 0.0,
            "probe_specificity": 0.5,
            "probe_summary": "no targeted probing was performed",
        }

    support = 0.5
    conflict = 0.0
    tokens = set(_tokenize(main))
    probe_tokens = set(_tokenize(probe_text))
    overlap = len(tokens & probe_tokens) / max(1, len(tokens))
    specificity = float(np.clip(0.35 + 0.04 * min(len(probe_tokens), 12), 0.35, 0.85))

    if family == "color":
        main_colors = COLOR_WORDS & tokens
        probe_colors = COLOR_WORDS & probe_tokens
        if main_colors:
            support = 0.85 if main_colors & probe_colors else 0.35
            conflict = 0.45 if probe_colors and not (main_colors & probe_colors) else 0.0
        else:
            support = 0.55 if probe_colors else 0.45
    elif family == "location":
        main_locs = LOCATION_WORDS & tokens
        probe_locs = LOCATION_WORDS & probe_tokens
        if main_locs:
            support = 0.85 if main_locs & probe_locs else 0.40
            conflict = 0.35 if probe_locs and not (main_locs & probe_locs) else 0.0
        else:
            support = 0.60 if probe_locs else 0.45
    elif family in {"finding_presence", "instrument"}:
        main_positive = any(term in main for term in PRESENCE_POSITIVE)
        main_negative = any(term in main for term in PRESENCE_NEGATIVE)
        probe_positive = any(term in probe_text for term in PRESENCE_POSITIVE)
        probe_negative = any(term in probe_text for term in PRESENCE_NEGATIVE)
        if main_positive and probe_positive:
            support = 0.78
        elif main_negative and probe_negative:
            support = 0.72
        elif (main_positive and probe_negative) or (main_negative and probe_positive):
            support = 0.30
            conflict = 0.65
        else:
            support = 0.52
            conflict = 0.15
    elif family == "count":
        main_has_number = any(ch.isdigit() for ch in main) or any(w in tokens for w in ["one", "two", "three", "single", "multiple", "no"])
        probe_has_number = any(ch.isdigit() for ch in probe_text) or any(w in probe_tokens for w in ["one", "two", "three", "single", "multiple", "no"])
        support = 0.72 if main_has_number and probe_has_number else 0.48
        conflict = 0.20 if main_has_number != probe_has_number else 0.0
    else:
        support = float(np.clip(0.45 + 0.45 * overlap, 0.35, 0.85))
        conflict = 0.15 if overlap < 0.10 else 0.0

    consistency = float(np.clip(0.55 * support + 0.45 * (1.0 - conflict), 0.0, 1.0))
    summary = _summarize_probe_answers(probe_answers, family)
    return {
        "probe_support": round(float(support), 4),
        "probe_consistency": round(float(consistency), 4),
        "probe_conflict_penalty": round(float(conflict), 4),
        "probe_specificity": round(float(specificity), 4),
        "probe_summary": summary,
    }


def _summarize_probe_answers(probe_answers: Sequence[Dict[str, str]], family: str) -> str:
    if not probe_answers:
        return "no additional probing information was available"
    answers = [str(item.get("answer", "")).strip() for item in probe_answers if str(item.get("answer", "")).strip()]
    if not answers:
        return "the follow-up probes did not produce specific additional details"
    clipped = [ans[:90].rstrip() for ans in answers[:3]]
    if family == "color":
        prefix = "color-focused probes indicated"
    elif family == "location":
        prefix = "location-focused probes indicated"
    elif family == "instrument":
        prefix = "instrument-focused probes indicated"
    elif family in {"finding_presence", "polyp_or_lesion_attribute"}:
        prefix = "finding-focused probes indicated"
    elif family == "count":
        prefix = "count-focused probes indicated"
    elif family == "visible_text":
        prefix = "visible-text probes indicated"
    else:
        prefix = "targeted probes indicated"
    return prefix + ": " + "; ".join(clipped)


def _extract_diagnostics(image: Image.Image, question: str, max_new_tokens: int) -> Dict[str, object]:
    model = task1.load_model()
    device = task1._DEVICE or task1._select_device()
    image_tensor = task1._pil_to_tensor(image).to(device)
    structural = {key: value.to(device) for key, value in task1._extract_online_structural_tensors(image).items()}

    with torch.inference_mode():
        prediction = model.generate(
            image=image_tensor,
            prior_mask=structural["prior_mask"],
            topo_features=structural["topo_features"],
            global_features=structural["global_features"],
            question_text=[question],
            max_new_tokens=max_new_tokens,
        )[0]
        answer = task1.normalize_prediction(prediction, question)

        forward_out = model.forward(
            image=image_tensor,
            prior_mask=structural["prior_mask"],
            topo_features=structural["topo_features"],
            global_features=structural["global_features"],
            question_text=[question],
            answer_text=None,
            return_diagnostics=True,
        )

    return {
        "answer": answer,
        "structural": structural,
        "forward_out": forward_out,
    }


def _build_evidence(
    image: Image.Image,
    diagnostics: Dict[str, object],
    answer: str,
    question: str,
    probe_info: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    structural = diagnostics["structural"]
    forward_out = diagnostics["forward_out"]
    probe_info = probe_info or {}

    prior_mask = _tensor_to_numpy(structural["prior_mask"])[0]
    topo_features = _tensor_to_numpy(structural["topo_features"])[0]
    global_features = _tensor_to_numpy(structural["global_features"])[0]

    morph_score = topo_features[..., 0]
    tissue_ratio = topo_features[..., 8]
    specular_ratio_map = topo_features[..., 7]
    topo_mask = normalize_map(morph_score)

    ot_patch_attention = np.zeros_like(prior_mask, dtype=np.float32)
    ot_prior_agreement = 0.5
    ot_cost = None
    if isinstance(forward_out, dict) and forward_out.get("ot_out") is not None:
        ot_out = forward_out["ot_out"]
        plan = _tensor_to_numpy(ot_out["transport_plan"])[0]
        # Expected transport layout follows compute_prior_alignment_loss: [text, visual], skip global token.
        patch_transport = plan[:, 1:]
        patch_attention = patch_transport.sum(axis=0)
        if patch_attention.size == prior_mask.size:
            ot_patch_attention = normalize_map(patch_attention.reshape(prior_mask.shape))
            ot_prior_agreement = _cosine_map_agreement(ot_patch_attention, prior_mask)
        if "ot_cost" in ot_out:
            ot_cost = float(_tensor_to_numpy(ot_out["ot_cost"]).reshape(-1)[0])

    evidence_map = normalize_map(
        0.35 * normalize_map(prior_mask)
        + 0.30 * normalize_map(topo_mask)
        + 0.30 * normalize_map(ot_patch_attention)
        + 0.05 * normalize_map(tissue_ratio)
    )
    evidence_map = normalize_map(evidence_map * (1.0 - 0.35 * normalize_map(specular_ratio_map)))

    concentration = _concentration(evidence_map)
    prior_topo_agreement = _cosine_map_agreement(prior_mask, topo_mask)
    specular_burden = float(np.mean(np.clip(specular_ratio_map, 0.0, 1.0)))
    tissue_coverage = float(np.mean(np.clip(tissue_ratio, 0.0, 1.0)))
    high_ot_cost_penalty = 0.35 if ot_cost is None else float(np.clip(ot_cost / 10.0, 0.0, 1.0))

    probe_scores = probe_info.get("scores", {}) if isinstance(probe_info, dict) else {}
    gate_evidence = {
        "prior_topo_agreement": prior_topo_agreement,
        "ot_prior_agreement": ot_prior_agreement,
        "evidence_concentration": concentration,
        "tissue_coverage": tissue_coverage,
        "specular_penalty": specular_burden,
        "high_ot_cost_penalty": high_ot_cost_penalty,
        "answer_specificity": _answer_specificity(answer),
        "answer_question_compatibility": _question_answer_compatibility(question, answer),
        "generic_answer_penalty": _generic_answer_penalty(answer),
        "diffuse_evidence_penalty": 1.0 - concentration,
        "probe_support": float(probe_scores.get("probe_support", 0.5)),
        "probe_consistency": float(probe_scores.get("probe_consistency", 0.5)),
        "probe_conflict_penalty": float(probe_scores.get("probe_conflict_penalty", 0.0)),
        "probe_specificity": float(probe_scores.get("probe_specificity", 0.5)),
    }
    gate = BayesianExplanationGate().score(gate_evidence)

    evidence = {
        "prior_topo_agreement": round(prior_topo_agreement, 4),
        "ot_prior_agreement": round(ot_prior_agreement, 4),
        "evidence_concentration": round(concentration, 4),
        "tissue_coverage": round(tissue_coverage, 4),
        "specular_burden": round(specular_burden, 4),
        "ot_cost": None if ot_cost is None else round(float(ot_cost), 4),
        "global_features": [round(float(x), 5) for x in global_features.tolist()],
        "location": quadrant_from_heatmap(evidence_map),
        "question_family": probe_info.get("family", "generic") if isinstance(probe_info, dict) else "generic",
        "probe_questions": probe_info.get("questions", []) if isinstance(probe_info, dict) else [],
        "probe_answers": probe_info.get("answers", []) if isinstance(probe_info, dict) else [],
        "probe_scores": probe_scores,
        "bayes_gate": gate,
        "gate_inputs": {key: round(float(value), 4) for key, value in gate_evidence.items()},
    }
    return evidence_map, evidence


def _visual_cues_from_evidence(evidence: Dict[str, object], answer: str) -> str:
    gate_inputs = evidence.get("gate_inputs", {})
    specular = float(gate_inputs.get("specular_penalty", 0.0)) if isinstance(gate_inputs, dict) else 0.0
    concentration = float(evidence.get("evidence_concentration", 0.0))
    cues: List[str] = []
    ans = answer.lower()
    if any(word in ans for word in ["red", "pink", "erythema", "inflammation", "ulcerative"]):
        cues.append("color change/redness")
    if any(word in ans for word in ["polyp", "lesion", "abnormal", "finding"]):
        cues.append("lesion-like mucosal morphology")
    if any(word in ans for word in ["instrument", "tube", "forceps"]):
        cues.append("linear high-contrast structures")
    if concentration >= 0.45:
        cues.append("focal structural evidence")
    else:
        cues.append("more diffuse structural evidence")
    if specular > 0.30:
        cues.append("possible reflective artifact regions")
    return ", ".join(cues)


def _make_textual_explanation(question: str, answer: str, evidence: Dict[str, object]) -> str:
    gate = evidence["bayes_gate"]
    bucket = str(gate["confidence_bucket"])
    location = str(evidence.get("location", "central field"))
    cues = _visual_cues_from_evidence(evidence, answer)
    family = str(evidence.get("question_family", "generic"))
    probe_scores = evidence.get("probe_scores", {}) if isinstance(evidence.get("probe_scores", {}), dict) else {}
    probe_summary = str(probe_scores.get("probe_summary", "targeted probing did not add specific details"))
    probe_consistency = float(probe_scores.get("probe_consistency", 0.5)) if probe_scores else 0.5
    consistency_phrase = "consistent" if probe_consistency >= 0.65 else "partially consistent" if probe_consistency >= 0.45 else "not fully consistent"
    confidence_phrase = {
        "high": "The visual and probing support is strong",
        "moderate": "The visual and probing support is moderate",
        "low": "The visual and probing support is limited",
    }.get(bucket, "The visual and probing support is moderate")

    caution_reasons = gate.get("caution_reasons", []) if isinstance(gate, dict) else []
    high_risk_presence = family in {"finding_presence", "instrument"} and bucket != "high"
    if high_risk_presence and "presence/absence questions require cautious interpretation" not in caution_reasons:
        caution_reasons = list(caution_reasons) + ["presence/absence questions require cautious interpretation"]
    if caution_reasons:
        caution = " Caution is warranted because " + "; ".join(str(x) for x in caution_reasons[:3]) + "."
    else:
        caution = " The highlighted evidence is consistent with the model's response, but clinical review remains necessary."

    return (
        f'The model predicts "{answer}". Supporting evidence is concentrated in the {location} '
        f"of the endoscopic frame, where the heatmap highlights {cues}. "
        f"Targeted self-probing for this {family.replace('_', ' ')} question found that {probe_summary}. "
        f"These follow-up answers are {consistency_phrase} with the primary answer. "
        f"{confidence_phrase} based on lesion-prior alignment, morphology/topology features, OT attention, "
        f"and probe consistency. The original question was: {question.strip()}"
        f"{caution}"
    )


def _existing_rows(output_jsonl: Path) -> Dict[int, Dict[str, object]]:
    if not output_jsonl.exists():
        return {}
    rows: Dict[int, Dict[str, object]] = {}
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[int(row["val_id"])] = row
            except Exception:
                continue
    return rows


def generate_task2(
    max_samples: int | None,
    max_new_tokens: int,
    output_dir: Path,
    max_probes: int = 3,
    probe_max_new_tokens: int = 24,
    resume: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "submission_task2.jsonl"

    dataset = _load_task2_subset(max_samples=max_samples)
    task1.load_model()

    existing = _existing_rows(output_jsonl) if resume else {}
    mode = "a" if existing else "w"
    if existing:
        print(f"Resuming Task 2 generation with {len(existing)} existing rows")

    with output_jsonl.open(mode, encoding="utf-8") as f:
        for sample in tqdm(dataset, desc="Generating Task 2", unit="samples"):
            val_id = int(sample["val_id"])
            if val_id in existing:
                continue
            image = sample["image"]
            question = str(sample["question"])
            img_id = str(sample["img_id"])

            diagnostics = _extract_diagnostics(image, question, max_new_tokens=max_new_tokens)
            answer = str(diagnostics["answer"])
            family = _infer_question_family(question, answer)
            effective_max_probes = max(max_probes, 5) if family == "finding_presence" and max_probes >= 5 else max_probes
            probe_questions = _build_probe_questions(question, answer, family, max_probes=effective_max_probes)
            probe_answers = _run_probe_answers(image, probe_questions, max_new_tokens=probe_max_new_tokens)
            probe_scores = _score_probe_consistency(answer, probe_answers, family)
            probe_info = {
                "family": family,
                "questions": probe_questions,
                "answers": probe_answers,
                "scores": probe_scores,
            }

            evidence_map, evidence = _build_evidence(
                image,
                diagnostics,
                answer=answer,
                question=question,
                probe_info=probe_info,
            )

            heatmap_rel = f"visuals/{val_id:04d}_heatmap.png"
            heatmap_path = output_dir / heatmap_rel
            overlay_heatmap(image, evidence_map, heatmap_path)

            evidence_rel = f"visuals/{val_id:04d}_evidence.json"
            with (output_dir / evidence_rel).open("w", encoding="utf-8") as ef:
                json.dump(evidence, ef, ensure_ascii=False, indent=2)

            explanation = _make_textual_explanation(question, answer, evidence)
            gate = evidence["bayes_gate"]
            row = {
                "val_id": str(val_id),
                "img_id": img_id,
                "question": question,
                "answer": answer,
                "textual_explanation": explanation,
                "visual_explanation": [
                    {
                        "type": "heatmap",
                        "data": heatmap_rel,
                        "description": "Heatmap combining lesion prior, morphology/topology evidence, and model OT attention.",
                    }
                ],
                "confidence_score": float(gate["confidence_score"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return output_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate Task 2 explanation submission.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--probe-max-new-tokens", type=int, default=24)
    parser.add_argument("--max-probes", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=TASK2_DIR)
    args = parser.parse_args()
    path = generate_task2(
        args.max_samples,
        args.max_new_tokens,
        args.output_dir,
        max_probes=args.max_probes,
        probe_max_new_tokens=args.probe_max_new_tokens,
        resume=not args.no_resume,
    )
    print(f"Task 2 JSONL written to {path}")


if __name__ == "__main__":
    main()
