"""Fast Task 2 generation from cached full-test Task 1 predictions.

This script is a lightweight alternative to generate_task2_cata_final.py when you
already have an eval/predictions.jsonl file for the full test split.

It does NOT load the Qwen/CATA checkpoint and does NOT run self-probing. Instead
it:
- loads the official Subtask 2 public subset in the same order as
  generate_task2_cata_final.py,
- maps each example to a cached prediction by image/question, record_index, or
  unique question,
- regenerates D1-style morphology/TDA + color/edge heatmaps and evidence JSON,
- writes Task 2 JSONL rows with deterministic textual explanations and
  reliability-style confidence scores.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import Image as HfImage
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name == "hf_submission" else SCRIPT_DIR

for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

QUESTION_FAMILIES = {
    "visible_text": ["text", "label", "caption", "writing"],
    "instrument": ["instrument", "surgical", "tool", "forceps", "tube", "catheter", "foreign"],
    "count": ["how many", "number of", "count"],
    "color": ["color", "colours", "colors", "red", "white", "pink", "black", "green"],
    "location": ["where", "which region", "location", "located", "position", "region"],
    "procedure": ["procedure", "obtained", "depicted"],
    "size": ["size", "large", "small", "millimeter", "mm"],
    "lesion": ["polyp", "lesion", "abnormal", "finding", "mucosa", "tissue"],
}

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = (14, 14)
_STRUCTURAL_EXTRACTORS = None


def question_family(question: str) -> str:
    q = question.lower()
    for family, terms in QUESTION_FAMILIES.items():
        if any(term in q for term in terms):
            return family
    return "generic"


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def get_structural_extractors():
    global _STRUCTURAL_EXTRACTORS
    if _STRUCTURAL_EXTRACTORS is not None:
        return _STRUCTURAL_EXTRACTORS
    from src.topology.lesion_prior import LesionPriorExtractor
    from src.topology.tda_morphology import TopologicalExtractor

    prior = LesionPriorExtractor(image_size=DEFAULT_IMAGE_SIZE, grid_size=DEFAULT_GRID_SIZE, use_morphology=False)
    morph = TopologicalExtractor(grid_size=DEFAULT_GRID_SIZE, image_size=DEFAULT_IMAGE_SIZE)
    _STRUCTURAL_EXTRACTORS = (prior, morph)
    return _STRUCTURAL_EXTRACTORS


def extract_structural(image) -> Dict[str, np.ndarray]:
    import cv2

    prior_extractor, morpho_extractor = get_structural_extractors()
    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    prior_out = prior_extractor.extract_prior(bgr)
    morph_out = morpho_extractor.extract_features(bgr)
    return {
        "prior_mask": prior_out["prior_mask"].astype(np.float32),
        "topo_features": morph_out["topo_features"].astype(np.float32),
        "global_features": morph_out["global_features"].astype(np.float32),
    }


def create_heatmap(image, structural: Dict[str, np.ndarray], out_png: Path, out_json: Path) -> Dict[str, Any]:
    import cv2
    from PIL import Image

    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    small = cv2.resize(rgb, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    red_like = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)).astype(np.float32) * saturation
    specular = ((value > 0.86) & (saturation < 0.35)).astype(np.float32)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0

    prior = structural["prior_mask"]
    topo = structural["topo_features"]
    topo_sal = normalize_map(np.linalg.norm(topo, axis=-1))
    topo_up = cv2.resize(topo_sal, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
    color_sal = normalize_map(0.65 * red_like + 0.35 * saturation)
    edge_sal = normalize_map(edges)
    artifact_penalty = 1.0 - 0.55 * cv2.GaussianBlur(specular, (0, 0), 2.0)
    heat = normalize_map((0.48 * topo_up + 0.32 * color_sal + 0.20 * edge_sal) * artifact_penalty)

    heat_uint8 = np.uint8(np.clip(heat * 255.0, 0, 255))
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.58 * small + 0.42 * color)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_png, optimize=True)

    evidence = {
        "prior_mean": round(float(prior.mean()), 6),
        "prior_max": round(float(prior.max()), 6),
        "topo_saliency_mean": round(float(topo_sal.mean()), 6),
        "topo_saliency_max": round(float(topo_sal.max()), 6),
        "heatmap_mean": round(float(heat.mean()), 6),
        "heatmap_max": round(float(heat.max()), 6),
        "heatmap_concentration": round(float((heat > 0.65).mean()), 6),
        "redness_mean": round(float(red_like.mean()), 6),
        "saturation_mean": round(float(saturation.mean()), 6),
        "specular_fraction": round(float(specular.mean()), 6),
        "global_features": [round(float(x), 6) for x in structural["global_features"].tolist()],
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    return evidence

DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "cata_d1_full_epoch5_modelonly_train_testadapt1_fulltest"
    / "eval"
    / "predictions.jsonl"
)

NEGATIVE_TERMS = (
    "no ",
    "not ",
    "none",
    "without",
    "absent",
    "no evidence",
    "not identified",
    "not observed",
)


def path_basename_any(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return Path(PureWindowsPath(text).name).name


def image_stem_any(value: Any) -> str:
    name = path_basename_any(value)
    return Path(name).stem.lower()


def normalize_question(text: Any) -> str:
    text = str(text or "").replace("<image>", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def clean_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def is_negative_answer(text: str) -> bool:
    t = " " + text.lower().strip() + " "
    return any(term in t for term in NEGATIVE_TERMS)


def answer_specificity(answer: str) -> float:
    words = [w for w in re.split(r"\W+", answer.lower()) if w]
    if not words:
        return 0.0
    # Reward concise but non-empty medical answers over extremely generic text.
    return float(np.clip(len(set(words)) / 8.0, 0.0, 1.0))


def candidate_image_stems(row: Dict[str, Any]) -> List[str]:
    stems: List[str] = []
    for key in ("img_id", "image_id", "image_ref", "image_path", "image", "file_name", "filename"):
        value = row.get(key)
        if value is None:
            continue
        stem = image_stem_any(value)
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def row_answer(row: Dict[str, Any]) -> str:
    for key in ("prediction", "answer", "pred", "generated_answer", "response"):
        if key in row and row[key] is not None:
            return clean_answer(row[key])
    return ""


def read_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            answer = row_answer(row)
            if not answer:
                continue
            row["__line_number"] = line_number
            row["__answer"] = answer
            row["__norm_question"] = normalize_question(row.get("question", ""))
            row["__image_stems"] = candidate_image_stems(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable prediction rows found in: {path}")
    return rows


class PredictionIndex:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        self.by_image_question: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_record_index: Dict[int, Dict[str, Any]] = {}

        for row in rows:
            q = row.get("__norm_question", "")
            if q:
                self.by_question[q].append(row)
            for stem in row.get("__image_stems", []):
                if stem and q:
                    self.by_image_question[(stem, q)].append(row)
            try:
                record_index = int(row.get("record_index"))
            except Exception:
                record_index = None
            if record_index is not None:
                self.by_record_index.setdefault(record_index, row)

    def find(self, example: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        q = normalize_question(example.get("question", ""))
        stems = []
        for key in ("img_id", "image_id", "image", "file_name", "filename"):
            stem = image_stem_any(example.get(key))
            if stem and stem not in stems:
                stems.append(stem)

        for stem in stems:
            matches = self.by_image_question.get((stem, q), [])
            if len(matches) == 1:
                return matches[0], "image_question"
            if len(matches) > 1:
                return matches[0], "image_question_duplicate_first"

        # If the eval predictions came from a local JSONL in HF test order,
        # record_index is commonly 1-based while __hf_original_index is 0-based.
        original_index = example.get("__hf_original_index")
        try:
            original_index = int(original_index)
        except Exception:
            original_index = None
        if original_index is not None:
            for candidate in (original_index, original_index + 1):
                row = self.by_record_index.get(candidate)
                if row is not None:
                    row_q = row.get("__norm_question", "")
                    if not row_q or row_q == q:
                        return row, f"record_index_{candidate}"

        q_matches = self.by_question.get(q, [])
        if len(q_matches) == 1:
            return q_matches[0], "question_unique"
        if len(q_matches) > 1:
            # Last-resort duplicate handling: prefer matching image stem if any
            # prediction row has a compatible stem hidden in one of its fields.
            stem_set = set(stems)
            for row in q_matches:
                if stem_set.intersection(set(row.get("__image_stems", []))):
                    return row, "question_duplicate_image_resolved"
            return q_matches[0], "question_duplicate_first"

        return None, "missing"


def build_task2_set(dataset_name: str, subset_size: int, shuffle_seed: int, cast_image: bool = True):
    ds = load_dataset(dataset_name)["test"]
    ds = ds.add_column("__hf_original_index", list(range(len(ds))))
    if "complexity" in ds.column_names:
        ds = ds.filter(lambda x: x["complexity"] == 1)
    if subset_size > len(ds):
        raise ValueError(f"Requested {subset_size} rows but filtered Task 2 split has only {len(ds)}")
    ds = ds.shuffle(seed=shuffle_seed).select(range(subset_size))
    ds = ds.add_column("val_id", list(range(subset_size)))
    return ds.cast_column("image", HfImage()) if cast_image and "image" in ds.column_names else ds


def safe_relpath(path: Path, root: Path = SCRIPT_DIR) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_cached_explanation(question: str, answer: str, family: str, evidence: Dict[str, Any]) -> Tuple[str, float]:
    concentration = float(evidence.get("heatmap_concentration", 0.0))
    topo = float(evidence.get("topo_saliency_mean", 0.0))
    color_support = max(
        float(evidence.get("redness_mean", 0.0)),
        float(evidence.get("saturation_mean", 0.0)),
    )
    spec = float(evidence.get("specular_fraction", 0.0))
    specificity = answer_specificity(answer)

    confidence = (
        0.50
        + 0.24 * min(1.0, topo * 2.5)
        + 0.14 * min(1.0, concentration * 8.0)
        + 0.08 * min(1.0, color_support * 2.0)
        + 0.04 * specificity
    )
    if is_negative_answer(answer):
        confidence -= 0.03
    confidence -= 0.10 * min(1.0, spec * 5.0)
    confidence = float(np.clip(confidence, 0.05, 0.95))

    evidence_bits: List[str] = []
    if topo > 0.12:
        evidence_bits.append("morphology/TDA structural support")
    if color_support > 0.10:
        evidence_bits.append("color and mucosal-texture support")
    if concentration > 0.03:
        evidence_bits.append("a relatively focal heatmap region")
    else:
        evidence_bits.append("diffuse heatmap evidence")
    if spec > 0.03:
        evidence_bits.append("possible specular artifact burden")

    support_phrase = "moderate" if confidence >= 0.55 else "limited"
    explanation = (
        f"The CATA-Final / D1 model predicts \"{answer}\". "
        f"For this {family} question, the generated visual explanation highlights "
        f"{', '.join(evidence_bits)}. "
        f"Overall support is {support_phrase}; the reliability score combines "
        f"morphology/TDA saliency, heatmap concentration, color/texture support, "
        f"artifact burden, and answer specificity. "
        f"The original question was: {question}. "
        "This explanation is intended for clinician review and should be interpreted "
        "alongside the image rather than as a standalone diagnosis."
    )
    return explanation, round(confidence, 4)


def iter_indices(total: int, start_index: int, limit: int) -> Iterable[int]:
    end = total if limit <= 0 else min(total, start_index + limit)
    return range(start_index, end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 2 JSONL from cached full-test predictions.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="Path to eval/predictions.jsonl")
    parser.add_argument("--output-jsonl", default="submission_task2_cata_final_from_predictions.jsonl")
    parser.add_argument("--visual-dir", default="visuals_from_predictions")
    parser.add_argument("--dataset-name", default="SimulaMet/Kvasir-VQA-x1")
    parser.add_argument("--subset-size", type=int, default=1500)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--overwrite-visuals",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Regenerate heatmap/evidence even if files already exist. Default: false.",
    )
    parser.add_argument(
        "--skip-visuals",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Do not generate heatmap/evidence files; use lightweight placeholder evidence. Useful for fast mapping smoke tests.",
    )
    parser.add_argument(
        "--allow-missing",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Write empty answers for unmatched rows instead of failing.",
    )
    parser.add_argument(
        "--include-debug",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Include match method and source prediction row in debug metadata.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = PROJECT_ROOT / predictions_path
    output_jsonl = Path(args.output_jsonl)
    if not output_jsonl.is_absolute():
        output_jsonl = SCRIPT_DIR / output_jsonl
    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = SCRIPT_DIR / visual_dir
    visual_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached predictions: {predictions_path}", flush=True)
    pred_rows = read_predictions(predictions_path)
    pred_index = PredictionIndex(pred_rows)
    print(f"Loaded {len(pred_rows)} usable cached prediction rows", flush=True)

    print(f"Loading Task 2 subset: {args.dataset_name}", flush=True)
    task2 = build_task2_set(args.dataset_name, args.subset_size, args.shuffle_seed, cast_image=not args.skip_visuals)
    selected_indices = list(iter_indices(len(task2), args.start_index, args.limit))

    mode = "a" if args.start_index > 0 and output_jsonl.exists() else "w"
    match_counts: Counter[str] = Counter()
    missing_examples: List[Dict[str, Any]] = []
    started = time.time()

    with output_jsonl.open(mode, encoding="utf-8") as f:
        for idx in tqdm(selected_indices, desc="Task 2 from cached predictions"):
            example = task2[idx]
            pred_row, match_method = pred_index.find(example)
            match_counts[match_method] += 1
            if pred_row is None:
                missing_examples.append(
                    {
                        "val_id": int(example["val_id"]),
                        "img_id": str(example.get("img_id", "")),
                        "question": str(example.get("question", "")),
                        "hf_original_index": int(example.get("__hf_original_index", -1)),
                    }
                )
                if not args.allow_missing:
                    continue
                answer = ""
            else:
                answer = str(pred_row["__answer"])

            question = str(example["question"])
            family = question_family(question)
            stem = f"{int(example['val_id']):04d}"
            heatmap_path = visual_dir / f"{stem}_heatmap.png"
            evidence_path = visual_dir / f"{stem}_evidence.json"

            if args.skip_visuals:
                evidence = {
                    "prior_mean": 0.0,
                    "prior_max": 0.0,
                    "topo_saliency_mean": 0.0,
                    "topo_saliency_max": 0.0,
                    "heatmap_mean": 0.0,
                    "heatmap_max": 0.0,
                    "heatmap_concentration": 0.0,
                    "redness_mean": 0.0,
                    "saturation_mean": 0.0,
                    "specular_fraction": 0.0,
                    "global_features": [],
                }
            elif args.overwrite_visuals or not heatmap_path.exists() or not evidence_path.exists():
                img = example["image"]
                structural = extract_structural(img)
                evidence = create_heatmap(img, structural, heatmap_path, evidence_path)
            else:
                with evidence_path.open("r", encoding="utf-8") as ef:
                    evidence = json.load(ef)

            explanation, confidence = build_cached_explanation(question, answer, family, evidence)
            row: Dict[str, Any] = {
                "val_id": str(int(example["val_id"])),
                "img_id": str(example.get("img_id", "")),
                "question": question,
                "answer": answer,
                "textual_explanation": explanation,
                "visual_explanation": [] if args.skip_visuals else [
                    {
                        "type": "heatmap",
                        "data": safe_relpath(heatmap_path),
                        "description": "Cached-prediction CATA-Final / D1 heatmap combining morphology/TDA saliency, color/edge evidence, and artifact suppression.",
                    },
                    {
                        "type": "evidence_json",
                        "data": safe_relpath(evidence_path),
                        "description": "Structured visual evidence and confidence inputs used by the cached-prediction explanation generator.",
                    },
                ],
                "confidence_score": confidence,
            }
            if args.include_debug:
                row["debug"] = {
                    "generator": "generate_task2_from_predictions.py",
                    "match_method": match_method,
                    "prediction_line_number": None if pred_row is None else pred_row.get("__line_number"),
                    "prediction_record_index": None if pred_row is None else pred_row.get("record_index"),
                    "hf_original_index": int(example.get("__hf_original_index", -1)),
                    "question_family": family,
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "predictions_path": str(predictions_path),
        "output_jsonl": str(output_jsonl),
        "visual_dir": str(visual_dir),
        "num_prediction_rows": len(pred_rows),
        "num_requested_rows": len(selected_indices),
        "num_missing": len(missing_examples),
        "match_counts": dict(match_counts),
        "missing_examples": missing_examples[:20],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"✅ Wrote Task 2 rows: {output_jsonl}")
    print(f"✅ Wrote visuals/evidence: {visual_dir}")
    print(f"✅ Wrote match summary: {summary_path}")
    print(f"Match counts: {dict(match_counts)}")
    if missing_examples and not args.allow_missing:
        raise RuntimeError(
            f"Missing {len(missing_examples)} / {len(selected_indices)} Task 2 rows. "
            f"See {summary_path} for examples. Re-run with --allow-missing true only if intentional."
        )


if __name__ == "__main__":
    main()
