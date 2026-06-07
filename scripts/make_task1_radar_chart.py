"""Create a publication-ready radar chart for Task 1 question-attribute performance.

The chart compares three prediction files across keyword-derived question groups.
It writes vector PDF and high-resolution PNG assets for direct LaTeX inclusion.

Default metric is a lightweight ROUGE-L-style token F1 implemented locally so the
script can run without downloading external metric packages. Use --metric meteor
only when the HuggingFace evaluate package is available in the environment.

Example:
    python scripts/make_task1_radar_chart.py \
      --baseline outputs/eval/a0_clean_image_only_30k_e2_fulltest/predictions.jsonl \
      --cata-epoch5 outputs/eval/cata_d1_full_epoch5_modelonly_train_fulltest/predictions.jsonl \
      --cata-test-adapt outputs/eval/cata_d1_full_epoch5_modelonly_train_testadapt1_fulltest/predictions.jsonl \
      --output-dir paper_assets/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    keywords: Tuple[str, ...]
    require_any: Tuple[str, ...] = ()


CATEGORIES: Tuple[Category, ...] = (
    Category(
        key="abnormality_location",
        label="Abnormality\nLocation",
        keywords=("where", "located", "location", "region", "quadrant", "scattered", "position"),
        require_any=("abnormal", "abnormality", "lesion", "finding", "polyp", "inflammation"),
    ),
    Category(
        key="polyp_size",
        label="Polyp\nSize",
        keywords=("size", "dimension", "diameter", "millimeter", "millimetre", " mm", "larger", "greater than"),
        require_any=("polyp", "lesion"),
    ),
    Category(
        key="instrument_location",
        label="Instrument\nLocation",
        keywords=("where", "located", "location", "region", "quadrant", "position", "central", "upper", "lower"),
        require_any=("instrument", "device", "tube", "forceps"),
    ),
    Category(
        key="finding_count",
        label="Finding\nCount",
        keywords=("how many", "number of", "count", "single", "multiple"),
        require_any=("finding", "findings", "polyp", "polyps", "instrument", "instruments", "abnormalit"),
    ),
    Category(
        key="finding_presence",
        label="Finding\nPresence",
        keywords=("is there", "are there", "visible", "observed", "present", "identified", "detected", "evidence"),
        require_any=("abnormal", "finding", "polyp", "instrument", "text", "landmark", "artifact", "artefact"),
    ),
    Category(
        key="landmark_color",
        label="Landmark /\nLesion Color",
        keywords=("color", "colour", "pink", "red", "white", "yellow", "grey", "gray", "flesh"),
        require_any=(),
    ),
)

RUNS: Tuple[Tuple[str, str, str, str, str, float], ...] = (
    # Okabe-Ito inspired palette: readable in color, grayscale, and print.
    # (key, legend label, color, line style, marker, fill alpha)
    ("baseline", "ViT + Qwen (Baseline)", "#1f2937", "--", "s", 0.00),
    ("cata_epoch5", "CATA Epoch 3", "#0072B2", "-.", "D", 0.08),
    ("cata_test_adapt", "CATA Test-Adapt", "#D55E00", "-", "o", 0.16),
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("<image>", " ")).strip()


def tokenize(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(value).lower())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["__line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def row_prediction(row: Dict[str, Any]) -> str:
    for key in ("prediction", "pred", "answer", "generated_answer", "response"):
        if row.get(key) not in (None, ""):
            return normalize_text(row[key])
    return ""


def row_reference(row: Dict[str, Any]) -> str:
    for key in ("ground_truth", "reference", "gt", "target", "answer_gt"):
        if row.get(key) not in (None, ""):
            return normalize_text(row[key])
    return ""


def category_matches(question: str, category: Category) -> bool:
    q = f" {question.lower()} "
    has_keyword = any(keyword in q for keyword in category.keywords)
    if not has_keyword:
        return False
    if category.require_any:
        return any(token in q for token in category.require_any)
    return True


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1_single(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def mean_rouge_l_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    scores = [rouge_l_f1_single(pred, ref) for pred, ref in zip(predictions, references)]
    return float(sum(scores) / len(scores))


def meteor_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    try:
        from evaluate import load as evaluate_load
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("--metric meteor requires `pip install evaluate nltk`." ) from exc
    meteor = evaluate_load("meteor")
    refs_nested = [[ref] for ref in references]
    result = meteor.compute(predictions=list(predictions), references=refs_nested)
    return float(result["meteor"])


def score_group(predictions: Sequence[str], references: Sequence[str], metric: str) -> float:
    if not predictions:
        return 0.0
    if metric == "rouge_l":
        return mean_rouge_l_f1(predictions, references)
    if metric == "meteor":
        return meteor_score(predictions, references)
    raise ValueError(f"Unsupported metric: {metric}")


def collect_scores(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORIES:
        preds: List[str] = []
        refs: List[str] = []
        for row in rows:
            question = normalize_text(row.get("question", row.get("question_text", "")))
            if not category_matches(question, category):
                continue
            pred = row_prediction(row)
            ref = row_reference(row)
            if pred and ref:
                preds.append(pred)
                refs.append(ref)
        output[category.key] = {
            "label": category.label.replace("\n", " "),
            "num_examples": len(preds),
            "score": round(score_group(preds, refs, metric), 6),
        }
    return output


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def close_loop(values: Sequence[float]) -> List[float]:
    vals = list(values)
    return vals + vals[:1]


def make_radar_chart(all_scores: Dict[str, Dict[str, Dict[str, Any]]], metric: str, output_dir: Path) -> None:
    configure_matplotlib()

    categories = [cat.label for cat in CATEGORIES]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.0), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")

    # Place the spatial/morphological axes on the left hemisphere.
    ax.set_theta_offset(2 * np.pi / 3)
    ax.set_theta_direction(1)

    for run_key, run_label, color, linestyle, marker, fill_alpha in RUNS:
        values = [float(all_scores[run_key][cat.key]["score"]) for cat in CATEGORIES]
        closed_values = close_loop(values)
        linewidth = 3.0 if run_key == "cata_test_adapt" else 2.4
        zorder = 4 if run_key == "cata_test_adapt" else 3
        ax.plot(
            angles,
            closed_values,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=run_label,
            color=color,
            zorder=zorder,
        )
        if fill_alpha > 0:
            ax.fill(angles, closed_values, alpha=fill_alpha, color=color, zorder=zorder - 1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="x", pad=12)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#6b7280", fontsize=9.5)
    ax.set_rlabel_position(82)

    ax.grid(color="#d1d5db", linestyle="-", linewidth=0.55)
    ax.spines["polar"].set_color("#d1d5db")
    ax.spines["polar"].set_linewidth(0.8)

    metric_label = "METEOR" if metric == "meteor" else "ROUGE-L style F1"
    ax.set_title(f"Task 1 Attribute-wise Performance ({metric_label})", y=1.11, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=False, fontsize=10.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"radar_chart_task1.{ext}"
        fig.savefig(path, format=ext, dpi=320, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def write_score_tables(all_scores: Dict[str, Dict[str, Dict[str, Any]]], metric: str, output_dir: Path) -> None:
    json_path = output_dir / "radar_chart_task1_scores.json"
    json_payload = {
        "metric": metric,
        "categories": [cat.key for cat in CATEGORIES],
        "runs": all_scores,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")

    csv_path = output_dir / "radar_chart_task1_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "category", "num_examples", "score", "metric"])
        for run_key, _, _, _, _, _ in RUNS:
            for category in CATEGORIES:
                item = all_scores[run_key][category.key]
                writer.writerow([run_key, category.key, item["num_examples"], item["score"], metric])
    print(f"Saved {csv_path}")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Task 1 attribute-wise radar chart from prediction JSONL files.")
    parser.add_argument("--baseline", default="outputs/eval/a0_clean_image_only_30k_e2_fulltest/predictions.jsonl")
    parser.add_argument("--cata-epoch5", default="outputs/eval/cata_d1_full_epoch5_modelonly_train_fulltest/predictions.jsonl")
    parser.add_argument("--cata-test-adapt", default="outputs/eval/cata_d1_full_epoch5_modelonly_train_testadapt1_fulltest/predictions.jsonl")
    parser.add_argument("--output-dir", default="paper_assets/figures")
    parser.add_argument("--metric", choices=("rouge_l", "meteor"), default="rouge_l")
    args = parser.parse_args()

    paths = {
        "baseline": resolve_path(args.baseline),
        "cata_epoch5": resolve_path(args.cata_epoch5),
        "cata_test_adapt": resolve_path(args.cata_test_adapt),
    }
    output_dir = resolve_path(args.output_dir)

    all_scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run_key, path in paths.items():
        print(f"Loading {run_key}: {path}")
        rows = load_jsonl(path)
        all_scores[run_key] = collect_scores(rows, args.metric)

    make_radar_chart(all_scores, args.metric, output_dir)
    write_score_tables(all_scores, args.metric, output_dir)


if __name__ == "__main__":
    main()
