"""Build Task 1 metric breakdown by MediaEval complexity level.

This script reads a JSON config of public model names and prediction files, maps
predictions to the Kvasir-VQA-x1 test split, computes metrics per complexity
level, and writes CSV/JSON/LaTeX outputs for the CATA paper.

The script intentionally emits only public model names from the config into paper
outputs.

Example:
    python scripts/make_task1_complexity_breakdown.py \
      --runs-json config/paper_runs.json \
      --output-dir paper_assets/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluate import load as evaluate_load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Example:
    index: int
    question: str
    reference: str
    complexity: int
    image_stem: str


@dataclass
class Prediction:
    index: int
    question: str
    prediction: str
    reference: str
    image_stem: str


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("<image>", " ")).strip()


def normalize_question(value: Any) -> str:
    return normalize_text(value).lower()


def path_basename_any(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return Path(PureWindowsPath(text).name).name


def image_stem_any(value: Any) -> str:
    name = path_basename_any(value)
    return Path(name).stem.lower() if name else ""


def candidate_image_stem(row: Dict[str, Any]) -> str:
    for key in ("img_id", "image_id", "image_ref", "image_path", "image", "file_name", "filename"):
        if key in row:
            stem = image_stem_any(row.get(key))
            if stem:
                return stem
    return ""


def row_prediction(row: Dict[str, Any]) -> str:
    for key in ("prediction", "pred", "answer", "generated_answer", "response"):
        if key in row and row[key] is not None:
            return normalize_text(row[key])
    return ""


def row_reference(row: Dict[str, Any]) -> str:
    for key in ("ground_truth", "reference", "gt", "target", "answer_gt"):
        if key in row and row[key] is not None:
            return normalize_text(row[key])
    return ""


def load_predictions(path: Path) -> List[Prediction]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["__line_number"] = line_number
                rows.append(obj)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "predictions" in data:
            raw_predictions = data["predictions"]
            references = data.get("references") or []
            for idx, item in enumerate(raw_predictions):
                if isinstance(item, dict):
                    obj = dict(item)
                else:
                    obj = {"prediction": item}
                if idx < len(references) and "ground_truth" not in obj:
                    obj["ground_truth"] = references[idx]
                obj["__line_number"] = idx + 1
                rows.append(obj)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                obj = dict(item) if isinstance(item, dict) else {"prediction": item}
                obj["__line_number"] = idx + 1
                rows.append(obj)
        else:
            raise ValueError(f"Unsupported JSON prediction format: {path}")

    predictions: List[Prediction] = []
    for idx, row in enumerate(rows):
        pred = row_prediction(row)
        if not pred:
            continue
        predictions.append(
            Prediction(
                index=idx,
                question=normalize_question(row.get("question", row.get("question_text", ""))),
                prediction=pred,
                reference=row_reference(row),
                image_stem=candidate_image_stem(row),
            )
        )
    if not predictions:
        raise ValueError(f"No usable predictions found in {path}")
    return predictions


def load_test_examples() -> List[Example]:
    from datasets import load_dataset

    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    examples: List[Example] = []
    for idx, row in enumerate(ds):
        image_stem = ""
        for key in ("img_id", "image_id", "image_path", "file_name", "filename"):
            if key in row:
                image_stem = image_stem_any(row.get(key))
                if image_stem:
                    break
        examples.append(
            Example(
                index=idx,
                question=normalize_question(row.get("question", "")),
                reference=normalize_text(row.get("answer", "")),
                complexity=int(row.get("complexity", 0) or 0),
                image_stem=image_stem,
            )
        )
    return examples


def build_example_indices(examples: Sequence[Example]) -> Tuple[Dict[Tuple[str, str], Example], Dict[str, List[Example]]]:
    by_image_question: Dict[Tuple[str, str], Example] = {}
    by_question: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        if ex.image_stem and ex.question:
            by_image_question[(ex.image_stem, ex.question)] = ex
        if ex.question:
            by_question[ex.question].append(ex)
    return by_image_question, by_question


def align_predictions(preds: Sequence[Prediction], examples: Sequence[Example]) -> List[Tuple[Prediction, Example]]:
    """Align predictions to examples.

    Most full-test prediction files are emitted in test-set order. We still try
    image/question and unique-question matching first for robustness.
    """
    by_image_question, by_question = build_example_indices(examples)
    aligned: List[Tuple[Prediction, Example]] = []
    used_indices = set()

    for pred in preds:
        ex: Optional[Example] = None
        if pred.image_stem and pred.question:
            ex = by_image_question.get((pred.image_stem, pred.question))
        if ex is None and pred.question:
            candidates = by_question.get(pred.question, [])
            if len(candidates) == 1:
                ex = candidates[0]
        if ex is None and pred.index < len(examples):
            ex = examples[pred.index]
        if ex is None:
            continue
        aligned.append((pred, ex))
        used_indices.add(ex.index)

    return aligned


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    refs_nested = [[r] for r in references]
    out: Dict[str, float] = {"num_examples": float(len(predictions))}

    if not predictions:
        out.update({"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0})
        return out

    bleu = evaluate_load("bleu")
    rouge = evaluate_load("rouge")
    meteor = evaluate_load("meteor")

    bleu_result = bleu.compute(predictions=predictions, references=refs_nested)
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=refs_nested)

    out["bleu"] = round(float(bleu_result["bleu"]), 6)
    out["rougeL"] = round(float(rouge_result["rougeL"]), 6)
    out["meteor"] = round(float(meteor_result["meteor"]), 6)
    return out


def fmt(value: float) -> str:
    return f"{value:.6f}"


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_table(rows: List[Dict[str, Any]], output_path: Path) -> None:
    # Wide but still manageable: model x levels, METEOR + ROUGE-L.
    lines: List[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Hiệu năng Task 1 phân theo cấp độ phức tạp trên full-test.}")
    lines.append(r"\label{tab:complexity_breakdown}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{Level 1} & \multicolumn{2}{c}{Level 2} & \multicolumn{2}{c}{Level 3} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r" & METEOR & ROUGE-L & METEOR & ROUGE-L & METEOR & ROUGE-L \\")
    lines.append(r"\midrule")

    # Bold best per complexity/metric.
    best: Dict[Tuple[int, str], float] = {}
    for level in (1, 2, 3):
        for metric in ("meteor", "rougeL"):
            vals = [float(row[f"level{level}_{metric}"]) for row in rows if row.get(f"level{level}_n", 0) > 0]
            best[(level, metric)] = max(vals) if vals else -1.0

    for row in rows:
        cells = [latex_escape(str(row["name"]))]
        for level in (1, 2, 3):
            for metric in ("meteor", "rougeL"):
                key = f"level{level}_{metric}"
                value = float(row[key])
                text = fmt(value)
                if abs(value - best[(level, metric)]) < 1e-12:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Task 1 complexity breakdown table.")
    parser.add_argument("--runs-json", default="config/paper_runs.json")
    parser.add_argument("--output-dir", default="paper_assets/tables")
    args = parser.parse_args()

    runs_path = Path(args.runs_json)
    if not runs_path.is_absolute():
        runs_path = PROJECT_ROOT / runs_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    examples = load_test_examples()
    print(f"Loaded {len(examples)} test examples.")

    output_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"runs": []}

    for run in runs:
        name = str(run["name"])
        pred_path = Path(run["predictions"])
        if not pred_path.is_absolute():
            pred_path = PROJECT_ROOT / pred_path
        print(f"\n{name}: {pred_path}")
        preds = load_predictions(pred_path)
        aligned = align_predictions(preds, examples)
        print(f"  predictions={len(preds)} aligned={len(aligned)}")

        by_level: Dict[int, Tuple[List[str], List[str]]] = {}
        for level in (1, 2, 3):
            level_pairs = [(pred, ex) for pred, ex in aligned if ex.complexity == level]
            by_level[level] = (
                [pred.prediction for pred, _ in level_pairs],
                [ex.reference or pred.reference for pred, ex in level_pairs],
            )

        row: Dict[str, Any] = {"name": name, "predictions": str(pred_path), "aligned": len(aligned)}
        run_summary = {"name": name, "predictions": str(pred_path), "aligned": len(aligned), "levels": {}}
        for level in (1, 2, 3):
            level_preds, level_refs = by_level[level]
            metrics = compute_metrics(level_preds, level_refs)
            row[f"level{level}_n"] = int(metrics["num_examples"])
            row[f"level{level}_bleu"] = metrics["bleu"]
            row[f"level{level}_meteor"] = metrics["meteor"]
            row[f"level{level}_rougeL"] = metrics["rougeL"]
            run_summary["levels"][str(level)] = metrics
            print(
                f"  L{level}: n={int(metrics['num_examples'])} "
                f"BLEU={metrics['bleu']:.6f} METEOR={metrics['meteor']:.6f} ROUGE-L={metrics['rougeL']:.6f}"
            )
        output_rows.append(row)
        summary["runs"].append(run_summary)

    csv_path = output_dir / "task1_complexity_metrics.csv"
    fieldnames = [
        "name",
        "aligned",
        "level1_n",
        "level1_bleu",
        "level1_meteor",
        "level1_rougeL",
        "level2_n",
        "level2_bleu",
        "level2_meteor",
        "level2_rougeL",
        "level3_n",
        "level3_bleu",
        "level3_meteor",
        "level3_rougeL",
        "predictions",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_path = output_dir / "task1_complexity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    latex_path = output_dir / "task1_complexity_table.tex"
    write_latex_table(output_rows, latex_path)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {latex_path}")


if __name__ == "__main__":
    main()
