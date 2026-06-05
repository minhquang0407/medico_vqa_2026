"""Compute paper-ready generation metrics from a predictions JSONL file.

Expected JSONL fields:
  - prediction
  - ground_truth (or answer/reference)
  - question (optional)

The script reports:
  - existing internal metrics from src.evaluation.generative_vqa_metrics
  - corpus BLEU via sacrebleu when available
  - ROUGE-1/2/L via rouge-score when available
  - METEOR via NLTK when available
  - chrF++ via sacrebleu when available
  - BERTScore-F1 via bert-score when explicitly enabled

All external metrics are optional: missing dependencies are skipped instead of
crashing, except when --require-bertscore is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generative_vqa_metrics import (
    aggregate_scores,
    compute_corpus_generation_metrics,
    score_prediction,
)


def _get_first(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = _get_first(obj, ["prediction", "pred", "answer", "generated_answer"])
            gt = _get_first(obj, ["ground_truth", "reference", "target", "gold", "gt", "answer_text"])
            question = _get_first(obj, ["question", "question_text"], default="")
            if not gt:
                raise ValueError(f"Missing ground-truth/reference field at line {line_no}: {obj.keys()}")
            rows.append({"prediction": pred, "ground_truth": gt, "question": question})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-ready generation metrics from predictions JSONL.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", default=None, help="Output metrics JSON path. Defaults to <predictions parent>/paper_metrics.json")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore-F1. This is slower and may download a model.")
    parser.add_argument("--bertscore-only", action="store_true", help="Only compute BERTScore-F1 and merge with existing paper_metrics.json if available.")
    parser.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--require-bertscore", action="store_true", help="Fail if BERTScore cannot be computed.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.output) if args.output else pred_path.parent / "paper_metrics.json"

    rows = load_rows(pred_path)
    predictions = [row["prediction"] for row in rows]
    references = [row["ground_truth"] for row in rows]
    questions = [row["question"] for row in rows]

    if args.bertscore_only:
        existing_path = pred_path.parent / "paper_metrics.json"
        if out_path.exists():
            metrics = json.loads(out_path.read_text(encoding="utf-8"))
        elif existing_path.exists():
            metrics = json.loads(existing_path.read_text(encoding="utf-8"))
        else:
            metrics = {}
        metrics.update(
            compute_corpus_generation_metrics(
                predictions,
                references,
                compute_bertscore=True,
                bertscore_model=args.bertscore_model,
                bertscore_batch_size=args.bertscore_batch_size,
                require_bertscore=args.require_bertscore,
            )
        )
    else:
        per_row = [score_prediction(p, r, q) for p, r, q in zip(predictions, references, questions)]
        metrics = aggregate_scores(per_row)
        metrics.update(
            compute_corpus_generation_metrics(
                predictions,
                references,
                compute_bertscore=args.bertscore,
                bertscore_model=args.bertscore_model,
                bertscore_batch_size=args.bertscore_batch_size,
                require_bertscore=args.require_bertscore,
            )
        )
    metrics["num_examples"] = len(rows)

    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
