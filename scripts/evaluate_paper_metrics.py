"""Compute paper-ready generation metrics from predictions_1.json or predictions.jsonl.

Uses huggingface evaluate library matching the official submission_task1.py pattern.

Metrics computed:
  - BLEU (evaluate bleu)
  - ROUGE-1 / ROUGE-2 / ROUGE-L (evaluate rouge)
  - METEOR (evaluate meteor)
  - chrF++ (evaluate chrf)  — lightweight, always enabled
  - SacreBLEU (evaluate sacrebleu) — corpus-level BLEU
  - BERTScore-F1 (evaluate bertscore) — optional, pass --bertscore

Usage:
  python scripts/evaluate_paper_metrics.py --input predictions_1.json
  python scripts/evaluate_paper_metrics.py --input eval/YOUR_RUN/predictions.jsonl
  python scripts/evaluate_paper_metrics.py --input predictions_1.json --bertscore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluate import load as evaluate_load


def load_preds_refs(path: Path) -> tuple[List[str], List[str], List[str]]:
    """Return (predictions, references, questions) from predictions_1.json or predictions.jsonl."""
    text = path.read_text(encoding="utf-8")

    # predictions_1.json = the official submission output format
    if path.suffix == ".json":
        data = json.loads(text)
        # official format: {"predictions": [...], "references": [...]}
        # or {"predictions": [{"answer": ..., "question": ...}], ...}
        if isinstance(data, dict) and "predictions" in data:
            preds_raw = data["predictions"]
            preds = [str(item.get("answer", item) if isinstance(item, dict) else item) for item in preds_raw]
            # references may be in the same file if the template saves them
            if "references" in data:
                refs = [str(r) for r in data["references"]]
            else:
                # fall back: load from HF dataset like submission_task1.py
                refs = _load_references_from_hf(len(preds))
            questions = [
                str(item.get("question", "")) for item in preds_raw
                if isinstance(item, dict)
            ]
            if len(questions) != len(preds):
                questions = [""] * len(preds)
            return preds, refs, questions
        # bare list
        if isinstance(data, list):
            preds = [str(item.get("answer", item) if isinstance(item, dict) else item) for item in data]
            refs = _load_references_from_hf(len(preds))
            return preds, refs, [""] * len(preds)
        raise ValueError(f"Unrecognized JSON structure in {path}")

    # .jsonl format from evaluate_structural_vqa_generative.py
    preds, refs, questions = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        preds.append(str(obj.get("prediction", obj.get("pred", obj.get("answer", "")))))
        refs.append(str(obj.get("ground_truth", obj.get("reference", obj.get("gt", "")))))
        questions.append(str(obj.get("question", obj.get("question_text", ""))))
    return preds, refs, questions


def _load_references_from_hf(n: int) -> List[str]:
    """Load reference answers from the official HF dataset (same as submission_task1.py)."""
    try:
        from datasets import load_dataset
        from PIL import Image as HfImage

        print("Loading references from SimulaMet/Kvasir-VQA-x1 ...")
        ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
        ds_shuffled = ds.shuffle(seed=42).select(range(1500))
        refs = ds_shuffled["answer"][:n]
        return [str(r) for r in refs]
    except Exception as exc:
        print(f"Warning: could not load HF references: {exc}")
        return [""] * n


def compute_metrics(
    preds: List[str],
    refs: List[str],
    *,
    use_bertscore: bool = False,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    bertscore_batch_size: int = 16,
    require_bertscore: bool = False,
) -> Dict[str, Any]:
    """Compute all metrics using the evaluate library (same pattern as submission_task1.py)."""
    # The evaluate library expects references as list of lists for bleu/meteor
    refs_nested = [[r] for r in refs]

    results: Dict[str, Any] = {"num_examples": len(preds)}

    # ---- BLEU ----
    try:
        bleu = evaluate_load("bleu")
        bleu_result = bleu.compute(predictions=preds, references=refs_nested)
        results["bleu"] = round(float(bleu_result["bleu"]), 6)
    except Exception as exc:
        results["bleu_error"] = str(exc)

    # ---- ROUGE ----
    try:
        rouge = evaluate_load("rouge")
        rouge_result = rouge.compute(predictions=preds, references=refs)
        results["rouge1"] = round(float(rouge_result["rouge1"]), 6)
        results["rouge2"] = round(float(rouge_result["rouge2"]), 6)
        results["rougeL"] = round(float(rouge_result["rougeL"]), 6)
        results["rougeLsum"] = round(float(rouge_result.get("rougeLsum", rouge_result["rougeL"])), 6)
    except Exception as exc:
        results["rouge_error"] = str(exc)

    # ---- METEOR ----
    try:
        meteor = evaluate_load("meteor")
        meteor_result = meteor.compute(predictions=preds, references=refs_nested)
        results["meteor"] = round(float(meteor_result["meteor"]), 6)
    except Exception as exc:
        results["meteor_error"] = str(exc)

    # ---- chrF++ ----
    try:
        chrf = evaluate_load("chrf")
        chrf_result = chrf.compute(
            predictions=preds,
            references=refs_nested,
            word_order=2,  # chrF++ (word_order=2)
        )
        results["chrf_pp"] = round(float(chrf_result["score"]) / 100.0, 6)
        results["chrf_pp_raw"] = round(float(chrf_result["score"]), 4)
    except Exception as exc:
        results["chrf_error"] = str(exc)

    # ---- SacreBLEU (corpus-level) ----
    try:
        sacrebleu = evaluate_load("sacrebleu")
        sacrebleu_result = sacrebleu.compute(predictions=preds, references=refs_nested)
        results["sacrebleu"] = round(float(sacrebleu_result["score"]) / 100.0, 6)
        results["sacrebleu_raw"] = round(float(sacrebleu_result["score"]), 4)
    except Exception as exc:
        results["sacrebleu_error"] = str(exc)

    # ---- BERTScore (optional) ----
    if use_bertscore:
        try:
            bertscore = evaluate_load("bertscore")
            bertscore_result = bertscore.compute(
                predictions=preds,
                references=refs,
                model_type=bertscore_model,
                batch_size=bertscore_batch_size,
            )
            results["bertscore_f1"] = round(float(sum(bertscore_result["f1"]) / len(bertscore_result["f1"])), 6)
            results["bertscore_precision"] = round(float(sum(bertscore_result["precision"]) / len(bertscore_result["precision"])), 6)
            results["bertscore_recall"] = round(float(sum(bertscore_result["recall"]) / len(bertscore_result["recall"])), 6)
            results["bertscore_model"] = bertscore_model
        except Exception as exc:
            if require_bertscore:
                raise
            results["bertscore_error"] = str(exc)

    return results


def print_table(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics."""
    print("\n" + "=" * 60)
    print("  PAPER METRICS SUMMARY")
    print("=" * 60)
    key_order = [
        "bleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor",
        "chrf_pp", "sacrebleu",
        "bertscore_f1", "bertscore_precision", "bertscore_recall",
        "num_examples",
    ]
    for key in key_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<25} {val:.6f}")
            else:
                print(f"  {key:<25} {val}")
    errors = {k: v for k, v in metrics.items() if "error" in k}
    if errors:
        print("\n  SKIPPED (missing packages or errors):")
        for k, v in errors.items():
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paper-ready metrics from predictions_1.json or predictions.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to predictions_1.json or predictions.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path. Defaults to <input parent>/paper_metrics.json",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        default=False,
        help="Also compute BERTScore-F1 (slow, downloads model on first run).",
    )
    parser.add_argument(
        "--bertscore-model",
        default="microsoft/deberta-xlarge-mnli",
        help="Model for BERTScore. Lighter option: roberta-large",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--require-bertscore",
        action="store_true",
        default=False,
        help="Fail instead of writing output if BERTScore cannot be computed.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output) if args.output else input_path.parent / "paper_metrics.json"

    print(f"Loading: {input_path}")
    preds, refs, questions = load_preds_refs(input_path)
    print(f"Loaded {len(preds)} examples.")

    metrics = compute_metrics(
        preds,
        refs,
        use_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        require_bertscore=args.require_bertscore,
    )

    print_table(metrics)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
