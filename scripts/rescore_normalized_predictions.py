import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.generative_vqa_metrics import aggregate_scores, score_prediction
from src.postprocessing.answer_normalization import normalize_prediction


CORE_KEYS = [
    "exact_match",
    "token_f1",
    "rouge_l",
    "bleu_1",
    "bleu_4",
    "qtype_yes_no_answer_acc",
    "qtype_numerical_count_answer_acc",
    "qtype_numerical_count_count_exact_acc",
    "qtype_numerical_count_count_tolerance1_acc",
]


def load_rows(path: Path, max_samples: int | None = None) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def metric_delta(before: Dict[str, float], after: Dict[str, float], key: str) -> float:
    return float(after.get(key, 0.0)) - float(before.get(key, 0.0))


def write_diff_report(path: Path, changed_rows: List[dict], raw_metrics: Dict[str, float], normalized_metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Normalization Diff Report\n\n")
        f.write("## Metric comparison\n\n")
        f.write("| Metric | Raw | Normalized | Delta |\n")
        f.write("|---|---:|---:|---:|\n")
        for key in CORE_KEYS:
            if key in raw_metrics or key in normalized_metrics:
                raw = float(raw_metrics.get(key, 0.0))
                norm = float(normalized_metrics.get(key, 0.0))
                f.write(f"| {key} | {raw:.6f} | {norm:.6f} | {norm - raw:+.6f} |\n")
        f.write("\n")
        f.write(f"Changed predictions: {len(changed_rows)}\n\n")

        f.write("## Changed examples\n\n")
        for row in changed_rows[:200]:
            before = row["raw_metrics"].get("token_f1", 0.0)
            after = row["metrics"].get("token_f1", 0.0)
            f.write(f"### Record {row['record_index']} | ΔF1={after - before:+.3f}\n\n")
            f.write(f"- Question: {row['question']}\n")
            f.write(f"- Ground truth: {row['ground_truth']}\n")
            f.write(f"- Raw: {row['prediction_raw']}\n")
            f.write(f"- Normalized: {row['prediction_normalized']}\n")
            f.write(f"- Raw F1: {before:.3f}\n")
            f.write(f"- Normalized F1: {after:.3f}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Re-score predictions after conservative answer normalization.")
    parser.add_argument("--predictions", required=True, help="Input predictions.jsonl from evaluate_structural_vqa_generative.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

    input_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path, max_samples=args.max_samples or None)
    raw_scores = []
    normalized_scores = []
    output_rows = []
    changed_rows = []

    for row in rows:
        question = row.get("question", "")
        gold = row.get("ground_truth", "")
        raw_pred = row.get("prediction", "")
        normalized_pred = normalize_prediction(raw_pred, question)

        raw_metric = row.get("metrics") or score_prediction(raw_pred, gold, question)
        normalized_metric = score_prediction(normalized_pred, gold, question)
        raw_scores.append(raw_metric)
        normalized_scores.append(normalized_metric)

        out = {
            "record_index": row.get("record_index"),
            "image_ref": row.get("image_ref"),
            "image_path": row.get("image_path"),
            "question": question,
            "ground_truth": gold,
            "prediction_raw": raw_pred,
            "prediction_normalized": normalized_pred,
            "changed": raw_pred != normalized_pred,
            "raw_metrics": raw_metric,
            "metrics": normalized_metric,
        }
        output_rows.append(out)
        if out["changed"]:
            changed_rows.append(out)

    raw_metrics = aggregate_scores(raw_scores)
    raw_metrics["num_examples"] = len(raw_scores)
    normalized_metrics = aggregate_scores(normalized_scores)
    normalized_metrics["num_examples"] = len(normalized_scores)
    summary = {
        "input_predictions": str(input_path),
        "num_examples": len(output_rows),
        "num_changed": len(changed_rows),
        "raw_metrics": raw_metrics,
        "normalized_metrics": normalized_metrics,
        "delta": {key: metric_delta(raw_metrics, normalized_metrics, key) for key in sorted(set(raw_metrics) | set(normalized_metrics))},
    }

    with (output_dir / "normalized_predictions.jsonl").open("w", encoding="utf-8") as f:
        for out in output_rows:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(normalized_metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_diff_report(output_dir / "diff_report.md", changed_rows, raw_metrics, normalized_metrics)

    print("\nMetric comparison")
    print("| Metric | Raw | Normalized | Delta |")
    print("|---|---:|---:|---:|")
    for key in CORE_KEYS:
        if key in raw_metrics or key in normalized_metrics:
            raw = float(raw_metrics.get(key, 0.0))
            norm = float(normalized_metrics.get(key, 0.0))
            print(f"| {key} | {raw:.6f} | {norm:.6f} | {norm - raw:+.6f} |")
    print(f"Changed predictions: {len(changed_rows)}/{len(output_rows)}")
    print(f"Wrote normalized outputs to {output_dir}")


if __name__ == "__main__":
    main()
