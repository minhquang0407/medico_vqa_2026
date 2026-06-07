"""Aggregate Task 2 judge scores into paper-ready tables.

Example:
    python scripts/aggregate_task2_judge_scores.py \
      --judge-results paper_assets/task2_judge_openai/task2_openai_judge_results.jsonl \
      --output-dir paper_assets/tables \
      --judge-name GPT-4o
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CRITERIA = ["correctness", "faithfulness", "relevance", "clarity", "completeness"]
DISPLAY = {
    "correctness": "Correctness",
    "faithfulness": "Faithfulness",
    "relevance": "Relevance",
    "clarity": "Clarity",
    "completeness": "Completeness",
}


def load_results(path: Path) -> List[Dict[str, Any]]:
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
        raise ValueError(f"No judge rows loaded from {path}")
    return rows


def score_value(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        score = float(value)
    except Exception:
        return float("nan")
    if score < 1 or score > 5:
        return float("nan")
    return score


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_samples": len(rows), "criteria": {}}
    valid_all = 0
    for row in rows:
        if all(1 <= score_value(row, key) <= 5 for key in CRITERIA):
            valid_all += 1
    summary["num_valid_samples"] = valid_all

    for key in CRITERIA:
        vals = [score_value(row, key) for row in rows]
        vals = [v for v in vals if v == v]
        if vals:
            summary["criteria"][key] = {
                "mean": round(float(statistics.mean(vals)), 4),
                "std": round(float(statistics.pstdev(vals)), 4) if len(vals) > 1 else 0.0,
                "min": round(float(min(vals)), 4),
                "max": round(float(max(vals)), 4),
                "n": len(vals),
            }
        else:
            summary["criteria"][key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    means = [summary["criteria"][key]["mean"] for key in CRITERIA]
    summary["overall_mean"] = round(float(statistics.mean(means)), 4) if means else 0.0
    return summary


def write_csv(summary: Dict[str, Any], output_path: Path, judge_name: str) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "judge", "criterion", "mean", "std", "min", "max", "n"])
        writer.writeheader()
        for key in CRITERIA:
            data = summary["criteria"][key]
            writer.writerow({"model": "CATA-Final", "judge": judge_name, "criterion": DISPLAY[key], **data})


def write_latex(summary: Dict[str, Any], output_path: Path, judge_name: str) -> None:
    values = {key: summary["criteria"][key]["mean"] for key in CRITERIA}
    best = max(values.values()) if values else 0.0

    def fmt(key: str) -> str:
        value = values[key]
        text = f"{value:.2f}"
        if abs(value - best) < 1e-12:
            return rf"\textbf{{{text}}}"
        return text

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Đánh giá chất lượng explanation Task 2 bằng {judge_name} trên {summary['num_valid_samples']} mẫu ngẫu nhiên.}}",
        r"\label{tab:task2_judge}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Correctness & Faithfulness & Relevance & Clarity & Completeness \\",
        r"\midrule",
        "CATA-Final & " + " & ".join(fmt(key) for key in CRITERIA) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Task 2 judge scores.")
    parser.add_argument("--judge-results", required=True)
    parser.add_argument("--output-dir", default="paper_assets/tables")
    parser.add_argument("--judge-name", default="Qwen7B")
    args = parser.parse_args()

    results_path = Path(args.judge_results)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_path)
    summary = summarize(rows)
    summary["judge_name"] = args.judge_name

    summary_path = output_dir / "task2_judge_summary.json"
    csv_path = output_dir / "task2_judge_scores.csv"
    tex_path = output_dir / "task2_judge_table.tex"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(summary, csv_path, args.judge_name)
    write_latex(summary, tex_path, args.judge_name)

    print("Task 2 judge summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()
