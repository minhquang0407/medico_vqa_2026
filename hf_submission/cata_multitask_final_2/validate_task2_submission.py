"""Validate the MediaEval Medico 2026 Subtask 2 JSONL submission.

Checks the organizer-defined Subtask 2 validation subset, JSONL schema, visual
paths, confidence range, and optional answer consistency with Task 1 predictions.

Usage:
  python validate_task2_submission.py
  python validate_task2_submission.py --task1-predictions ../task1_cata/predictions_1.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


NEG_PATTERNS = (
    r"\bno\b",
    r"not observed",
    r"not identified",
    r"no evidence",
    r"absent",
    r"without",
)
POS_PATTERNS = (
    r"evidence of",
    r"observed",
    r"visible",
    r"present",
    r"identified",
    r"lesion",
    r"polyp",
    r"instrument visible",
    r"text is present",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at line {line_no}: {exc}") from exc
            rows.append(row)
    return rows


def build_val_set() -> List[Dict[str, Any]]:
    from datasets import Image as HfImage
    from datasets import load_dataset

    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    val_set_task2 = (
        ds.filter(lambda x: x["complexity"] == 1)
        .shuffle(seed=42)
        .select(range(1500))
        .add_column("val_id", list(range(1500)))
        .remove_columns(["complexity", "answer", "original", "question_class"])
        .cast_column("image", HfImage())
    )
    return [dict(row) for row in val_set_task2]


def load_task1_predictions(path: Path) -> Dict[Tuple[str, str], str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", data if isinstance(data, list) else [])
    mapping: Dict[Tuple[str, str], str] = {}
    for row in preds:
        img_id = str(row.get("img_id", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        if img_id and question:
            mapping[(img_id, question)] = answer
    return mapping


def has_any(patterns: Iterable[str], text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in patterns)


def detect_possible_contradiction(row: Dict[str, Any]) -> str | None:
    answer = str(row.get("answer", ""))
    explanation = str(row.get("textual_explanation", ""))
    answer_negative = has_any(NEG_PATTERNS, answer)
    explanation_positive = has_any(POS_PATTERNS, explanation)
    says_consistent = "consistent with the primary answer" in explanation.lower()
    says_conflict = any(
        phrase in explanation.lower()
        for phrase in ("conflict", "mixed", "partially consistent", "limited support", "caution")
    )
    if answer_negative and explanation_positive and says_consistent and not says_conflict:
        return "negative answer but explanation/probes contain positive evidence and say consistent"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task 2 submission JSONL.")
    parser.add_argument("--submission", default="submission_task2.jsonl")
    parser.add_argument("--task1-predictions", default=None)
    parser.add_argument("--skip-dataset-check", action="store_true")
    parser.add_argument("--max-report", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    submission_path = Path(args.submission)
    if not submission_path.is_absolute():
        submission_path = root / submission_path

    rows = load_jsonl(submission_path)
    errors: List[str] = []
    warnings: List[str] = []

    if len(rows) != 1500:
        errors.append(f"Expected 1500 rows, found {len(rows)}")

    expected_rows = None if args.skip_dataset_check else build_val_set()

    task1_mapping = None
    if args.task1_predictions:
        task1_path = Path(args.task1_predictions)
        if not task1_path.is_absolute():
            task1_path = root / task1_path
        task1_mapping = load_task1_predictions(task1_path)

    required = {"val_id", "img_id", "question", "answer", "textual_explanation"}

    for idx, row in enumerate(rows):
        prefix = f"row {idx}"
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{prefix}: missing required fields {missing}")
            continue

        val_id_raw = row.get("val_id")
        try:
            val_id = int(val_id_raw)
        except Exception:
            errors.append(f"{prefix}: val_id is not int-like: {val_id_raw!r}")
            continue
        if val_id != idx:
            errors.append(f"{prefix}: val_id={val_id}, expected {idx}")

        img_id = str(row.get("img_id", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        explanation = str(row.get("textual_explanation", ""))

        if expected_rows is not None and idx < len(expected_rows):
            expected = expected_rows[idx]
            if img_id != str(expected.get("img_id", "")):
                errors.append(f"{prefix}: img_id mismatch: {img_id!r} != {expected.get('img_id')!r}")
            if question != str(expected.get("question", "")):
                errors.append(f"{prefix}: question mismatch")

        if not answer.strip():
            errors.append(f"{prefix}: empty answer")
        if len(explanation.strip()) < 40:
            warnings.append(f"{prefix}: textual_explanation is very short")

        confidence = row.get("confidence_score")
        if confidence is not None:
            try:
                c = float(confidence)
                if not (0.0 <= c <= 1.0):
                    errors.append(f"{prefix}: confidence_score outside [0,1]: {confidence!r}")
            except Exception:
                errors.append(f"{prefix}: confidence_score is not numeric: {confidence!r}")

        visual_items = row.get("visual_explanation", [])
        if visual_items is None:
            visual_items = []
        if not isinstance(visual_items, list):
            errors.append(f"{prefix}: visual_explanation is not a list")
        else:
            for v_idx, item in enumerate(visual_items):
                if not isinstance(item, dict):
                    errors.append(f"{prefix}: visual_explanation[{v_idx}] is not object")
                    continue
                data = item.get("data")
                if isinstance(data, str) and data and not data.startswith("[["):
                    data_path = root / data
                    if not data_path.exists():
                        errors.append(f"{prefix}: missing visual file {data}")

        contradiction = detect_possible_contradiction(row)
        if contradiction:
            warnings.append(f"{prefix}: possible contradiction: {contradiction}")

        if task1_mapping is not None:
            task1_answer = task1_mapping.get((img_id, question))
            if task1_answer is None:
                warnings.append(f"{prefix}: no matching Task 1 prediction for img_id/question")
            elif answer.strip() != task1_answer.strip():
                errors.append(
                    f"{prefix}: answer mismatch with Task 1: {answer!r} != {task1_answer!r}"
                )

    print("=" * 80)
    print("Task 2 submission validation")
    print("=" * 80)
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for item in errors[: args.max_report]:
            print(" -", item)
        if len(errors) > args.max_report:
            print(f" ... {len(errors) - args.max_report} more errors")

    if warnings:
        print("\nWARNINGS:")
        for item in warnings[: args.max_report]:
            print(" -", item)
        if len(warnings) > args.max_report:
            print(f" ... {len(warnings) - args.max_report} more warnings")

    if errors:
        raise SystemExit(1)
    print("\n✅ Task 2 JSONL structure and paths look valid.")


if __name__ == "__main__":
    main()
