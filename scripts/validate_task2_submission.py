"""Validate MediaEval Medico 2026 Task 2 JSONL submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Image as HfImage
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK2_DIR = PROJECT_ROOT / "hf_submission" / "task2"


def load_official_subset():
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    return (
        ds.filter(lambda x: x["complexity"] == 1)
        .shuffle(seed=42)
        .select(range(1500))
        .add_column("val_id", list(range(1500)))
        .remove_columns(["complexity", "answer", "original", "question_class"])
        .cast_column("image", HfImage())
    )


def validate(task2_dir: Path, expected_count: int = 1500) -> None:
    jsonl_path = task2_dir / "submission_task2.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

    if len(rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} rows, got {len(rows)}")

    official = load_official_subset()
    seen = set()
    required = {"val_id", "img_id", "question", "answer", "textual_explanation"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise AssertionError(f"Missing keys {missing} in row {row}")
        val_id = int(row["val_id"])
        if val_id in seen:
            raise AssertionError(f"Duplicate val_id {val_id}")
        seen.add(val_id)
        if not (0 <= val_id < expected_count):
            raise AssertionError(f"val_id out of range: {val_id}")
        ref = official[val_id]
        if str(row["img_id"]) != str(ref["img_id"]):
            raise AssertionError(f"img_id mismatch at val_id={val_id}: {row['img_id']} != {ref['img_id']}")
        if str(row["question"]) != str(ref["question"]):
            raise AssertionError(f"question mismatch at val_id={val_id}")
        if not str(row["answer"]).strip():
            raise AssertionError(f"Empty answer at val_id={val_id}")
        if len(str(row["textual_explanation"]).strip()) < 40:
            raise AssertionError(f"Explanation too short at val_id={val_id}")
        if "confidence_score" in row:
            conf = float(row["confidence_score"])
            if not 0.0 <= conf <= 1.0:
                raise AssertionError(f"confidence_score out of range at val_id={val_id}: {conf}")
        for visual in row.get("visual_explanation", []) or []:
            data = visual.get("data")
            if data and not (task2_dir / data).exists():
                raise FileNotFoundError(f"Missing visual for val_id={val_id}: {data}")

    if seen != set(range(expected_count)):
        missing = sorted(set(range(expected_count)) - seen)[:10]
        raise AssertionError(f"Missing val_ids, first missing: {missing}")

    print(f"Valid Task 2 submission: {len(rows)} rows in {jsonl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-dir", type=Path, default=DEFAULT_TASK2_DIR)
    parser.add_argument("--expected-count", type=int, default=1500)
    args = parser.parse_args()
    validate(args.task2_dir, args.expected_count)


if __name__ == "__main__":
    main()
