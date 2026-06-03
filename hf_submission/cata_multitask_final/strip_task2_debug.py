"""Remove internal debug metadata from a Task 2 JSONL file.

Usage:
    python strip_task2_debug.py --input submission_task2.jsonl --output submission_task2_clean.jsonl

Use --in-place to overwrite the input file after writing a .bak backup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_debug(input_path: Path, output_path: Path) -> tuple[int, int]:
    rows = 0
    removed = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if "debug" in row:
                row.pop("debug", None)
                removed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip debug field from Task 2 JSONL.")
    parser.add_argument("--input", default="submission_task2.jsonl", help="Input JSONL path.")
    parser.add_argument("--output", default="submission_task2_clean.jsonl", help="Output JSONL path.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input after creating <input>.bak.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path

    if args.in_place:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        rows, removed = strip_debug(input_path, temp_path)
        backup_path.write_bytes(input_path.read_bytes())
        temp_path.replace(input_path)
        output_path = input_path
        print(f"✅ Wrote clean file in place: {output_path}")
        print(f"✅ Backup saved: {backup_path}")
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        rows, removed = strip_debug(input_path, output_path)
        print(f"✅ Wrote clean file: {output_path}")

    print(f"Rows processed: {rows}")
    print(f"Rows with debug removed: {removed}")


if __name__ == "__main__":
    main()
