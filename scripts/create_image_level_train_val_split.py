"""Create an internal image-level train/validation split from a VQA JSONL file.

Why image-level?
  Kvasir-style VQA datasets often contain multiple questions for the same image.
  A row-level random split would leak the same image into both train and val.
  This script assigns each unique image to exactly one split.

Example:
  python scripts/create_image_level_train_val_split.py \
    --input data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl \
    --output-dir data/processed/splits/kvasir_train_image_level_seed42_val5 \
    --val-ratio 0.05 \
    --seed 42

Outputs:
  - train.jsonl
  - val.jsonl
  - split_summary.json
  - train_images.txt
  - val_images.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    records: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL decode error at {path}:{line_no}: {exc}") from exc
    return records


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_basename_any(value: str) -> str:
    """Return filename for POSIX or Windows-like paths on any OS."""
    return Path(PureWindowsPath(str(value)).name).name


def canonical_image_id(record: Dict[str, Any], *, basename_only: bool = True) -> str:
    """Extract a stable image id from a VQA record."""
    image_ref = ""
    images = record.get("images")
    if isinstance(images, list) and images:
        image_ref = str(images[0])
    elif isinstance(images, str):
        image_ref = images
    elif record.get("image") is not None:
        image_ref = str(record["image"])
    elif record.get("img_id") is not None:
        image_ref = str(record["img_id"])
    elif record.get("image_id") is not None:
        image_ref = str(record["image_id"])

    image_ref = image_ref.strip()
    if not image_ref:
        # Last-resort hash to avoid crashing, but this should be rare.
        return "missing_image__" + hashlib.sha1(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]

    if basename_only:
        return path_basename_any(image_ref).lower()
    return image_ref.replace("\\", "/").lower()


def infer_question_type(record: Dict[str, Any]) -> str:
    """Coarse question-type bucket for split diagnostics only."""
    q = str(record.get("question", record.get("query", ""))).strip().lower()
    if not q and isinstance(record.get("messages"), list):
        for message in record["messages"]:
            if isinstance(message, dict) and str(message.get("role", "")).lower() in {"user", "human"}:
                q = str(message.get("content", "")).strip().lower()
                break
    q_clean = q.replace("<image>", "").strip()
    first = q_clean.split(maxsplit=1)[0] if q_clean else "unknown"
    if first in {"is", "are", "does", "do", "can", "has", "have", "was", "were"}:
        return "yes_no"
    if first in {"how"}:
        if "how many" in q_clean:
            return "count"
        return "how"
    if first in {"what", "which"}:
        return "what_which"
    if first in {"where"}:
        return "where"
    if first in {"why"}:
        return "why"
    return first or "unknown"


def split_images(
    image_ids: List[str],
    *,
    val_ratio: float,
    seed: int,
    min_val_images: int,
) -> Tuple[set[str], set[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    rng = random.Random(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)

    n_val = max(min_val_images, round(len(shuffled) * val_ratio))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_images = set(shuffled[:n_val])
    train_images = set(shuffled[n_val:])
    return train_images, val_images


def summarize_split(
    input_path: Path,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    train_images: set[str],
    val_images: set[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_qtypes = Counter(infer_question_type(r) for r in train_records)
    val_qtypes = Counter(infer_question_type(r) for r in val_records)
    overlap = sorted(train_images & val_images)
    return {
        "input": str(input_path),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "basename_only": args.basename_only,
        "num_records": len(train_records) + len(val_records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_images": len(train_images) + len(val_images),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "image_overlap_count": len(overlap),
        "image_overlap_examples": overlap[:20],
        "train_question_type_counts": dict(train_qtypes.most_common()),
        "val_question_type_counts": dict(val_qtypes.most_common()),
        "outputs": {
            "train_jsonl": str(Path(args.output_dir) / "train.jsonl"),
            "val_jsonl": str(Path(args.output_dir) / "val.jsonl"),
            "train_images": str(Path(args.output_dir) / "train_images.txt"),
            "val_images": str(Path(args.output_dir) / "val_images.txt"),
            "summary": str(Path(args.output_dir) / "split_summary.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an internal image-level train/validation split from a VQA JSONL file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input VQA JSONL, e.g. data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.jsonl/val.jsonl and summary files will be written.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-val-images",
        type=int,
        default=1,
        help="Minimum number of unique images assigned to validation.",
    )
    parser.add_argument(
        "--basename-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use image basename as identity. Recommended when JSONL stores paths.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_with_lines = load_jsonl(input_path)
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, record in records_with_lines:
        by_image[canonical_image_id(record, basename_only=args.basename_only)].append(record)

    unique_images = sorted(by_image.keys())
    if len(unique_images) < 2:
        raise ValueError(f"Need at least 2 unique images, got {len(unique_images)}")

    train_images, val_images = split_images(
        unique_images,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_val_images=args.min_val_images,
    )

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    for _, record in records_with_lines:
        image_id = canonical_image_id(record, basename_only=args.basename_only)
        if image_id in val_images:
            val_records.append(record)
        else:
            train_records.append(record)

    dump_jsonl(output_dir / "train.jsonl", train_records)
    dump_jsonl(output_dir / "val.jsonl", val_records)
    (output_dir / "train_images.txt").write_text("\n".join(sorted(train_images)) + "\n", encoding="utf-8")
    (output_dir / "val_images.txt").write_text("\n".join(sorted(val_images)) + "\n", encoding="utf-8")

    summary = summarize_split(input_path, train_records, val_records, train_images, val_images, args)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["image_overlap_count"] != 0:
        raise SystemExit("Image-level leakage detected: train/val image overlap is non-zero")
    print(f"✅ Image-level split written to: {output_dir}")


if __name__ == "__main__":
    main()
