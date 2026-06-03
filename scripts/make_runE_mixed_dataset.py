#!/usr/bin/env python3
"""Build a local Run E mixed dataset for Colab training.

This script creates a self-contained dataset directory/zip with a flat image
folder so it can be trained exactly like Run D:

    --jsonl data/raw/runE_mixed/train_25000_orig_25000_weakaug_seed42.jsonl
    --images-dir data/raw/runE_mixed/images

Why flat images?
The current dataset resolver falls back to `images_dir / basename`, so mixed
records that point to both `images/...` and `image_weak_augmented/...` can be
fragile. This script copies selected images into one flat folder with safe
prefixes:

    orig__<stem>.jpg
    aug__<stem>.jpg

and rewrites every JSONL record to `images=["orig__...jpg"]` or
`images=["aug__...jpg"]`.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def basename_any(value: str | Path) -> str:
    text = str(value)
    return Path(PureWindowsPath(text).name).name


def get_first_image_ref(row: Dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    for key in ("image", "image_path", "img_path", "path", "file_name", "filename"):
        if row.get(key):
            return str(row[key])
    return ""


def get_image_stem(row: Dict[str, Any]) -> str:
    # Prefer true image identifiers. Do not use generic `id`; it may be QA id.
    for key in ("img_id", "image_id"):
        if row.get(key):
            return Path(basename_any(str(row[key]))).stem
    ref = get_first_image_ref(row)
    if ref:
        return Path(basename_any(ref)).stem
    return ""


def ensure_question_answer(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    if row.get("question") is not None and row.get("answer") is not None:
        row["question"] = str(row["question"]).replace("<image>", "").strip()
        row["answer"] = str(row["answer"]).strip()
        return row

    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    messages = row.get("messages", [])

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                    if text:
                        parts.append(str(text))
            content = " ".join(parts)
        else:
            content = str(content)

        if role == "user":
            question = content.replace("<image>", "").strip()
        elif role == "assistant":
            answer = content.strip()

    row["question"] = question.replace("<image>", "").strip()
    row["answer"] = answer.strip()
    return row


def set_flat_image_ref(row: Dict[str, Any], filename: str) -> Dict[str, Any]:
    row = ensure_question_answer(row)
    row["images"] = [filename]
    for key in ("image", "image_path", "img_path", "path", "file_name", "filename"):
        row.pop(key, None)
    return row


def index_images(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    indexed: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            indexed.setdefault(path.stem, path)
    return indexed


def choose_image(row: Dict[str, Any], index: Dict[str, Path]) -> Optional[Path]:
    stem = get_image_stem(row)
    if stem in index:
        return index[stem]

    ref = get_first_image_ref(row)
    if ref:
        ref_name = basename_any(ref)
        ref_stem = Path(ref_name).stem
        if ref_stem in index:
            return index[ref_stem]

    return None


def copy_as(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def build_rows(
    rows: List[Dict[str, Any]],
    image_index: Dict[str, Path],
    out_images_dir: Path,
    prefix: str,
) -> Tuple[List[Dict[str, Any]], int]:
    fixed: List[Dict[str, Any]] = []
    missing = 0

    for row in rows:
        src = choose_image(row, image_index)
        if src is None:
            missing += 1
            continue

        flat_name = f"{prefix}__{src.stem}{src.suffix.lower()}"
        copy_as(src, out_images_dir / flat_name)
        fixed.append(set_flat_image_ref(row, flat_name))

    return fixed, missing


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Run E mixed original+weak-aug dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/Kvasir-VQA-x1"),
        help="Dataset root containing images/, image_weak_augmented/, and JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/runE_mixed_local"),
        help="Output directory to create.",
    )
    parser.add_argument("--n-orig", type=int, default=25000)
    parser.add_argument("--n-aug", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true", help="Also create a zip beside output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = args.root
    orig_jsonl = root / "Kvasir-VQA-x1-train.jsonl"
    aug_jsonl = root / "Kvasir-VQA-x1-train-aug.jsonl"
    orig_img_dir = root / "images"
    aug_img_dir = root / "image_weak_augmented"

    if not orig_jsonl.exists():
        raise FileNotFoundError(f"Missing original JSONL: {orig_jsonl}")
    if not aug_jsonl.exists():
        raise FileNotFoundError(f"Missing augmented JSONL: {aug_jsonl}")

    out_dir = args.output_dir
    out_images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    orig_index = index_images(orig_img_dir)
    aug_index = index_images(aug_img_dir)
    print(f"Original image files:  {len(orig_index)}")
    print(f"Augmented image files: {len(aug_index)}")

    print("Reading JSONLs...")
    orig_raw = read_jsonl(orig_jsonl)
    aug_raw = read_jsonl(aug_jsonl)
    print(f"Original rows raw:  {len(orig_raw)}")
    print(f"Augmented rows raw: {len(aug_raw)}")

    print("Resolving/copying original rows...")
    orig_rows, missing_orig = build_rows(orig_raw, orig_index, out_images_dir, "orig")
    print(f"Original kept: {len(orig_rows)} | missing: {missing_orig}")

    print("Resolving/copying augmented rows...")
    aug_rows, missing_aug = build_rows(aug_raw, aug_index, out_images_dir, "aug")
    print(f"Augmented kept: {len(aug_rows)} | missing: {missing_aug}")

    if len(orig_rows) < 1000:
        raise RuntimeError(f"Too few original rows after image matching: {len(orig_rows)}")
    if len(aug_rows) < 1000:
        raise RuntimeError(f"Too few augmented rows after image matching: {len(aug_rows)}")

    n_orig = min(args.n_orig, len(orig_rows))
    n_aug = min(args.n_aug, len(aug_rows))
    orig_sample = random.sample(orig_rows, n_orig)
    aug_sample = random.sample(aug_rows, n_aug)

    mixed = orig_sample + aug_sample
    random.shuffle(mixed)

    out_jsonl = out_dir / f"train_{n_orig}_orig_{n_aug}_weakaug_seed{args.seed}.jsonl"
    write_jsonl(out_jsonl, mixed)

    manifest = {
        "seed": args.seed,
        "root": str(root),
        "n_orig_requested": args.n_orig,
        "n_aug_requested": args.n_aug,
        "n_orig_used": n_orig,
        "n_aug_used": n_aug,
        "n_total": len(mixed),
        "original_missing": missing_orig,
        "augmented_missing": missing_aug,
        "jsonl": out_jsonl.name,
        "images_dir": "images",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("Output dir:", out_dir)
    print("Mixed JSONL:", out_jsonl)
    print("Images dir:", out_images_dir)
    print("Total copied images:", sum(1 for _ in out_images_dir.iterdir()))
    print("First 5 refs:")
    for row in mixed[:5]:
        ref = get_first_image_ref(row)
        print(" ", ref, "exists=", (out_images_dir / ref).exists())

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        print("Creating zip:", zip_path)
        zip_dir(out_dir, zip_path)
        print("Zip ready:", zip_path)


if __name__ == "__main__":
    main()
