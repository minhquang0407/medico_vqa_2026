import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


AUG_INDEX_PATTERN = re.compile(r"_aug(\d{2})(?=\.[^.]+$)", re.IGNORECASE)
DEFAULT_SPLITS = (1, 3, 5, 10)


def read_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi ở dòng {line_number} của {path}: {exc}") from exc


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def get_image_ref(record):
    images = record.get("images")
    if not images:
        return ""
    return str(images[0])


def get_aug_index(record):
    """
    Lấy index augment từ metadata hoặc từ tên ảnh dạng *_aug01.jpg.
    Trả về None nếu record không phải augmented sample.
    """
    augmentation = record.get("augmentation")
    if isinstance(augmentation, dict):
        index = augmentation.get("index")
        if isinstance(index, int):
            return index
        if isinstance(index, str) and index.isdigit():
            return int(index)

    image_ref = get_image_ref(record)
    match = AUG_INDEX_PATTERN.search(Path(image_ref).name)
    if match:
        return int(match.group(1))
    return None


def load_records(path, label):
    records = []
    for _, record in tqdm(list(read_jsonl(path)), desc=f"Load {label}"):
        records.append(record)
    return records


def filter_aug_records(aug_records, max_aug_index):
    selected = []
    skipped_non_aug = 0
    for record in aug_records:
        aug_index = get_aug_index(record)
        if aug_index is None:
            skipped_non_aug += 1
            continue
        if 1 <= aug_index <= max_aug_index:
            selected.append(record)
    return selected, skipped_non_aug


def verify_image_paths(records, max_missing_report=20):
    missing = []
    for record in records:
        image_ref = get_image_ref(record)
        if not image_ref:
            missing.append("<empty images>")
            continue
        if not Path(image_ref).exists():
            missing.append(image_ref)
            if len(missing) >= max_missing_report:
                break
    return missing


def make_output_paths(output_dir, prefix, split):
    output_dir = Path(output_dir)
    aug_path = output_dir / f"{prefix}-aug{split}.jsonl"
    mix_path = output_dir / f"{prefix}-mix-original-aug{split}.jsonl"
    return aug_path, mix_path


def prepare_ablation_splits(
    original_jsonl,
    augmented_jsonl,
    output_dir,
    prefix="Kvasir-VQA-x1-train",
    splits=DEFAULT_SPLITS,
    verify_images=False,
):
    original_jsonl = Path(original_jsonl)
    augmented_jsonl = Path(augmented_jsonl)
    output_dir = Path(output_dir)

    if not original_jsonl.exists():
        raise FileNotFoundError(f"Không tìm thấy original_jsonl: {original_jsonl}")
    if not augmented_jsonl.exists():
        raise FileNotFoundError(f"Không tìm thấy augmented_jsonl: {augmented_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)

    original_records = load_records(original_jsonl, "original")
    aug_records = load_records(augmented_jsonl, "augmented")

    print("\n======================================")
    print("📦 PREPARE AUGMENTATION ABLATION SPLITS")
    print("======================================")
    print(f"Original JSONL:  {original_jsonl}")
    print(f"Augmented JSONL: {augmented_jsonl}")
    print(f"Output dir:      {output_dir}")
    print(f"Original rows:   {len(original_records)}")
    print(f"Augmented rows:  {len(aug_records)}")
    print(f"Splits:          {', '.join('aug' + str(s) for s in splits)}")
    print("--------------------------------------")

    summary_rows = []
    global_skipped_non_aug = None

    for split in splits:
        selected_aug, skipped_non_aug = filter_aug_records(aug_records, split)
        if global_skipped_non_aug is None:
            global_skipped_non_aug = skipped_non_aug

        aug_path, mix_path = make_output_paths(output_dir, prefix, split)

        aug_count = write_jsonl(selected_aug, aug_path)
        mix_count = write_jsonl([*original_records, *selected_aug], mix_path)

        expected_aug = len(original_records) * split
        aug_delta = aug_count - expected_aug

        image_missing = []
        if verify_images:
            image_missing = verify_image_paths(selected_aug)

        summary = {
            "split": f"aug{split}",
            "aug_path": str(aug_path),
            "mix_path": str(mix_path),
            "aug_rows": aug_count,
            "mix_rows": mix_count,
            "expected_aug_rows": expected_aug,
            "aug_delta": aug_delta,
            "missing_image_examples": image_missing,
        }
        summary_rows.append(summary)

        print(f"✅ aug{split}")
        print(f"   Aug JSONL: {aug_path}")
        print(f"   Mix JSONL: {mix_path}")
        print(f"   Aug rows: {aug_count} | expected≈{expected_aug} | delta={aug_delta:+d}")
        print(f"   Mix rows: {mix_count}")
        if verify_images:
            if image_missing:
                print(f"   ⚠️ Missing images examples: {image_missing[:5]}")
            else:
                print("   Image path check: OK")

    summary_path = output_dir / f"{prefix}-ablation-summary.json"
    summary_payload = {
        "original_jsonl": str(original_jsonl),
        "augmented_jsonl": str(augmented_jsonl),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "original_rows": len(original_records),
        "augmented_rows": len(aug_records),
        "skipped_non_aug_records_in_augmented_jsonl": global_skipped_non_aug or 0,
        "splits": summary_rows,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("--------------------------------------")
    print(f"Summary JSON: {summary_path}")
    print("✅ Hoàn tất prepare ablation splits.")
    print("======================================\n")


def parse_splits(value):
    if not value:
        return DEFAULT_SPLITS
    splits = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        split = int(item)
        if split <= 0:
            raise argparse.ArgumentTypeError("split phải là số nguyên dương, ví dụ 1,3,5,10")
        splits.append(split)
    return tuple(sorted(set(splits)))


def main():
    parser = argparse.ArgumentParser(
        description="Tạo JSONL ablation splits từ file augmented xN."
    )
    parser.add_argument(
        "--original-jsonl",
        required=True,
        help="JSONL train gốc.",
    )
    parser.add_argument(
        "--augmented-jsonl",
        required=True,
        help="JSONL augmented đầy đủ, ví dụ aug10 hoặc aug20.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Thư mục ghi các file aug/mix ablation.",
    )
    parser.add_argument(
        "--prefix",
        default="Kvasir-VQA-x1-train",
        help="Prefix tên file output.",
    )
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=DEFAULT_SPLITS,
        help="Danh sách split, ví dụ: 1,3,5,10.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Kiểm tra nhanh đường dẫn ảnh trong JSONL có tồn tại không.",
    )
    args = parser.parse_args()

    prepare_ablation_splits(
        original_jsonl=args.original_jsonl,
        augmented_jsonl=args.augmented_jsonl,
        output_dir=args.output_dir,
        prefix=args.prefix,
        splits=args.splits,
        verify_images=args.verify_images,
    )


if __name__ == "__main__":
    main()
