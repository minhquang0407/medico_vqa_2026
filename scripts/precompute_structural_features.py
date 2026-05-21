import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.topology.lesion_prior import LesionPriorExtractor
from src.topology.tda_morphology import TopologicalExtractor as MorphologyTopologicalExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_IMAGE_SIZE = (224, 224)


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


def get_image_ref(record):
    images = record.get("images")
    if not images:
        return ""
    return str(images[0])


def resolve_image_path(image_ref, images_dir=None):
    """
    Resolve ảnh từ JSONL.

    Ưu tiên:
    1. Nếu image_ref là path tồn tại, dùng luôn.
    2. Nếu images_dir được truyền, tìm theo basename trong images_dir.
    3. Thử stem + các extension phổ biến trong images_dir.
    """
    image_ref_path = Path(image_ref)

    if image_ref_path.exists():
        return image_ref_path

    if images_dir is not None:
        images_dir = Path(images_dir)
        candidate = images_dir / image_ref_path.name
        if candidate.exists():
            return candidate

        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{image_ref_path.stem}{ext}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Không tìm thấy ảnh '{image_ref}'. images_dir='{images_dir}'"
    )


def stable_cache_name(image_path):
    """
    Tạo tên cache ổn định, tránh đụng tên khi original/augmented có cùng basename.
    """
    image_path = Path(image_path)
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{image_path.stem}_{digest}.npz"


def extract_structural_features(image_path, prior_extractor, morpho_extractor):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"cv2 không đọc được ảnh: {image_path}")

    prior_output = prior_extractor.extract_prior(image_bgr)
    morpho_output = morpho_extractor.extract_features(image_bgr)

    prior_mask = prior_output["prior_mask"].astype(np.float32)
    red_map = prior_output["red_map"].astype(np.float32)
    center_map = prior_output["center_map"].astype(np.float32)
    morpho_prior_map = prior_output["morpho_map"].astype(np.float32)

    topo_mask = morpho_output["topo_mask"].astype(np.float32)
    topo_features = morpho_output["topo_features"].astype(np.float32)
    global_features = morpho_output["global_features"].astype(np.float32)

    return {
        "prior_mask": prior_mask,
        "red_map": red_map,
        "center_map": center_map,
        "morpho_prior_map": morpho_prior_map,
        "topo_mask": topo_mask,
        "topo_features": topo_features,
        "global_features": global_features,
        "topo_feature_names": np.array(morpho_output.get("feature_names", ()), dtype=object),
        "global_feature_names": np.array(morpho_output.get("global_feature_names", ()), dtype=object),
    }


def save_npz(features, output_path, image_path, image_ref, record_index, compress=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_path,
        image_path=str(image_path),
        image_ref=str(image_ref),
        record_index=int(record_index),
        **features,
    )


def write_manifest(rows, manifest_csv):
    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_hw(value):
    if isinstance(value, tuple):
        return value
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Kích thước phải có dạng HxW, ví dụ 14x14")
    h, w = value.lower().split("x", 1)
    return int(h), int(w)


def precompute_structural_features(
    jsonl_path,
    output_dir,
    images_dir=None,
    manifest_csv=None,
    grid_size=DEFAULT_GRID_SIZE,
    image_size=DEFAULT_IMAGE_SIZE,
    max_samples=None,
    skip_existing=True,
    deduplicate_images=True,
    compress=True,
):
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy jsonl_path: {jsonl_path}")

    if manifest_csv is None:
        manifest_csv = output_dir / "structural_features_manifest.csv"
    else:
        manifest_csv = Path(manifest_csv)

    raw_records = list(read_jsonl(jsonl_path))
    if max_samples is not None:
        raw_records = raw_records[:max_samples]

    records = raw_records
    if deduplicate_images:
        unique_records = []
        seen_refs = set()
        for record_index, record in raw_records:
            image_ref = get_image_ref(record)
            key = str(Path(image_ref).name).lower() if image_ref else f"missing:{record_index}"
            if key in seen_refs:
                continue
            seen_refs.add(key)
            unique_records.append((record_index, record))
        records = unique_records

    prior_extractor = LesionPriorExtractor(
        image_size=image_size,
        grid_size=grid_size,
        use_morphology=False,
    )
    morpho_extractor = MorphologyTopologicalExtractor(
        grid_size=grid_size,
        image_size=image_size,
    )

    rows = []
    processed = 0
    skipped_existing = 0
    failed = 0

    print("\n======================================")
    print("🧬 PRECOMPUTE STRUCTURAL FEATURES")
    print("======================================")
    print(f"JSONL:        {jsonl_path}")
    print(f"Images dir:   {images_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Grid size:    {grid_size}")
    print(f"Image size:   {image_size}")
    print(f"Samples:      {len(records)}")
    if deduplicate_images:
        print(f"Raw records:   {len(raw_records)}")
        print(f"Unique images: {len(records)}")
    print(f"NPZ compress:  {compress}")
    print("--------------------------------------")

    for record_index, record in tqdm(records, desc="Structural precompute"):
        image_ref = get_image_ref(record)
        if not image_ref:
            failed += 1
            rows.append(
                {
                    "record_index": record_index,
                    "image_ref": image_ref,
                    "image_path": "",
                    "cache_path": "",
                    "status": "failed",
                    "error": "missing images field",
                }
            )
            continue

        try:
            image_path = resolve_image_path(image_ref, images_dir=images_dir)
            cache_path = output_dir / stable_cache_name(image_path)

            if skip_existing and cache_path.exists():
                skipped_existing += 1
                rows.append(
                    {
                        "record_index": record_index,
                        "image_ref": image_ref,
                        "image_path": str(image_path),
                        "cache_path": str(cache_path),
                        "status": "skipped_existing",
                        "error": "",
                    }
                )
                continue

            features = extract_structural_features(
                image_path=image_path,
                prior_extractor=prior_extractor,
                morpho_extractor=morpho_extractor,
            )
            save_npz(
                features=features,
                output_path=cache_path,
                image_path=image_path,
                image_ref=image_ref,
                record_index=record_index,
                compress=compress,
            )

            processed += 1
            rows.append(
                {
                    "record_index": record_index,
                    "image_ref": image_ref,
                    "image_path": str(image_path),
                    "cache_path": str(cache_path),
                    "status": "processed",
                    "error": "",
                }
            )

        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "record_index": record_index,
                    "image_ref": image_ref,
                    "image_path": "",
                    "cache_path": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )

    write_manifest(rows, manifest_csv)

    print("\n======================================")
    print("✅ STRUCTURAL PRECOMPUTE HOÀN TẤT")
    print("======================================")
    print(f"Tổng records:       {len(records)}")
    print(f"Processed:          {processed}")
    print(f"Skipped existing:   {skipped_existing}")
    print(f"Failed:             {failed}")
    print(f"Output dir:         {output_dir}")
    print(f"Manifest CSV:       {manifest_csv}")
    print("======================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute lesion prior + morphology topology features thành cache .npz."
    )
    parser.add_argument("--jsonl", required=True, help="JSONL cần precompute.")
    parser.add_argument("--output-dir", required=True, help="Thư mục lưu .npz structural features.")
    parser.add_argument("--images-dir", default=None, help="Fallback thư mục ảnh nếu JSONL path không tồn tại.")
    parser.add_argument("--manifest-csv", default=None, help="Path manifest CSV output.")
    parser.add_argument("--grid-size", type=parse_hw, default=DEFAULT_GRID_SIZE, help="Grid HxW, mặc định 14x14.")
    parser.add_argument("--image-size", type=parse_hw, default=DEFAULT_IMAGE_SIZE, help="Image HxW cho extractor, mặc định 224x224.")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu để test nhanh.")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè cache đã tồn tại.")
    parser.add_argument(
        "--no-deduplicate-images",
        action="store_true",
        help="Không deduplicate ảnh; xử lý theo từng record JSONL. Mặc định sẽ xử lý unique images để nhanh hơn.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Dùng np.savez thay vì np.savez_compressed để tăng tốc, đổi lại tốn dung lượng hơn.",
    )
    args = parser.parse_args()

    precompute_structural_features(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        manifest_csv=args.manifest_csv,
        grid_size=args.grid_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        skip_existing=not args.overwrite,
        deduplicate_images=not args.no_deduplicate_images,
        compress=not args.no_compress,
    )


if __name__ == "__main__":
    main()
