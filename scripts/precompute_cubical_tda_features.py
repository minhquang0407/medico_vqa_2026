"""Precompute true Cubical Persistent Homology features for Medico VQA.

This script computes patch-level cubical PH features once and stores them as
``.npz`` files, so Cubical-TDA-only training can load cached features instead of
recomputing persistence diagrams inside the training loop.

Output files are compatible with the existing ``StructuralFeatureIndex`` format:
- ``prior_mask`` is a zero map for strict TDA-only use.
- ``topo_mask`` is the normalized PH-based topology mask.
- ``topo_features`` has shape ``(14, 14, 12)`` by default.
- ``global_features`` has shape ``(12,)``.

The cache naming scheme matches ``scripts/train_qwen3b_cubical_tda_only.py``.
If you precompute with the same image size, grid size, scalar mode, backend,
min-persistence, and normalization settings, the trainer can load these files
without recomputing them.

Example
-------
python scripts/precompute_cubical_tda_features.py ^
  --jsonl data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl ^
  --images-dir data/raw/Kvasir-VQA-x1/images ^
  --output-dir data/processed/cubical_tda_features ^
  --manifest-csv data/processed/cubical_tda_features/train_manifest.csv ^
  --backend gudhi ^
  --scalar-mode grayscale
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.dataset import get_image_ref, path_basename_any, resolve_image_path
from src.topology.cubical_tda_extractor import CubicalTDAExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_IMAGE_SIZE = (224, 224)
SCALAR_MODES = (
    "grayscale",
    "inverted_grayscale",
    "red",
    "red_excess",
    "lab_redness",
    "lesion_score",
)


def read_jsonl(path: str | Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL decode error at line {line_number} of {path}: {exc}") from exc


def parse_hw(value: str | Sequence[int] | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    text = str(value).lower().strip()
    if "x" not in text:
        raise argparse.ArgumentTypeError("Size must be HxW, e.g. 14x14 or 224x224")
    h, w = text.split("x", 1)
    return int(h), int(w)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.lower().strip()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def cache_path_for(
    image_path: str | Path,
    output_dir: str | Path,
    image_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    scalar_mode: str,
    backend: str,
    min_persistence: float,
    normalize_features: bool,
) -> Path:
    """Return a platform-stable cache path for a Cubical-TDA feature file.

    The same JSONL often runs on Windows for precompute and Colab/Linux for
    training. Hashing absolute paths would produce different cache filenames for
    identical images, so the key uses the image basename and extractor settings.
    """
    image_name = path_basename_any(image_path).lower()
    key_payload = {
        "image_name": image_name,
        "image_size": tuple(image_size),
        "grid_size": tuple(grid_size),
        "scalar_mode": scalar_mode,
        "backend": backend,
        "min_persistence": float(min_persistence),
        "normalize_features": bool(normalize_features),
        "feature_version": 2,
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = Path(image_name).stem or "image"
    return Path(output_dir) / scalar_mode / f"{stem}_{digest}.npz"


def extract_cubical_features(
    image_path: str | Path,
    extractor: CubicalTDAExtractor,
    grid_size: Tuple[int, int],
) -> Dict[str, Any]:
    out = extractor.extract_features(image_path)
    topo_mask = np.asarray(out["topo_mask"], dtype=np.float32)
    topo_features = np.asarray(out["topo_features"], dtype=np.float32)
    global_features = np.asarray(out["global_features"], dtype=np.float32)

    zero_map = np.zeros(grid_size, dtype=np.float32)
    return {
        # Compatibility with StructuralFeatureIndex / existing manifests.
        "prior_mask": zero_map,
        "red_map": zero_map,
        "center_map": zero_map,
        "morpho_prior_map": zero_map,
        # True cubical PH outputs.
        "topo_mask": topo_mask,
        "topo_features": topo_features,
        "global_features": global_features,
        "topo_feature_names": np.asarray(out["feature_names"], dtype=object),
        "global_feature_names": np.asarray(out["global_feature_names"], dtype=object),
        "feature_names": np.asarray(out["feature_names"], dtype=object),
        "backend": np.asarray(out.get("backend", ""), dtype=object),
        "scalar_mode": np.asarray(out.get("scalar_mode", ""), dtype=object),
    }


def save_npz(
    features: Dict[str, Any],
    output_path: str | Path,
    image_path: str | Path,
    image_ref: str,
    record_index: int,
    config: Dict[str, Any],
    compress: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    save_fn = np.savez_compressed if compress else np.savez
    with tmp_path.open("wb") as f:
        save_fn(
            f,
            image_path=str(image_path),
            image_ref=str(image_ref),
            record_index=int(record_index),
            extractor="cubical_persistent_homology",
            config_json=json.dumps(config, ensure_ascii=False, sort_keys=True),
            **features,
        )
    tmp_path.replace(output_path)


def write_manifest(rows: List[Dict[str, Any]], manifest_csv: str | Path) -> None:
    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_records(
    records: List[Tuple[int, Dict[str, Any]]],
    images_dir: str | Path | None,
) -> List[Tuple[int, Dict[str, Any]]]:
    unique: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    for record_index, record in records:
        image_ref = get_image_ref(record)
        if not image_ref:
            key = f"missing:{record_index}"
        else:
            try:
                image_path = resolve_image_path(image_ref, images_dir=images_dir)
                key = str(image_path.resolve()).replace("\\", "/").lower()
            except Exception:
                key = path_basename_any(image_ref).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((record_index, record))
    return unique


def precompute_cubical_tda_features(
    jsonl_path: str | Path,
    output_dir: str | Path,
    images_dir: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    scalar_mode: str = "grayscale",
    backend: str = "auto",
    min_persistence: float = 1e-6,
    normalize_features: bool = False,
    max_samples: int | None = None,
    skip_existing: bool = True,
    deduplicate_images: bool = True,
    compress: bool = True,
) -> None:
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_csv is None:
        manifest_csv = output_dir / f"{jsonl_path.stem}_{scalar_mode}_manifest.csv"
    else:
        manifest_csv = Path(manifest_csv)

    raw_records = list(read_jsonl(jsonl_path))
    if max_samples is not None:
        raw_records = raw_records[:max_samples]
    records = deduplicate_records(raw_records, images_dir) if deduplicate_images else raw_records

    config = {
        "image_size": tuple(image_size),
        "grid_size": tuple(grid_size),
        "scalar_mode": scalar_mode,
        "backend_requested": backend,
        "min_persistence": float(min_persistence),
        "normalize_features": bool(normalize_features),
        "feature_names": CubicalTDAExtractor.FEATURE_NAMES,
        "global_feature_names": CubicalTDAExtractor.GLOBAL_FEATURE_NAMES,
        "feature_version": 1,
    }

    extractor = CubicalTDAExtractor(
        image_size=image_size,
        grid_size=grid_size,
        scalar_mode=scalar_mode,
        backend=backend,
        min_persistence=min_persistence,
        normalize_features=normalize_features,
    )
    config["backend_resolved"] = extractor.backend

    rows: List[Dict[str, Any]] = []
    processed = 0
    skipped_existing = 0
    failed = 0

    print("\n======================================")
    print("🧊 PRECOMPUTE CUBICAL TDA FEATURES")
    print("======================================")
    print(f"JSONL:          {jsonl_path}")
    print(f"Images dir:     {images_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Manifest CSV:   {manifest_csv}")
    print(f"Grid size:      {grid_size}")
    print(f"Image size:     {image_size}")
    print(f"Scalar mode:    {scalar_mode}")
    print(f"Backend:        requested={backend} resolved={extractor.backend}")
    print(f"Min persistence:{min_persistence}")
    print(f"Normalize feat: {normalize_features}")
    print(f"Samples:        {len(records)}")
    if deduplicate_images:
        print(f"Raw records:    {len(raw_records)}")
        print(f"Unique images:  {len(records)}")
    print(f"NPZ compress:   {compress}")
    print("--------------------------------------")

    for record_index, record in tqdm(records, desc="Cubical TDA precompute"):
        image_ref = get_image_ref(record)
        base_row = {
            "record_index": record_index,
            "image_ref": image_ref,
            "image_path": "",
            "cache_path": "",
            "status": "failed",
            "scalar_mode": scalar_mode,
            "backend": extractor.backend,
            "topo_features_shape": "",
            "global_features_shape": "",
            "error": "",
        }

        if not image_ref:
            failed += 1
            base_row["error"] = "missing images field"
            rows.append(base_row)
            continue

        try:
            image_path = resolve_image_path(image_ref, images_dir=images_dir)
            cache_path = cache_path_for(
                image_path=image_path,
                output_dir=output_dir,
                image_size=image_size,
                grid_size=grid_size,
                scalar_mode=scalar_mode,
                backend=backend,
                min_persistence=min_persistence,
                normalize_features=normalize_features,
            )

            base_row.update(
                {
                    "image_path": str(image_path),
                    "cache_path": str(cache_path),
                }
            )

            if skip_existing and cache_path.exists():
                skipped_existing += 1
                try:
                    with np.load(cache_path, allow_pickle=True) as data:
                        base_row["topo_features_shape"] = str(tuple(data["topo_features"].shape))
                        base_row["global_features_shape"] = str(tuple(data["global_features"].shape))
                except Exception:
                    pass
                base_row["status"] = "skipped_existing"
                rows.append(base_row)
                continue

            features = extract_cubical_features(
                image_path=image_path,
                extractor=extractor,
                grid_size=grid_size,
            )
            save_npz(
                features=features,
                output_path=cache_path,
                image_path=image_path,
                image_ref=image_ref,
                record_index=record_index,
                config=config,
                compress=compress,
            )

            processed += 1
            base_row.update(
                {
                    "status": "processed",
                    "topo_features_shape": str(tuple(features["topo_features"].shape)),
                    "global_features_shape": str(tuple(features["global_features"].shape)),
                }
            )
            rows.append(base_row)

        except Exception as exc:
            failed += 1
            base_row["error"] = repr(exc)
            rows.append(base_row)

    write_manifest(rows, manifest_csv)
    metadata_path = Path(manifest_csv).with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "jsonl_path": str(jsonl_path),
                "images_dir": str(images_dir),
                "output_dir": str(output_dir),
                "manifest_csv": str(manifest_csv),
                "config": config,
                "num_raw_records": len(raw_records),
                "num_records_processed_loop": len(records),
                "processed": processed,
                "skipped_existing": skipped_existing,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n======================================")
    print("✅ CUBICAL TDA PRECOMPUTE DONE")
    print("======================================")
    print(f"Records looped:      {len(records)}")
    print(f"Processed:           {processed}")
    print(f"Skipped existing:    {skipped_existing}")
    print(f"Failed:              {failed}")
    print(f"Output dir:          {output_dir}")
    print(f"Manifest CSV:        {manifest_csv}")
    print(f"Metadata JSON:       {metadata_path}")
    print("======================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute true Cubical PH features for Medico VQA images.")
    parser.add_argument("--jsonl", required=True, help="JSONL split to precompute.")
    parser.add_argument("--images-dir", default=None, help="Fallback image directory.")
    parser.add_argument("--output-dir", default="data/processed/cubical_tda_features", help="Root output cache directory.")
    parser.add_argument("--manifest-csv", default=None, help="Manifest CSV output path.")
    parser.add_argument("--grid-size", type=parse_hw, default=DEFAULT_GRID_SIZE, help="Grid HxW, default 14x14.")
    parser.add_argument("--image-size", type=parse_hw, default=DEFAULT_IMAGE_SIZE, help="Image HxW, default 224x224.")
    parser.add_argument("--scalar-mode", choices=SCALAR_MODES, default="grayscale")
    parser.add_argument("--backend", choices=["auto", "gudhi", "gtda"], default="auto")
    parser.add_argument("--min-persistence", type=float, default=1e-6)
    parser.add_argument("--normalize-features", type=str2bool, default=False)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument(
        "--no-deduplicate-images",
        action="store_true",
        help="Process every JSONL row instead of unique images only.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for speed at larger disk cost.",
    )
    args = parser.parse_args()

    precompute_cubical_tda_features(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        manifest_csv=args.manifest_csv,
        grid_size=args.grid_size,
        image_size=args.image_size,
        scalar_mode=args.scalar_mode,
        backend=args.backend,
        min_persistence=args.min_persistence,
        normalize_features=args.normalize_features,
        max_samples=args.max_samples,
        skip_existing=not args.overwrite,
        deduplicate_images=not args.no_deduplicate_images,
        compress=not args.no_compress,
    )


if __name__ == "__main__":
    main()
