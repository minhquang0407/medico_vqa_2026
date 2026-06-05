import csv
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_TOPO_FEATURE_DIM = 12
DEFAULT_GLOBAL_FEATURE_DIM = 8


@dataclass
class MedicoVQASample:
    record_index: int
    record: Dict[str, Any]
    image_ref: str
    image_path: Path
    question_text: str
    answer_text: str


def load_jsonl_records(jsonl_path: str | Path) -> List[Tuple[int, Dict[str, Any]]]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy JSONL: {jsonl_path}")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi ở dòng {line_number} của {jsonl_path}: {exc}") from exc
    return records


def get_image_ref(record: Dict[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        return str(images[0])
    return ""


def path_basename_any(value: str | Path) -> str:
    """Return filename for both POSIX paths and Windows paths on any OS."""
    text = str(value)
    return Path(PureWindowsPath(text).name).name


def resolve_image_path(image_ref: str, images_dir: Optional[str | Path] = None) -> Path:
    if not image_ref:
        raise ValueError("Record không có field images hợp lệ.")

    image_ref_path = Path(image_ref)
    if image_ref_path.exists():
        return image_ref_path

    if images_dir is not None:
        images_dir = Path(images_dir)
        image_name = path_basename_any(image_ref)
        image_stem = Path(image_name).stem
        candidate = images_dir / image_name
        if candidate.exists():
            return candidate

        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{image_stem}{ext}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Không tìm thấy ảnh '{image_ref}'. images_dir='{images_dir}'")


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def parse_question_answer(record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Parse question/answer từ các format VQA thường gặp.

    Hỗ trợ ưu tiên:
    - messages: [{role: user, content: ...}, {role: assistant, content: ...}]
    - question/answer
    - query/response
    - conversations: [{from: human/gpt, value: ...}]
    """
    if "question" in record and "answer" in record:
        return str(record.get("question", "")).strip(), str(record.get("answer", "")).strip()

    if "query" in record and "response" in record:
        return str(record.get("query", "")).strip(), str(record.get("response", "")).strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        question = ""
        answer = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            text = _message_text(message)
            if role in {"user", "human"} and not question:
                question = text
            elif role in {"assistant", "gpt", "model"} and not answer:
                answer = text
        if question or answer:
            return question, answer

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        question = ""
        answer = ""
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from", message.get("role", ""))).lower()
            text = str(message.get("value", message.get("content", ""))).strip()
            if role in {"human", "user"} and not question:
                question = text
            elif role in {"gpt", "assistant", "model"} and not answer:
                answer = text
        if question or answer:
            return question, answer

    raise ValueError("Không parse được question/answer từ record.")


def parse_vqa_record(
    record_index: int,
    record: Dict[str, Any],
    images_dir: Optional[str | Path] = None,
) -> MedicoVQASample:
    image_ref = get_image_ref(record)
    image_path = resolve_image_path(image_ref, images_dir=images_dir)
    question_text, answer_text = parse_question_answer(record)

    return MedicoVQASample(
        record_index=record_index,
        record=record,
        image_ref=image_ref,
        image_path=image_path,
        question_text=question_text,
        answer_text=answer_text,
    )


class StructuralFeatureIndex:
    """
    Index manifest CSV sinh bởi scripts/precompute_structural_features.py.

    Manifest có các cột chính:
    - image_ref
    - image_path
    - cache_path
    - status
    """

    def __init__(self, manifest_csv: Optional[str | Path] = None):
        self.manifest_csv = Path(manifest_csv) if manifest_csv else None
        self.by_image_ref: Dict[str, Path] = {}
        self.by_image_path: Dict[str, Path] = {}
        self.by_basename: Dict[str, Path] = {}
        self.rows: List[Dict[str, str]] = []

        if self.manifest_csv is not None:
            self._load_manifest(self.manifest_csv)

    def _load_manifest(self, manifest_csv: Path) -> None:
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy structural manifest CSV: {manifest_csv}")

        with manifest_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "")
                cache_path = row.get("cache_path", "")
                if not cache_path or status not in {"processed", "skipped_existing"}:
                    continue

                cache = Path(cache_path)
                if not cache.exists():
                    remapped_cache = manifest_csv.parent / path_basename_any(cache_path)
                    if remapped_cache.exists():
                        cache = remapped_cache
                image_ref = row.get("image_ref", "")
                image_path = row.get("image_path", "")

                self.rows.append(row)
                if image_ref:
                    self.by_image_ref[self._norm_key(image_ref)] = cache
                    self.by_basename[path_basename_any(image_ref).lower()] = cache
                if image_path:
                    self.by_image_path[self._norm_key(image_path)] = cache
                    self.by_basename[path_basename_any(image_path).lower()] = cache

    @staticmethod
    def _norm_key(value: str | Path) -> str:
        return str(value).replace("\\", "/").lower()

    def find_cache_path(self, image_ref: str, image_path: str | Path) -> Optional[Path]:
        candidates = [
            self.by_image_ref.get(self._norm_key(image_ref)),
            self.by_image_path.get(self._norm_key(image_path)),
            self.by_basename.get(path_basename_any(image_ref).lower()),
            self.by_basename.get(path_basename_any(image_path).lower()),
        ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    def load_features(
        self,
        image_ref: str,
        image_path: str | Path,
        strict: bool = True,
        grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
        topo_feature_dim: int = DEFAULT_TOPO_FEATURE_DIM,
        global_feature_dim: int = DEFAULT_GLOBAL_FEATURE_DIM,
    ) -> Dict[str, Any]:
        cache_path = self.find_cache_path(image_ref, image_path)
        if cache_path is None:
            if strict:
                raise FileNotFoundError(f"Không tìm thấy structural cache cho image_ref={image_ref}")
            return self.zero_features(grid_size, topo_feature_dim, global_feature_dim, cache_path="")

        if not cache_path.exists():
            if strict:
                raise FileNotFoundError(f"Structural cache không tồn tại: {cache_path}")
            return self.zero_features(grid_size, topo_feature_dim, global_feature_dim, cache_path=str(cache_path))

        with np.load(cache_path, allow_pickle=True) as data:
            return {
                "prior_mask": data["prior_mask"].astype(np.float32),
                "red_map": data["red_map"].astype(np.float32) if "red_map" in data else None,
                "center_map": data["center_map"].astype(np.float32) if "center_map" in data else None,
                "morpho_prior_map": data["morpho_prior_map"].astype(np.float32) if "morpho_prior_map" in data else None,
                "topo_mask": data["topo_mask"].astype(np.float32),
                "topo_features": data["topo_features"].astype(np.float32),
                "global_features": data["global_features"].astype(np.float32),
                "cache_path": str(cache_path),
            }

    @staticmethod
    def zero_features(
        grid_size: Tuple[int, int],
        topo_feature_dim: int,
        global_feature_dim: int,
        cache_path: str = "",
    ) -> Dict[str, Any]:
        return {
            "prior_mask": np.zeros(grid_size, dtype=np.float32),
            "red_map": np.zeros(grid_size, dtype=np.float32),
            "center_map": np.zeros(grid_size, dtype=np.float32),
            "morpho_prior_map": np.zeros(grid_size, dtype=np.float32),
            "topo_mask": np.zeros(grid_size, dtype=np.float32),
            "topo_features": np.zeros((*grid_size, topo_feature_dim), dtype=np.float32),
            "global_features": np.zeros((global_feature_dim,), dtype=np.float32),
            "cache_path": cache_path,
        }


class MedicoVQADataset(Dataset):
    """
    PyTorch Dataset cho Medico VQA + structural features đã precompute offline.

    Dataset này chưa tokenize question/answer để giữ độc lập với backbone model.
    Tokenizer/model-specific collator có thể xử lý text ở training script.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        images_dir: Optional[str | Path] = None,
        structural_manifest_csv: Optional[str | Path] = None,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
        topo_feature_dim: int = DEFAULT_TOPO_FEATURE_DIM,
        global_feature_dim: int = DEFAULT_GLOBAL_FEATURE_DIM,
        strict_structural: bool = True,
        normalize_image: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.images_dir = Path(images_dir) if images_dir is not None else None
        self.image_size = image_size
        self.grid_size = grid_size
        self.topo_feature_dim = topo_feature_dim
        self.global_feature_dim = global_feature_dim
        self.strict_structural = bool(strict_structural)
        self.normalize_image = bool(normalize_image)

        raw_records = load_jsonl_records(self.jsonl_path)
        if max_samples is not None:
            raw_records = raw_records[:max_samples]
        self.raw_records = raw_records

        self.structural_index = StructuralFeatureIndex(structural_manifest_csv)

    def __len__(self) -> int:
        return len(self.raw_records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index, record = self.raw_records[index]
        sample = parse_vqa_record(record_index, record, images_dir=self.images_dir)

        image = self._load_image_tensor(sample.image_path)
        structural = self.structural_index.load_features(
            image_ref=sample.image_ref,
            image_path=sample.image_path,
            strict=self.strict_structural,
            grid_size=self.grid_size,
            topo_feature_dim=self.topo_feature_dim,
            global_feature_dim=self.global_feature_dim,
        )

        return {
            "image": image,
            "question_text": sample.question_text,
            "answer_text": sample.answer_text,
            "prior_mask": torch.from_numpy(structural["prior_mask"]).float(),
            "red_map": torch.from_numpy(structural["red_map"]).float() if structural["red_map"] is not None else None,
            "center_map": torch.from_numpy(structural["center_map"]).float() if structural["center_map"] is not None else None,
            "morpho_prior_map": torch.from_numpy(structural["morpho_prior_map"]).float() if structural["morpho_prior_map"] is not None else None,
            "topo_mask": torch.from_numpy(structural["topo_mask"]).float(),
            "topo_features": torch.from_numpy(structural["topo_features"]).float(),
            "global_features": torch.from_numpy(structural["global_features"]).float(),
            "image_ref": sample.image_ref,
            "image_path": str(sample.image_path),
            "cache_path": structural["cache_path"],
            "record_index": sample.record_index,
            "record": sample.record,
        }

    def _load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"cv2 không đọc được ảnh: {image_path}")

        image_bgr = cv2.resize(image_bgr, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float32)
        if self.normalize_image:
            image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float()


def _stack_optional_tensor(batch: List[Dict[str, Any]], key: str):
    values = [item[key] for item in batch]
    if any(value is None for value in values):
        return None
    return torch.stack(values, dim=0)


def medico_vqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function giữ text dạng list string và stack các tensor structural.
    """
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "question_text": [item["question_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch],
        "prior_mask": torch.stack([item["prior_mask"] for item in batch], dim=0),
        "red_map": _stack_optional_tensor(batch, "red_map"),
        "center_map": _stack_optional_tensor(batch, "center_map"),
        "morpho_prior_map": _stack_optional_tensor(batch, "morpho_prior_map"),
        "topo_mask": torch.stack([item["topo_mask"] for item in batch], dim=0),
        "topo_features": torch.stack([item["topo_features"] for item in batch], dim=0),
        "global_features": torch.stack([item["global_features"] for item in batch], dim=0),
        "image_ref": [item["image_ref"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "cache_path": [item["cache_path"] for item in batch],
        "record_index": torch.tensor([item["record_index"] for item in batch], dtype=torch.long),
        "record": [item["record"] for item in batch],
    }


def build_medico_vqa_dataloader(
    jsonl_path: str | Path,
    images_dir: Optional[str | Path] = None,
    structural_manifest_csv: Optional[str | Path] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
):
    dataset = MedicoVQADataset(
        jsonl_path=jsonl_path,
        images_dir=images_dir,
        structural_manifest_csv=structural_manifest_csv,
        **dataset_kwargs,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=medico_vqa_collate_fn,
    )
