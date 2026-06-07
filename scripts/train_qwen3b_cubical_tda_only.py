"""Train Qwen3B with true Cubical TDA only.

This entry point is a strict ablation/training wrapper around
``scripts/train_qwen3b_curriculum_topo_adapter.py``. It replaces the structural
feature dataset with an on-the-fly/cache-backed Cubical Persistent Homology
extractor and forces the model to use only cubical PH patch features for:

1. Visual TDA Fusion: ViT patch tokens + cubical ``topo_features``.
2. Decoder TDA Adapter: mean/std/max aggregation of cubical ``topo_features``.

It deliberately disables handcrafted prior masks, morphology priors, global
structural tokens, OT prefix fusion, and auxiliary topology losses. The training
objective is therefore the language-model loss plus the optional decoder gate
regularizer from the TopoAdapter wrapper.

Example
-------
python scripts/train_qwen3b_cubical_tda_only.py ^
  --mode train_eval ^
  --epochs 1 ^
  --max-samples 128 ^
  --cubical-backend auto ^
  --cubical-scalar-mode grayscale

Notes
-----
- Precompute first with ``scripts/precompute_cubical_tda_features.py``.
- By default, training requires cache files and will not recompute Cubical PH.
- Pass ``--cubical-require-cache false`` only for debugging/on-the-fly fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Make project imports work when the script is launched from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_qwen3b_curriculum_topo_adapter as base_trainer
from src.data_pipeline.dataset import MedicoVQADataset as BaseMedicoVQADataset
from src.topology.cubical_tda_extractor import CubicalTDAExtractor


SCALAR_MODES = (
    "grayscale",
    "inverted_grayscale",
    "red",
    "red_excess",
    "lab_redness",
    "lesion_score",
)


class CubicalTDAMedicoVQADataset(BaseMedicoVQADataset):
    """Medico VQA dataset that overwrites structural tensors with cubical PH.

    The parent dataset is still used for JSONL parsing and image tensor loading,
    but all manifest-based structural features are ignored. This class computes
    or loads cubical PH features from a dedicated cache.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        images_dir: str | Path | None = None,
        structural_manifest_csv: str | Path | None = None,
        image_size: Tuple[int, int] = (224, 224),
        grid_size: Tuple[int, int] = (14, 14),
        topo_feature_dim: int = 12,
        global_feature_dim: int = 12,
        strict_structural: bool = False,
        normalize_image: bool = True,
        max_samples: int | None = None,
    ):
        del structural_manifest_csv, topo_feature_dim, global_feature_dim, strict_structural
        super().__init__(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            structural_manifest_csv=None,
            image_size=image_size,
            grid_size=grid_size,
            topo_feature_dim=len(CubicalTDAExtractor.FEATURE_NAMES),
            global_feature_dim=len(CubicalTDAExtractor.GLOBAL_FEATURE_NAMES),
            strict_structural=False,
            normalize_image=normalize_image,
            max_samples=max_samples,
        )
        self.cubical_settings = dict(_CUBICAL_SETTINGS)
        self._extractor: CubicalTDAExtractor | None = None
        self._cache_dir = Path(self.cubical_settings["cache_dir"])
        if self.cubical_settings["use_cache"]:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        result, cache_path = self._load_or_compute_cubical(item["image_path"])

        topo_mask = np.asarray(result["topo_mask"], dtype=np.float32)
        topo_features = np.asarray(result["topo_features"], dtype=np.float32)
        global_features = np.asarray(result["global_features"], dtype=np.float32)

        # Strict cubical-TDA-only policy:
        # - Visual branch receives only topo_features because visual_structural_mode
        #   is forced to tda_only below.
        # - prior_mask is zeroed so no handcrafted prior can leak into OT/fusion.
        # - global_features are present for API compatibility, but global structural
        #   tokens/losses are forced off by CLI arguments below.
        item["prior_mask"] = torch.zeros_like(torch.from_numpy(topo_mask)).float()
        item["topo_mask"] = torch.from_numpy(topo_mask).float()
        item["topo_features"] = torch.from_numpy(topo_features).float()
        item["global_features"] = torch.from_numpy(global_features).float()
        item["cache_path"] = str(cache_path) if cache_path is not None else "cubical:on_the_fly"
        item["cubical_backend"] = result.get("backend", "")
        item["cubical_scalar_mode"] = result.get("scalar_mode", self.cubical_settings["scalar_mode"])
        return item

    def _get_extractor(self) -> CubicalTDAExtractor:
        if self._extractor is None:
            self._extractor = CubicalTDAExtractor(
                image_size=tuple(self.image_size),
                grid_size=tuple(self.grid_size),
                scalar_mode=self.cubical_settings["scalar_mode"],
                backend=self.cubical_settings["backend"],
                min_persistence=float(self.cubical_settings["min_persistence"]),
                normalize_features=bool(self.cubical_settings["normalize_features"]),
            )
        return self._extractor

    def _cache_image_name(self, image_path: str | Path) -> str:
        """Return a platform-stable image filename for cache keys.

        JSONL files may contain Windows paths while Colab resolves the same image
        under ``/content``. Hashing absolute paths makes identical images produce
        different cache names across machines, so cache keys use only the image
        basename plus the extraction configuration.
        """
        return Path(PureWindowsPath(str(image_path)).name).name.lower()

    def _cache_path_for(self, image_path: str | Path) -> Path:
        image_name = self._cache_image_name(image_path)
        key_payload = {
            "image_name": image_name,
            "image_size": tuple(self.image_size),
            "grid_size": tuple(self.grid_size),
            "scalar_mode": self.cubical_settings["scalar_mode"],
            "backend": self.cubical_settings["backend"],
            "min_persistence": float(self.cubical_settings["min_persistence"]),
            "normalize_features": bool(self.cubical_settings["normalize_features"]),
            "feature_version": 2,
        }
        digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        stem = Path(image_name).stem or "image"
        mode = str(self.cubical_settings["scalar_mode"])
        return self._cache_dir / mode / f"{stem}_{digest}.npz"

    def _cache_candidates_for(self, image_path: str | Path, primary_path: Path) -> List[Path]:
        """Return primary cache path plus legacy basename matches.

        Legacy caches created before feature_version=2 were path-hashed, so the
        digest differs between Windows and Colab. The stem is still the image
        basename; if the new portable key is absent, load an existing
        ``<stem>_*.npz`` file from the same scalar-mode directory.
        """
        candidates = [primary_path]
        stem = Path(self._cache_image_name(image_path)).stem or "image"
        if primary_path.parent.exists():
            for candidate in sorted(primary_path.parent.glob(f"{stem}_*.npz")):
                if candidate != primary_path and candidate.is_file():
                    candidates.append(candidate)
        return candidates

    def _read_cache_file(self, cache_path: Path) -> Dict[str, Any]:
        with np.load(cache_path, allow_pickle=True) as data:
            return {
                "topo_mask": data["topo_mask"].astype(np.float32),
                "topo_features": data["topo_features"].astype(np.float32),
                "global_features": data["global_features"].astype(np.float32),
                "feature_names": tuple(str(x) for x in data["feature_names"].tolist()),
                "global_feature_names": tuple(str(x) for x in data["global_feature_names"].tolist()),
                "backend": str(data["backend"].item()) if "backend" in data else self.cubical_settings["backend"],
                "scalar_mode": str(data["scalar_mode"].item()) if "scalar_mode" in data else self.cubical_settings["scalar_mode"],
            }

    def _load_or_compute_cubical(self, image_path: str | Path) -> Tuple[Dict[str, Any], Path | None]:
        cache_path = self._cache_path_for(image_path)
        if self.cubical_settings["use_cache"]:
            for candidate in self._cache_candidates_for(image_path, cache_path):
                if candidate.exists():
                    return self._read_cache_file(candidate), candidate

        if self.cubical_settings["require_cache"]:
            raise FileNotFoundError(
                "Missing precomputed Cubical TDA cache: "
                f"{cache_path}. Run scripts/precompute_cubical_tda_features.py "
                "with matching --output-dir/--scalar-mode/--backend/"
                "--min-persistence/--normalize-features settings, or pass "
                "--cubical-require-cache false for on-the-fly debugging."
            )

        result = self._get_extractor().extract_features(image_path)
        if self.cubical_settings["use_cache"]:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp_path.open("wb") as f:
                np.savez_compressed(
                    f,
                    topo_mask=result["topo_mask"],
                    topo_features=result["topo_features"],
                    global_features=result["global_features"],
                    feature_names=np.asarray(result["feature_names"], dtype=object),
                    global_feature_names=np.asarray(result["global_feature_names"], dtype=object),
                    backend=np.asarray(result.get("backend", self.cubical_settings["backend"]), dtype=object),
                    scalar_mode=np.asarray(result.get("scalar_mode", self.cubical_settings["scalar_mode"]), dtype=object),
                )
            tmp_path.replace(cache_path)
            return result, cache_path
        return result, None


_CUBICAL_SETTINGS: Dict[str, Any] = {
    "cache_dir": "data/processed/cubical_tda_features",
    "use_cache": True,
    "require_cache": True,
    "scalar_mode": "grayscale",
    "backend": "auto",
    "min_persistence": 1e-6,
    "normalize_features": False,
}


def _custom_bool(value: str | bool) -> bool:
    return base_trainer.str2bool(value)


def _parse_custom_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Strict Cubical-PH-only Qwen3B Visual TDA + TDA Adapter trainer.",
        epilog=(
            "All unrecognized arguments are forwarded to "
            "scripts/train_qwen3b_curriculum_topo_adapter.py."
        ),
    )
    parser.add_argument("--cubical-cache-dir", default="data/processed/cubical_tda_features")
    parser.add_argument("--cubical-use-cache", type=_custom_bool, default=True)
    parser.add_argument(
        "--cubical-require-cache",
        type=_custom_bool,
        default=True,
        help="If true, training only loads precomputed Cubical TDA caches and fails if a cache file is missing.",
    )
    parser.add_argument("--cubical-backend", choices=["auto", "gudhi", "gtda"], default="auto")
    parser.add_argument("--cubical-scalar-mode", choices=SCALAR_MODES, default="grayscale")
    parser.add_argument("--cubical-min-persistence", type=float, default=1e-6)
    parser.add_argument("--cubical-normalize-features", type=_custom_bool, default=False)
    return parser.parse_known_args(argv)


def _remove_arg(argv: List[str], name: str) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == name:
            i += 2
            continue
        if token.startswith(name + "="):
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def _force_arg(argv: List[str], name: str, value: str) -> List[str]:
    argv = _remove_arg(argv, name)
    return [*argv, name, value]


def _ensure_arg(argv: List[str], name: str, value: str) -> List[str]:
    if any(token == name or token.startswith(name + "=") for token in argv):
        return argv
    return [*argv, name, value]


def _build_forwarded_argv(remaining: List[str]) -> List[str]:
    forwarded = list(remaining)

    # Default output path for this ablation, unless the user explicitly set one.
    forwarded = _ensure_arg(forwarded, "--output-dir", "outputs/qwen3b_cubical_tda_only")

    # Force strict Cubical-TDA-only model behavior. These replace user-supplied
    # conflicting flags because this script is specifically for this ablation.
    forced_flags = {
        "--topo-mode": "tda_only",
        "--visual-structural-mode": "tda_only",
        "--zero-prior-mask": "true",
        "--zero-global-features": "true",
        "--use-global-structural-token": "false",
        "--use-base-ot": "false",
        "--use-base-ot-fusion": "false",
        "--use-base-prior-as-ot-target": "false",
        "--use-base-topological-loss": "false",
        "--use-base-prior-align-loss": "false",
        "--use-base-global-topo-loss": "false",
        "--use-base-patch-topo-loss": "false",
    }
    for name, value in forced_flags.items():
        forwarded = _force_arg(forwarded, name, value)

    return forwarded


def main() -> None:
    custom_args, remaining = _parse_custom_args(sys.argv[1:])
    _CUBICAL_SETTINGS.update(
        {
            "cache_dir": custom_args.cubical_cache_dir,
            "use_cache": bool(custom_args.cubical_use_cache),
            "require_cache": bool(custom_args.cubical_require_cache),
            "scalar_mode": custom_args.cubical_scalar_mode,
            "backend": custom_args.cubical_backend,
            "min_persistence": float(custom_args.cubical_min_persistence),
            "normalize_features": bool(custom_args.cubical_normalize_features),
        }
    )

    # Monkeypatch the trainer's dataset symbol. The rest of the trainer remains
    # unchanged, including Qwen3B LoRA, Visual TDA Fusion, and TopoAdapter logic.
    base_trainer.MedicoVQADataset = CubicalTDAMedicoVQADataset

    forwarded = _build_forwarded_argv(remaining)
    sys.argv = [sys.argv[0], *forwarded]

    print("\n======================================")
    print("🧊 QWEN3B CUBICAL TDA-ONLY TRAINING")
    print("======================================")
    print("Cubical settings:", json.dumps(_CUBICAL_SETTINGS, ensure_ascii=False, indent=2))
    print("Forced model settings:")
    print("  topo_mode=tda_only")
    print("  visual_structural_mode=tda_only")
    print("  prior/global/OT/topology auxiliary losses disabled")
    print("Forwarded trainer argv:", " ".join(sys.argv[1:]))
    print("======================================\n")

    base_trainer.main()


if __name__ == "__main__":
    main()
