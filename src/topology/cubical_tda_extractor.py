"""Cubical persistent-homology extractor for endoscopic images.

This module is intentionally different from ``tda_morphology.py``.
``tda_morphology.py`` builds morphology-aware, topology-inspired descriptors.
This file computes actual cubical persistent homology on image-derived scalar
fields and exports persistence statistics for each visual patch.

Default API is compatible with the existing topology extractors:

    extractor = CubicalTDAExtractor()
    out = extractor.extract_features(image)
    out["topo_mask"]       # (14, 14)
    out["topo_features"]   # (14, 14, 12)
    out["global_features"] # (12,)

Dependency policy:
- Prefer GUDHI's CubicalComplex when installed.
- Fall back to giotto-tda's CubicalPersistence when GUDHI is unavailable.
- Raise a clear ImportError if neither package is available.

Install one of:
    pip install gudhi
    pip install giotto-tda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

ScalarMode = Literal[
    "grayscale",
    "inverted_grayscale",
    "red",
    "red_excess",
    "lab_redness",
    "lesion_score",
]


class CubicalTDAExtractor:
    """Patch-level cubical persistent homology extractor.

    The extractor computes a scalar field from the image, divides it into a
    fixed patch grid, runs cubical persistent homology on each patch, and
    summarizes H0/H1 diagrams with 12 statistics.

    Notes
    -----
    ``grayscale`` scalar field is the default for a neutral TDA baseline.
    ``lesion_score`` is available as an optional medical saliency filtration, but
    it uses color/edge heuristics before cubical PH and is therefore not the
    strictest pure-TDA setting.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "h0_count",
        "h0_total_persistence",
        "h0_max_persistence",
        "h0_mean_persistence",
        "h0_persistence_entropy",
        "h0_lifetime_std",
        "h1_count",
        "h1_total_persistence",
        "h1_max_persistence",
        "h1_mean_persistence",
        "h1_persistence_entropy",
        "h1_lifetime_std",
    )

    GLOBAL_FEATURE_NAMES: Tuple[str, ...] = (
        "global_h0_count",
        "global_h0_total_persistence",
        "global_h0_max_persistence",
        "global_h0_mean_persistence",
        "global_h0_persistence_entropy",
        "global_h0_lifetime_std",
        "global_h1_count",
        "global_h1_total_persistence",
        "global_h1_max_persistence",
        "global_h1_mean_persistence",
        "global_h1_persistence_entropy",
        "global_h1_lifetime_std",
    )

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        grid_size: Tuple[int, int] = (14, 14),
        scalar_mode: ScalarMode = "grayscale",
        homology_dimensions: Sequence[int] = (0, 1),
        min_persistence: float = 1e-6,
        backend: Literal["auto", "gudhi", "gtda"] = "auto",
        normalize_features: bool = False,
        topo_score_weights: Tuple[float, float] = (0.45, 0.55),
    ) -> None:
        self.image_size = tuple(image_size)
        self.grid_size = tuple(grid_size)
        self.scalar_mode = scalar_mode
        self.homology_dimensions = tuple(int(d) for d in homology_dimensions)
        self.min_persistence = float(min_persistence)
        self.backend = self._resolve_backend(backend)
        self.normalize_features = bool(normalize_features)
        self.topo_score_weights = tuple(float(x) for x in topo_score_weights)

        if len(self.topo_score_weights) != 2:
            raise ValueError("topo_score_weights must contain exactly two values: (H0, H1).")

        self._gtda_transformer = None
        if self.backend == "gtda":
            from gtda.homology import CubicalPersistence  # type: ignore

            self._gtda_transformer = CubicalPersistence(
                homology_dimensions=list(self.homology_dimensions),
                n_jobs=-1,
            )

    def fit_transform(self, image_path_or_img: Any) -> np.ndarray:
        """Backward-compatible API: return only the normalized topology mask."""
        return self.extract_features(image_path_or_img)["topo_mask"]

    def extract_features(self, image_path_or_img: Any) -> Dict[str, np.ndarray | Tuple[str, ...] | str]:
        """Extract patch-level and global cubical PH statistics.

        Returns
        -------
        dict
            ``topo_mask``: ``float32`` array of shape ``(grid_h, grid_w)``.
            ``topo_features``: ``float32`` array of shape ``(grid_h, grid_w, 12)``.
            ``global_features``: ``float32`` array of shape ``(12,)``.
            ``feature_names`` and ``global_feature_names``.
        """
        img = self._read_image(image_path_or_img)
        scalar_field = self._build_scalar_field(img)

        grid_h, grid_w = self.grid_size
        height, width = scalar_field.shape
        patch_h = height // grid_h
        patch_w = width // grid_w

        topo_features = np.zeros((grid_h, grid_w, len(self.FEATURE_NAMES)), dtype=np.float32)
        topo_mask = np.zeros((grid_h, grid_w), dtype=np.float32)

        for row in range(grid_h):
            for col in range(grid_w):
                y0 = row * patch_h
                x0 = col * patch_w
                y1 = height if row == grid_h - 1 else (row + 1) * patch_h
                x1 = width if col == grid_w - 1 else (col + 1) * patch_w

                patch = scalar_field[y0:y1, x0:x1]
                diagram = self._compute_diagram(patch)
                stats_h0 = self._persistence_stats(diagram, dim=0)
                stats_h1 = self._persistence_stats(diagram, dim=1)

                feature_vector = np.asarray((*stats_h0, *stats_h1), dtype=np.float32)
                topo_features[row, col] = feature_vector

                h0_weight, h1_weight = self.topo_score_weights
                topo_mask[row, col] = (
                    h0_weight * feature_vector[1]
                    + h1_weight * feature_vector[7]
                )

        if self.normalize_features:
            topo_features = self._normalize_feature_channels(topo_features)

        topo_mask = self._normalize(topo_mask).astype(np.float32)
        global_features = self._extract_global_features(scalar_field).astype(np.float32)

        return {
            "topo_mask": topo_mask,
            "topo_features": topo_features.astype(np.float32),
            "global_features": global_features,
            "feature_names": self.FEATURE_NAMES,
            "global_feature_names": self.GLOBAL_FEATURE_NAMES,
            "backend": self.backend,
            "scalar_mode": self.scalar_mode,
        }

    def _resolve_backend(self, backend: str) -> Literal["gudhi", "gtda"]:
        if backend not in {"auto", "gudhi", "gtda"}:
            raise ValueError("backend must be one of: 'auto', 'gudhi', 'gtda'.")

        if backend in {"auto", "gudhi"}:
            try:
                import gudhi  # noqa: F401  # type: ignore

                return "gudhi"
            except Exception:
                if backend == "gudhi":
                    raise ImportError("GUDHI is not installed. Install with: pip install gudhi")

        try:
            import gtda  # noqa: F401  # type: ignore

            return "gtda"
        except Exception as exc:
            raise ImportError(
                "No cubical persistence backend found. Install one of: "
                "pip install gudhi  OR  pip install giotto-tda"
            ) from exc

    def _read_image(self, image_path_or_img: Any) -> np.ndarray:
        if isinstance(image_path_or_img, (str, Path)):
            img = cv2.imread(str(image_path_or_img), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image at: {image_path_or_img}")
            return img

        if image_path_or_img is None:
            raise ValueError("Input image is None.")

        img = np.asarray(image_path_or_img)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Input image must have shape (H, W, 3).")
        return img[:, :, :3]

    def _build_scalar_field(self, img: np.ndarray) -> np.ndarray:
        """Build the scalar field used for cubical filtration.

        GUDHI and giotto-tda compute sublevel-set persistence. For lesion-like
        modes, this function returns ``1 - score`` so high-saliency regions have
        lower filtration values and appear earlier.
        """
        img_resized = cv2.resize(img, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_float)

        gray_u8 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0

        if self.scalar_mode == "grayscale":
            return self._normalize(gray).astype(np.float32)

        if self.scalar_mode == "inverted_grayscale":
            return self._normalize(1.0 - gray).astype(np.float32)

        if self.scalar_mode == "red":
            return self._normalize(1.0 - r).astype(np.float32)

        red_excess = np.clip(r - 0.5 * g - 0.5 * b, 0.0, 1.0)
        red_excess = self._normalize(red_excess)

        if self.scalar_mode == "red_excess":
            return self._normalize(1.0 - red_excess).astype(np.float32)

        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        lab_a = lab[:, :, 1].astype(np.float32) / 255.0
        lab_redness = self._normalize(np.clip((lab_a - 0.5) * 2.0, 0.0, 1.0))

        if self.scalar_mode == "lab_redness":
            return self._normalize(1.0 - lab_redness).astype(np.float32)

        if self.scalar_mode != "lesion_score":
            raise ValueError(f"Unknown scalar_mode: {self.scalar_mode}")

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0

        edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = self._normalize(np.abs(edges))

        tissue_mask = (gray > 0.08).astype(np.float32)
        tissue_mask = cv2.morphologyEx(
            tissue_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        specular_mask = ((value > 0.86) & (saturation < 0.38)).astype(np.float32)
        specular_mask = cv2.dilate(
            specular_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        lesion_score = 0.50 * red_excess + 0.40 * lab_redness + 0.10 * edges
        lesion_score *= tissue_mask
        lesion_score *= 1.0 - 0.95 * specular_mask
        lesion_score = cv2.GaussianBlur(lesion_score, (5, 5), 0)
        lesion_score = self._normalize(lesion_score)

        return self._normalize(1.0 - lesion_score).astype(np.float32)

    def _compute_diagram(self, scalar_patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(scalar_patch, dtype=np.float64)
        if self.backend == "gudhi":
            return self._compute_diagram_gudhi(patch)
        return self._compute_diagram_gtda(patch)

    def _compute_diagram_gudhi(self, scalar_patch: np.ndarray) -> np.ndarray:
        import gudhi as gd  # type: ignore

        cubical_complex = gd.CubicalComplex(
            dimensions=list(scalar_patch.shape),
            top_dimensional_cells=scalar_patch.flatten(),
        )
        persistence = cubical_complex.persistence(
            homology_coeff_field=2,
            min_persistence=self.min_persistence,
        )

        rows = []
        for dim, pair in persistence:
            if dim not in self.homology_dimensions:
                continue
            birth, death = pair
            rows.append((float(birth), float(death), float(dim)))

        if not rows:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def _compute_diagram_gtda(self, scalar_patch: np.ndarray) -> np.ndarray:
        if self._gtda_transformer is None:
            raise RuntimeError("giotto-tda transformer is not initialized.")
        patch_tensor = np.expand_dims(scalar_patch.astype(np.float32), axis=0)
        diagram = self._gtda_transformer.fit_transform(patch_tensor)[0]
        return np.asarray(diagram, dtype=np.float64)

    def _persistence_stats(self, diagram: np.ndarray, dim: int) -> Tuple[float, float, float, float, float, float]:
        """Return count, total, max, mean, entropy, and std for one dimension."""
        if diagram.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        dgm_dim = diagram[diagram[:, 2].astype(int) == int(dim)]
        if dgm_dim.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        finite = dgm_dim[np.isfinite(dgm_dim[:, 1])]
        if finite.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        lifetimes = finite[:, 1] - finite[:, 0]
        lifetimes = lifetimes[lifetimes > self.min_persistence]
        if lifetimes.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        total = float(np.sum(lifetimes))
        max_life = float(np.max(lifetimes))
        mean = float(np.mean(lifetimes))
        std = float(np.std(lifetimes))
        count = float(lifetimes.size)
        entropy = self._persistence_entropy(lifetimes)
        return count, total, max_life, mean, entropy, std

    def _extract_global_features(self, scalar_field: np.ndarray) -> np.ndarray:
        diagram = self._compute_diagram(scalar_field)
        stats_h0 = self._persistence_stats(diagram, dim=0)
        stats_h1 = self._persistence_stats(diagram, dim=1)
        return np.asarray((*stats_h0, *stats_h1), dtype=np.float32)

    def _persistence_entropy(self, lifetimes: Iterable[float]) -> float:
        lifetimes_arr = np.asarray(list(lifetimes), dtype=np.float64)
        total = float(np.sum(lifetimes_arr))
        if total <= 1e-12:
            return 0.0
        prob = lifetimes_arr / total
        return float(-np.sum(prob * np.log(prob + 1e-12)))

    def _normalize_feature_channels(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32).copy()
        for channel in range(features.shape[-1]):
            features[:, :, channel] = self._normalize(features[:, :, channel])
        return features

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        min_val = float(np.nanmin(array))
        max_val = float(np.nanmax(array))
        if max_val - min_val <= 1e-8 or np.isnan(min_val) or np.isnan(max_val):
            return np.zeros_like(array, dtype=np.float32)
        result = (array - min_val) / (max_val - min_val)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# Backward-compatible name matching existing topology modules.
TopologicalExtractor = CubicalTDAExtractor


def _json_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    topo_mask = np.asarray(result["topo_mask"])
    topo_features = np.asarray(result["topo_features"])
    global_features = np.asarray(result["global_features"])
    return {
        "backend": result.get("backend"),
        "scalar_mode": result.get("scalar_mode"),
        "topo_mask_shape": list(topo_mask.shape),
        "topo_features_shape": list(topo_features.shape),
        "global_features_shape": list(global_features.shape),
        "topo_mask_min": float(np.min(topo_mask)),
        "topo_mask_max": float(np.max(topo_mask)),
        "topo_mask_mean": float(np.mean(topo_mask)),
        "feature_names": list(result["feature_names"]),
        "global_feature_names": list(result["global_feature_names"]),
        "global_features": [float(x) for x in global_features.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cubical PH features from an image.")
    parser.add_argument("image", type=str, help="Path to an input image.")
    parser.add_argument("--out", type=str, default="", help="Optional .npz output path.")
    parser.add_argument(
        "--scalar-mode",
        choices=[
            "grayscale",
            "inverted_grayscale",
            "red",
            "red_excess",
            "lab_redness",
            "lesion_score",
        ],
        default="grayscale",
        help="Scalar field used for cubical filtration.",
    )
    parser.add_argument("--backend", choices=["auto", "gudhi", "gtda"], default="auto")
    parser.add_argument("--grid-size", type=int, nargs=2, default=(14, 14), metavar=("H", "W"))
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    parser.add_argument("--normalize-features", action="store_true")
    args = parser.parse_args()

    extractor = CubicalTDAExtractor(
        image_size=tuple(args.image_size),
        grid_size=tuple(args.grid_size),
        scalar_mode=args.scalar_mode,
        backend=args.backend,
        normalize_features=args.normalize_features,
    )
    result = extractor.extract_features(args.image)

    print(json.dumps(_json_summary(result), indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            topo_mask=result["topo_mask"],
            topo_features=result["topo_features"],
            global_features=result["global_features"],
            feature_names=np.asarray(result["feature_names"], dtype=object),
            global_feature_names=np.asarray(result["global_feature_names"], dtype=object),
            backend=np.asarray(result["backend"], dtype=object),
            scalar_mode=np.asarray(result["scalar_mode"], dtype=object),
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
