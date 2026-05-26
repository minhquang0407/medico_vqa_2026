import cv2
import numpy as np

from src.topology.tda_morphology import TopologicalExtractor as MorphologyTopologicalExtractor


class LesionPriorExtractor:
    """
    Trích xuất lesion spatial prior ở mức patch 14x14.

    Module này tách vai trò "nhìn vào đâu" khỏi TDA feature enrichment:
    - prior_mask: spatial prior chính để modulate ViT tokens.
    - red_map: redness/color prior.
    - center_map: center-location prior, mạnh trên Kvasir-SEG nhưng cần cẩn thận khi generalize.
    - morpho_map: morphology-aware topology map từ tda_morphology.py.

    Theo grid search hiện tại trên Kvasir-SEG, cấu hình tốt nhất theo mean best-IoU là:
        prior = 0.45 * red_map + 0.55 * center_map + 0.00 * morpho_map

    Nếu muốn dùng morphology/TDA rất nhẹ để tăng ranking/AUC, có thể thử:
        weights={"red": 0.35, "center": 0.60, "morpho": 0.05}
    """

    DEFAULT_WEIGHTS = {
        "red": 0.45,
        "center": 0.55,
        "morpho": 0.0,
    }

    FEATURE_NAMES = (
        "prior_mask",
        "red_map",
        "center_map",
        "morpho_map",
    )

    def __init__(
        self,
        image_size=(224, 224),
        grid_size=(14, 14),
        weights=None,
        center_sigma=0.35,
        use_morphology=True,
    ):
        self.image_size = image_size
        self.grid_size = grid_size
        self.center_sigma = float(center_sigma)
        self.weights = self._normalize_weights(weights or self.DEFAULT_WEIGHTS)
        self.use_morphology = bool(use_morphology)
        self.morphology_extractor = (
            MorphologyTopologicalExtractor(grid_size=grid_size, image_size=image_size)
            if self.use_morphology
            else None
        )

    def fit_transform(self, image_path_or_img):
        """Tương thích với các script cũ: chỉ trả về prior_mask."""
        return self.extract_prior(image_path_or_img)["prior_mask"]

    def extract_prior(self, image_path_or_img):
        """
        Trả về dict đầy đủ để dùng trong fusion hoặc debug.

        Returns:
            {
                "prior_mask": np.ndarray (grid_h, grid_w),
                "red_map": np.ndarray (grid_h, grid_w),
                "center_map": np.ndarray (grid_h, grid_w),
                "morpho_map": np.ndarray (grid_h, grid_w),
                "weights": dict,
                "morphology_output": dict | None,
            }
        """
        img = self._read_image(image_path_or_img)

        red_map = self._build_red_excess_map(img)
        center_map = self._build_center_prior()

        morphology_output = None
        if self.morphology_extractor is not None:
            morphology_output = self.morphology_extractor.extract_features(img)
            morpho_map = self._normalize(morphology_output["topo_mask"])
        else:
            morpho_map = np.zeros(self.grid_size, dtype=np.float32)

        prior_mask = self._combine_maps(red_map, center_map, morpho_map)

        return {
            "prior_mask": prior_mask.astype(np.float32),
            "red_map": red_map.astype(np.float32),
            "center_map": center_map.astype(np.float32),
            "morpho_map": morpho_map.astype(np.float32),
            "weights": dict(self.weights),
            "morphology_output": morphology_output,
        }

    def extract_features(self, image_path_or_img):
        """
        API dạng feature extractor để dễ nối vào pipeline.

        prior_features có shape (grid_h, grid_w, 4), gồm:
        [prior_mask, red_map, center_map, morpho_map]
        """
        output = self.extract_prior(image_path_or_img)
        prior_features = np.stack(
            [
                output["prior_mask"],
                output["red_map"],
                output["center_map"],
                output["morpho_map"],
            ],
            axis=-1,
        ).astype(np.float32)

        global_features = self._extract_global_features(output)

        return {
            **output,
            "prior_features": prior_features,
            "global_features": global_features,
            "feature_names": self.FEATURE_NAMES,
            "global_feature_names": self._global_feature_names(),
        }

    def _combine_maps(self, red_map, center_map, morpho_map):
        prior = (
            self.weights["red"] * red_map
            + self.weights["center"] * center_map
            + self.weights["morpho"] * morpho_map
        )
        return self._normalize(prior)

    def _build_red_excess_map(self, img):
        img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_float)

        red_excess = np.clip(r - 0.5 * g - 0.5 * b, 0.0, 1.0)
        red_patch = cv2.resize(
            red_excess,
            self.grid_size[::-1],
            interpolation=cv2.INTER_AREA,
        )
        return self._normalize(red_patch)

    def _build_center_prior(self):
        grid_h, grid_w = self.grid_size
        ys, xs = np.mgrid[0:grid_h, 0:grid_w]
        cy = (grid_h - 1) / 2.0
        cx = (grid_w - 1) / 2.0
        dist2 = ((ys - cy) / grid_h) ** 2 + ((xs - cx) / grid_w) ** 2
        prior = np.exp(-dist2 / (2.0 * self.center_sigma ** 2))
        return self._normalize(prior)

    def _extract_global_features(self, output):
        prior = output["prior_mask"]
        red = output["red_map"]
        center = output["center_map"]
        morpho = output["morpho_map"]

        return np.array(
            [
                float(np.mean(prior)),
                float(np.max(prior)),
                self._entropy(prior),
                self._concentration(prior),
                float(np.mean(red)),
                float(np.mean(center)),
                float(np.mean(morpho)),
                float(np.max(morpho)),
                float(self.weights["red"]),
                float(self.weights["center"]),
                float(self.weights["morpho"]),
            ],
            dtype=np.float32,
        )

    def _global_feature_names(self):
        return (
            "prior_mean",
            "prior_max",
            "prior_entropy",
            "prior_concentration",
            "red_mean",
            "center_mean",
            "morpho_mean",
            "morpho_max",
            "weight_red",
            "weight_center",
            "weight_morpho",
        )

    def _read_image(self, image_path_or_img):
        if isinstance(image_path_or_img, str):
            img = cv2.imread(image_path_or_img)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path_or_img}")
            return img

        if image_path_or_img is None:
            raise ValueError("Input image is None. Hãy kiểm tra lại đường dẫn ảnh.")

        img = np.asarray(image_path_or_img)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Input image phải có shape (H, W, 3).")
        return img

    def _normalize_weights(self, weights):
        normalized = {
            "red": float(weights.get("red", 0.0)),
            "center": float(weights.get("center", 0.0)),
            "morpho": float(weights.get("morpho", 0.0)),
        }
        total = sum(max(0.0, value) for value in normalized.values())
        if total <= 1e-8:
            return dict(self.DEFAULT_WEIGHTS)
        return {key: max(0.0, value) / total for key, value in normalized.items()}

    def _entropy(self, score_map):
        flat = np.asarray(score_map, dtype=np.float32).flatten()
        total = float(np.sum(flat))
        if total <= 1e-8:
            return 0.0
        prob = flat / total
        return float(-np.sum(prob * np.log(prob + 1e-8)))

    def _concentration(self, score_map, ratio=0.10):
        flat = np.asarray(score_map, dtype=np.float32).flatten()
        total = float(np.sum(flat))
        if total <= 1e-8:
            return 0.0
        sorted_values = np.sort(flat)[::-1]
        top_k = max(1, int(np.ceil(ratio * len(sorted_values))))
        return float(np.sum(sorted_values[:top_k]) / total)

    def _normalize(self, array):
        array = np.asarray(array, dtype=np.float32)
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        if max_val - min_val <= 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)
