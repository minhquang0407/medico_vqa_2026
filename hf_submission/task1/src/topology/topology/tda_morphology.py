import cv2
import numpy as np


class TopologicalExtractor:
    """
    Morphology-aware Topological Prior cho ảnh nội soi.

    Đây không phải Cubical Persistent Homology thuần. Module này tạo prior hình thái
    dựa trên level-set/morphology domain knowledge và trả về API giống TDA extractor:
    - topo_mask: (grid_h, grid_w)
    - topo_features: (grid_h, grid_w, 12)
    - global_features: (8,)
    """

    FEATURE_NAMES = (
        "morph_score",
        "red_gray_energy",
        "red_excess_mean",
        "lab_redness_mean",
        "edge_mean",
        "edge_std",
        "local_contrast",
        "specular_ratio",
        "tissue_ratio",
        "h0_like_components",
        "h1_like_holes",
        "patch_entropy",
    )

    GLOBAL_FEATURE_NAMES = (
        "global_morph_energy",
        "global_morph_max",
        "global_red_excess_mean",
        "global_lab_redness_mean",
        "global_edge_mean",
        "global_specular_ratio",
        "topo_entropy",
        "topo_concentration",
    )

    def __init__(self, grid_size=(14, 14), image_size=(224, 224)):
        self.grid_size = grid_size
        self.image_size = image_size

    def fit_transform(self, image_path_or_img) -> np.ndarray:
        """Tương thích ngược: chỉ trả về morphology/topology mask."""
        return self.extract_features(image_path_or_img)["topo_mask"]

    def extract_features(self, image_path_or_img) -> dict:
        img = self._read_image(image_path_or_img)
        maps = self._build_morphology_maps(img)

        topo_mask = cv2.resize(
            maps["morph_score"],
            self.grid_size[::-1],
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        topo_mask = self._normalize(np.power(topo_mask, 3.0))

        topo_features = self._extract_patch_features(maps)
        global_features = self._extract_global_features(topo_mask, topo_features)

        return {
            "topo_mask": topo_mask.astype(np.float32),
            "topo_features": topo_features.astype(np.float32),
            "global_features": global_features.astype(np.float32),
            "feature_names": self.FEATURE_NAMES,
            "global_feature_names": self.GLOBAL_FEATURE_NAMES,
        }

    def _read_image(self, image_path_or_img):
        if isinstance(image_path_or_img, str):
            img = cv2.imread(image_path_or_img)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh tại: {image_path_or_img}")
            return img

        if image_path_or_img is None:
            raise ValueError("Input image is None. Hãy kiểm tra lại đường dẫn ảnh.")

        img = np.asarray(image_path_or_img)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Input image phải có shape (H, W, 3).")
        return img

    def _build_morphology_maps(self, img):
        img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_float)

        gray_u8 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0

        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        lab_a = lab[:, :, 1].astype(np.float32) / 255.0
        lab_redness = self._normalize(np.clip((lab_a - 0.5) * 2.0, 0.0, 1.0))

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0

        red_gray_energy = cv2.addWeighted(gray, 0.30, r, 0.70, 0.0)
        red_excess = self._normalize(np.clip(r - 0.5 * g - 0.5 * b, 0.0, 1.0))

        smoothed_energy = cv2.GaussianBlur(red_gray_energy, (25, 25), 0)
        smoothed_energy = self._normalize(np.power(smoothed_energy, 3.0))

        edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = self._normalize(np.abs(edges))

        local_mean = cv2.GaussianBlur(gray, (15, 15), 0)
        local_contrast = self._normalize(np.abs(gray - local_mean))

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

        morph_score = (
            0.50 * smoothed_energy
            + 0.25 * red_excess
            + 0.20 * lab_redness
            + 0.05 * local_contrast
        )
        morph_score *= tissue_mask
        morph_score *= 1.0 - 0.85 * specular_mask
        morph_score = cv2.GaussianBlur(morph_score, (5, 5), 0)
        morph_score = self._normalize(morph_score)

        return {
            "morph_score": morph_score,
            "red_gray_energy": self._normalize(red_gray_energy),
            "red_excess": red_excess,
            "lab_redness": lab_redness,
            "edges": edges,
            "local_contrast": local_contrast,
            "specular_mask": specular_mask,
            "tissue_mask": tissue_mask,
        }

    def _extract_patch_features(self, maps):
        grid_h, grid_w = self.grid_size
        height, width = maps["morph_score"].shape
        patch_h = height // grid_h
        patch_w = width // grid_w

        features = np.zeros((grid_h, grid_w, len(self.FEATURE_NAMES)), dtype=np.float32)

        for row in range(grid_h):
            for col in range(grid_w):
                y0 = row * patch_h
                x0 = col * patch_w
                y1 = height if row == grid_h - 1 else (row + 1) * patch_h
                x1 = width if col == grid_w - 1 else (col + 1) * patch_w

                patch_slices = {name: value[y0:y1, x0:x1] for name, value in maps.items()}
                morph_patch = patch_slices["morph_score"]
                edge_patch = patch_slices["edges"]

                binary = (morph_patch >= max(0.35, float(np.percentile(morph_patch, 75)))).astype(np.uint8)
                h0_like = float(self._count_components(binary))
                h1_like = float(self._count_holes(binary))

                features[row, col] = np.array(
                    [
                        float(np.mean(morph_patch)),
                        float(np.mean(patch_slices["red_gray_energy"])),
                        float(np.mean(patch_slices["red_excess"])),
                        float(np.mean(patch_slices["lab_redness"])),
                        float(np.mean(edge_patch)),
                        float(np.std(edge_patch)),
                        float(np.mean(patch_slices["local_contrast"])),
                        float(np.mean(patch_slices["specular_mask"])),
                        float(np.mean(patch_slices["tissue_mask"])),
                        h0_like,
                        h1_like,
                        self._entropy(morph_patch),
                    ],
                    dtype=np.float32,
                )

        return features

    def _extract_global_features(self, topo_mask, topo_features):
        mask_flat = topo_mask.flatten().astype(np.float32)
        mask_sum = float(np.sum(mask_flat))

        if mask_sum <= 1e-8:
            topo_entropy = 0.0
            topo_concentration = 0.0
        else:
            prob = mask_flat / mask_sum
            topo_entropy = float(-np.sum(prob * np.log(prob + 1e-8)))
            sorted_mask = np.sort(mask_flat)[::-1]
            top_k = max(1, int(np.ceil(0.1 * len(sorted_mask))))
            topo_concentration = float(np.sum(sorted_mask[:top_k]) / mask_sum)

        return np.array(
            [
                float(np.mean(topo_features[:, :, 0])),
                float(np.max(topo_features[:, :, 0])),
                float(np.mean(topo_features[:, :, 2])),
                float(np.mean(topo_features[:, :, 3])),
                float(np.mean(topo_features[:, :, 4])),
                float(np.mean(topo_features[:, :, 7])),
                topo_entropy,
                topo_concentration,
            ],
            dtype=np.float32,
        )

    def _count_components(self, binary):
        num_labels, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return max(0, num_labels - 1)

    def _count_holes(self, binary):
        background = (binary == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(background, connectivity=4)
        holes = 0

        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            touches_border = (
                ys.min() == 0
                or xs.min() == 0
                or ys.max() == binary.shape[0] - 1
                or xs.max() == binary.shape[1] - 1
            )
            if not touches_border:
                holes += 1

        return holes

    def _entropy(self, patch):
        patch = np.asarray(patch, dtype=np.float32)
        hist, _ = np.histogram(patch, bins=16, range=(0.0, 1.0), density=False)
        prob = hist.astype(np.float32)
        prob = prob / (prob.sum() + 1e-8)
        return float(-np.sum(prob * np.log(prob + 1e-8)))

    def _normalize(self, array):
        array = np.asarray(array, dtype=np.float32)
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        if max_val - min_val <= 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)