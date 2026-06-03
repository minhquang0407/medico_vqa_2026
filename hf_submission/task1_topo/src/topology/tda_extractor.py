import cv2
import numpy as np
from gtda.homology import CubicalPersistence


class TopologicalExtractor:
    """
    Module trích xuất TDA đúng nghĩa bằng Cubical Persistent Homology.

    Output đa tầng:
    - topo_mask: (grid_h, grid_w), attention/topology prior đã normalize về [0, 1]
    - topo_features: (grid_h, grid_w, 8), đặc trưng H0/H1 theo từng patch
    - global_features: (8,), đặc trưng topo toàn ảnh
    """

    FEATURE_NAMES = (
        "h0_count",
        "h1_count",
        "h0_total_persistence",
        "h1_total_persistence",
        "h0_max_persistence",
        "h1_max_persistence",
        "h0_mean_persistence",
        "h1_mean_persistence",
    )

    GLOBAL_FEATURE_NAMES = (
        "global_h0_count",
        "global_h1_count",
        "global_h0_total_persistence",
        "global_h1_total_persistence",
        "global_h0_max_persistence",
        "global_h1_max_persistence",
        "topo_entropy",
        "topo_concentration",
    )

    def __init__(self, image_size=(224, 224), grid_size=(14, 14), alpha=0.4, beta=0.6):
        self.image_size = image_size
        self.grid_size = grid_size
        self.patch_h = image_size[0] // grid_size[0]
        self.patch_w = image_size[1] // grid_size[1]

        # alpha: trọng số H0/components, beta: trọng số H1/holes.
        # Với ảnh y khoa, H1 thường quan trọng cho biên/lỗ/vòng tổn thương.
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.cubical_persistence = CubicalPersistence(
            homology_dimensions=[0, 1],
            n_jobs=-1,
        )

    def fit_transform(self, image_path_or_img) -> np.ndarray:
        """
        Interface tương thích ngược cho visualization.
        Chỉ trả về mặt nạ topo kích thước grid_size.
        """
        return self.extract_features(image_path_or_img)["topo_mask"]

    def extract_features(self, image_path_or_img) -> dict:
        """
        Trích xuất đặc trưng topo đa tầng.

        Returns:
            dict:
                topo_mask: np.ndarray, shape (grid_h, grid_w)
                topo_features: np.ndarray, shape (grid_h, grid_w, 8)
                global_features: np.ndarray, shape (8,)
                feature_names: tuple[str, ...]
                global_feature_names: tuple[str, ...]
        """
        img = self._read_image(image_path_or_img)
        height_map = self._build_height_map(img)

        topo_features = np.zeros((*self.grid_size, len(self.FEATURE_NAMES)), dtype=np.float32)
        topo_mask = np.zeros(self.grid_size, dtype=np.float32)

        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                r_start = row * self.patch_h
                r_end = (row + 1) * self.patch_h
                c_start = col * self.patch_w
                c_end = (col + 1) * self.patch_w

                patch = height_map[r_start:r_end, c_start:c_end]
                diagram = self._compute_patch_diagram(patch)

                h0_c, h0_tot, h0_max, h0_mean = self._compute_ph_stats(diagram, dim=0)
                h1_c, h1_tot, h1_max, h1_mean = self._compute_ph_stats(diagram, dim=1)

                feature_vector = np.array(
                    [
                        h0_c,
                        h1_c,
                        h0_tot,
                        h1_tot,
                        h0_max,
                        h1_max,
                        h0_mean,
                        h1_mean,
                    ],
                    dtype=np.float32,
                )

                topo_features[row, col, :] = feature_vector

                # Persistence thuần có thể cao ở chữ/đèn lóa/biên tối.
                # Gate thêm bằng lesion saliency trung bình của patch để ưu tiên vùng nghi ngờ bệnh lý.
                lesion_patch = 1.0 - patch
                lesion_gate = float(np.mean(lesion_patch))
                lesion_peak = float(np.percentile(lesion_patch, 90))
                saliency_gate = 0.25 + 0.75 * ((0.6 * lesion_gate) + (0.4 * lesion_peak))

                raw_score = (self.alpha * h0_tot) + (self.beta * h1_tot)
                topo_mask[row, col] = raw_score * saliency_gate

        topo_mask = self._normalize(topo_mask)
        global_features = self._compute_global_features(topo_features, topo_mask)

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
                raise ValueError(f"Không thể đọc ảnh: {image_path_or_img}")
            return img

        if image_path_or_img is None:
            raise ValueError("Input image is None. Hãy kiểm tra lại đường dẫn ảnh.")

        img = np.asarray(image_path_or_img)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Input image phải là ảnh màu có shape (H, W, 3).")
        return img

    def _build_height_map(self, img):
        """
        Tạo lesion-oriented scalar field cho cubical filtration.

        Không dùng raw red channel đơn thuần vì ảnh nội soi dễ bị nhiễu bởi:
        - specular highlight/đèn lóa,
        - chữ overlay sáng,
        - vùng mô bình thường cũng đỏ mạnh.

        Scalar field mới kết hợp:
        - red excess: đỏ tương đối so với xanh/lục,
        - Lab-a redness: trục xanh-đỏ trong không gian Lab,
        - local edge/texture: biên và cấu trúc niêm mạc,
        - tissue mask: bỏ vùng đen ngoài ống soi,
        - specular suppression: giảm vùng quá sáng nhưng saturation thấp.

        CubicalPersistence dùng sublevel filtration, nên trả về `1.0 - lesion_score`
        để vùng nghi ngờ tổn thương có filtration value thấp và xuất hiện sớm.
        """
        img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

        img_float = img_resized.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_float)

        gray_uint8 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray = gray_uint8.astype(np.float32) / 255.0

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0

        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        lab_a = lab[:, :, 1].astype(np.float32) / 255.0

        # Đỏ tương đối: giảm ảnh hưởng của vùng chỉ đơn thuần sáng.
        red_excess = r - 0.5 * g - 0.5 * b
        red_excess = np.clip(red_excess, 0.0, 1.0)
        red_excess = self._normalize(red_excess)

        # Lab-a cao thể hiện xu hướng đỏ hơn xanh.
        lab_redness = np.clip((lab_a - 0.5) * 2.0, 0.0, 1.0)
        lab_redness = self._normalize(lab_redness)

        # Edge/texture nhẹ để bắt biên cấu trúc, nhưng sẽ bị tissue/specular mask kiểm soát.
        edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = self._normalize(np.abs(edges))

        # Bỏ nền đen ngoài nội soi và vùng quá tối.
        tissue_mask = (gray > 0.08).astype(np.float32)
        tissue_mask = cv2.morphologyEx(
            tissue_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        # Specular highlight: value rất cao nhưng saturation thấp/trung bình.
        specular_mask = ((value > 0.86) & (saturation < 0.38)).astype(np.float32)
        specular_mask = cv2.dilate(
            specular_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        # Edge chỉ đóng vai trò phụ. Nếu quá cao, heatmap sẽ bắt chữ overlay và đèn lóa.
        lesion_score = (
            0.55 * red_excess
            + 0.40 * lab_redness
            + 0.05 * edges
        )

        lesion_score *= tissue_mask
        lesion_score *= 1.0 - 0.98 * specular_mask
        lesion_score = cv2.GaussianBlur(lesion_score, (5, 5), 0)
        lesion_score = self._normalize(lesion_score)

        return 1.0 - lesion_score

    def _compute_patch_diagram(self, patch):
        """
        Tính persistence diagram cho một patch 2D bằng CubicalPersistence.
        Output diagram có cột: birth, death, homology_dimension.
        """
        patch_tensor = np.expand_dims(patch.astype(np.float32), axis=0)
        return self.cubical_persistence.fit_transform(patch_tensor)[0]

    def _compute_ph_stats(self, diagram, dim):
        """
        Trích xuất 4 thống kê từ persistence diagram cho H0 hoặc H1:
        count, total persistence, max persistence, mean persistence.
        """
        if diagram.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        dgm_dim = diagram[diagram[:, 2] == dim]
        dgm_valid = dgm_dim[np.isfinite(dgm_dim[:, 1])]

        if len(dgm_valid) == 0:
            return 0.0, 0.0, 0.0, 0.0

        persistences = dgm_valid[:, 1] - dgm_valid[:, 0]
        persistences = persistences[persistences > 1e-8]

        if len(persistences) == 0:
            return 0.0, 0.0, 0.0, 0.0

        count = float(len(persistences))
        total_p = float(np.sum(persistences))
        max_p = float(np.max(persistences))
        mean_p = float(np.mean(persistences))

        return count, total_p, max_p, mean_p

    def _compute_global_features(self, topo_features, topo_mask):
        global_sums = np.sum(topo_features[:, :, [0, 1, 2, 3]], axis=(0, 1))
        global_maxs = np.max(topo_features[:, :, [4, 5]], axis=(0, 1))

        mask_flat = topo_mask.flatten().astype(np.float32)
        mask_sum = float(np.sum(mask_flat))

        if mask_sum <= 1e-8:
            topo_entropy = 0.0
            topo_concentration = 0.0
        else:
            prob_dist = mask_flat / mask_sum
            topo_entropy = float(-np.sum(prob_dist * np.log(prob_dist + 1e-8)))

            sorted_mask = np.sort(mask_flat)[::-1]
            top_k = max(1, int(np.ceil(0.1 * len(sorted_mask))))
            topo_concentration = float(np.sum(sorted_mask[:top_k]) / mask_sum)

        return np.concatenate(
            [
                global_sums.astype(np.float32),
                global_maxs.astype(np.float32),
                np.array([topo_entropy, topo_concentration], dtype=np.float32),
            ]
        )

    def _normalize(self, array):
        array = np.asarray(array, dtype=np.float32)
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        if max_val - min_val <= 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)