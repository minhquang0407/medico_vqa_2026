import numpy as np
import cv2
from ripser import ripser


class TopologicalExtractor:
    """
    Module trích xuất đặc trưng hình học không gian (Spatial TDA).
    Sử dụng Representative Cocycles để ánh xạ cấu trúc topo về lưới 14x14.
    """

    def __init__(self, num_points=800, alpha_spatial=0.05, beta_color=5.0, grid_size=(14, 14)):
        self.num_points = num_points
        self.alpha_spatial = alpha_spatial
        self.beta_color = beta_color
        self.grid_size = grid_size

    def _farthest_point_sampling(self, point_cloud, coords):
        """
        Lấy mẫu rải rác nhưng giữ lại tọa độ (x, y) gốc của pixel.
        """
        N = point_cloud.shape[0]
        K = self.num_points
        if N <= K:
            return point_cloud, coords

        farthest_pts = np.zeros((K, point_cloud.shape[1]))
        farthest_coords = np.zeros((K, 2), dtype=int)  # Lưu tọa độ pixel

        idx = np.random.randint(N)
        farthest_pts[0] = point_cloud[idx]
        farthest_coords[0] = coords[idx]
        distances = np.linalg.norm(point_cloud - farthest_pts[0], axis=1)

        for i in range(1, K):
            farthest_idx = np.argmax(distances)
            farthest_pts[i] = point_cloud[farthest_idx]
            farthest_coords[i] = coords[farthest_idx]

            new_distances = np.linalg.norm(point_cloud - farthest_pts[i], axis=1)
            distances = np.minimum(distances, new_distances)

        return farthest_pts, farthest_coords

    def _extract_point_cloud(self, image_bgr):
        """
        Trích xuất đám mây điểm 5D và ma trận tọa độ tương ứng.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = img_rgb.shape

        y_coords, x_coords = np.mgrid[0:H, 0:W]

        # Tọa độ gốc để mapping lại sau này
        pixel_coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)

        x_norm = (x_coords / float(W)) * self.alpha_spatial
        y_norm = (y_coords / float(H)) * self.alpha_spatial
        r_norm = (img_rgb[:, :, 0] / 255.0) * self.beta_color
        g_norm = (img_rgb[:, :, 1] / 255.0) * self.beta_color
        b_norm = (img_rgb[:, :, 2] / 255.0) * self.beta_color

        point_cloud_5d = np.stack([x_norm, y_norm, r_norm, g_norm, b_norm], axis=-1)
        flat_pc = point_cloud_5d.reshape(-1, 5)

        # Lọc bỏ nhiễu đen (Background)
        color_sum = img_rgb.sum(axis=-1).flatten()
        valid_idx = np.where(color_sum > 30)[0]

        return flat_pc[valid_idx], pixel_coords[valid_idx], H, W

    def fit_transform(self, image):
        """
        Luồng thực thi: Point Cloud -> Ripser (Cocycles) -> Heatmap 14x14.
        """
        pc_5d, coords_2d, H, W = self._extract_point_cloud(image)
        sampled_pc, sampled_coords = self._farthest_point_sampling(pc_5d, coords_2d)

        # Chạy thuật toán Ripser, yêu cầu trả về Cocycles
        res = ripser(sampled_pc, maxdim=1, do_cocycles=True)
        dgms = res['dgms']
        cocycles = res['cocycles']

        # Khởi tạo ma trận không gian 14x14
        grid_h, grid_w = self.grid_size
        topo_mask = np.zeros(self.grid_size, dtype=np.float32)

        # Trích xuất H1 (Lỗ hổng / Vết loét)
        if len(dgms[1]) > 0:
            dgm1 = dgms[1]
            # Tính tuổi thọ (Persistence = death - birth)
            pers = dgm1[:, 1] - dgm1[:, 0]
            # Bỏ qua các điểm nhiễu có tuổi thọ quá thấp
            valid_h1_idx = np.where(pers > np.percentile(pers, 80))[0]

            for idx in valid_h1_idx:
                cocycle = cocycles[1][idx]
                persistence_val = pers[idx]
                # Mỗi phần tử trong cocycle là một cạnh chứa 2 đỉnh [v1, v2]
                for edge in cocycle:
                    for vertex_idx in edge[:2]:
                        x_px, y_px = sampled_coords[vertex_idx]
                        # Chuyển đổi tọa độ pixel về tọa độ lưới 14x14
                        grid_x = min(int((x_px / W) * grid_w), grid_w - 1)
                        grid_y = min(int((y_px / H) * grid_h), grid_h - 1)
                        topo_mask[grid_y, grid_x] += persistence_val
        topo_mask = cv2.GaussianBlur(topo_mask, (3, 3), sigmaX=1.2, sigmaY=1.2)
        # Chuẩn hóa ma trận (Normalize) về [0, 1]
        max_val = np.max(topo_mask)
        if max_val > 0:
            topo_mask = topo_mask / max_val

        return topo_mask