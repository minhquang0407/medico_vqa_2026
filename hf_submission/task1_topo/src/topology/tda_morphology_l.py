import cv2
import numpy as np


class TopologicalExtractor:
    """
    Module trích xuất Đặc trưng Hình thái (Morphological/Topological)
    dựa trên lý thuyết Lọc Tập mức (Level Set Filtration).
    """

    def __init__(self, grid_size=(14, 14)):
        self.grid_size = grid_size

    def fit_transform(self, image_path_or_img):
        """
        Xây dựng ma trận 14x14 bằng Không gian CIELAB bất biến với ánh sáng.
        """
        if isinstance(image_path_or_img, str):
            img = cv2.imread(image_path_or_img)
        else:
            img = image_path_or_img

        # 1. Chuyển đổi sang Không gian màu Y khoa (CIELAB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 2. Tách kênh 'a' (Trục Lục - Đỏ)
        # Bỏ qua hoàn toàn kênh 'L' (Ánh sáng) để triệt tiêu hiệu ứng chóa đèn và bóng râm
        a_channel = lab_img[:, :, 1]

        # 3. Tạo Bản đồ Địa hình (Height Map)
        # Làm mịn mạnhtay để loại bỏ nhiễu hạt (cấu trúc vi mô)
        smoothed_terrain = cv2.GaussianBlur(a_channel, (35, 35), 0)

        # 4. Phân rã xuống không gian Vận tải Tối ưu (14x14 Grid)
        grid_terrain = cv2.resize(smoothed_terrain, self.grid_size, interpolation=cv2.INTER_AREA)
        grid_terrain = grid_terrain.astype(np.float32)

        # 5. Tăng cường Đỉnh núi (Topological Prominence)
        # Dùng hàm mũ 3 để ép các thung lũng (niêm mạc bình thường) chìm hẳn xuống
        grid_terrain = np.power(grid_terrain, 3)

        # 6. Chuẩn hóa về [0, 1]
        min_val = np.min(grid_terrain)
        max_val = np.max(grid_terrain)

        if max_val - min_val > 0:
            topo_mask = (grid_terrain - min_val) / (max_val - min_val)
        else:
            topo_mask = np.zeros(self.grid_size, dtype=np.float32)

        return topo_mask
