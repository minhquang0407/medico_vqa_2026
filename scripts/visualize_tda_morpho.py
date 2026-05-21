import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.topology.tda_morphology_l import TopologicalExtractor


def quick_test_tda(image_path="data/raw/sample_kvasir_image.jpg"):

    img = cv2.imread(image_path)


    print("[1] Khởi tạo Topological Extractor (Level Set)...")
    extractor = TopologicalExtractor(grid_size=(14, 14))
    topo_mask = extractor.fit_transform(image_path)
    print("[3] Kết xuất biểu đồ trực quan...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Phóng to ma trận 14x14 lên bằng kích thước ảnh gốc
    h, w, _ = img_rgb.shape
    heatmap = cv2.resize(topo_mask, (w, h), interpolation=cv2.INTER_CUBIC)

    # Chuyển heatmap thành ảnh màu
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Trộn ảnh gốc và heatmap
    blended_img = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # Vẽ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Ảnh Gốc")
    axes[0].axis("off")

    axes[1].imshow(topo_mask, cmap='jet')
    axes[1].set_title("2. Ma trận TDA (14x14)")
    axes[1].axis("off")

    axes[2].imshow(blended_img)
    axes[2].set_title("3. Ảnh Trộn (Xác thực Vị trí)")
    axes[2].axis("off")

    save_path = "tda_morpho_result.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[4] Xong! Vui lòng mở ảnh: {save_path}")


if __name__ == "__main__":
    quick_test_tda("sample_kvasir_image.jpg")
