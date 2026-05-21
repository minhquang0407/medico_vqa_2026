import csv
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.topology.tda_morphology import TopologicalExtractor


GRID_SIZE = (14, 14)
THRESHOLDS = np.linspace(0.1, 0.9, 9)
GT_PATCH_THRESHOLD = 0.10
TOP_K = 5


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if pred_mask.sum() == 0 else 0.0
    return float(intersection / union)


def calculate_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def downsample_gt_mask(mask_path, grid_size=GRID_SIZE, threshold=127):
    """
    Chuyển ground-truth segmentation mask về patch-level 14x14.

    gt_ratio[i, j] là tỷ lệ pixel lesion trong patch.
    gt_binary[i, j] = 1 nếu patch có ít nhất GT_PATCH_THRESHOLD diện tích lesion.
    """
    gt_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise ValueError(f"Không thể đọc mask: {mask_path}")

    gt_binary_full = (gt_img > threshold).astype(np.float32)
    gt_ratio = cv2.resize(
        gt_binary_full,
        grid_size[::-1],
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    gt_binary_patch = (gt_ratio >= GT_PATCH_THRESHOLD).astype(np.uint8)
    return gt_ratio, gt_binary_patch


def calculate_best_threshold_metrics(topo_mask, gt_binary_patch):
    best = {
        "threshold": 0.5,
        "iou": -1.0,
        "dice": -1.0,
    }

    metrics_at_05 = None
    for threshold in THRESHOLDS:
        pred = (topo_mask >= threshold).astype(np.uint8)
        iou = calculate_iou(pred, gt_binary_patch)
        dice = calculate_dice(pred, gt_binary_patch)

        if abs(threshold - 0.5) < 1e-8:
            metrics_at_05 = (iou, dice)

        if iou > best["iou"]:
            best = {
                "threshold": float(threshold),
                "iou": float(iou),
                "dice": float(dice),
            }

    if metrics_at_05 is None:
        pred_05 = (topo_mask >= 0.5).astype(np.uint8)
        metrics_at_05 = (
            calculate_iou(pred_05, gt_binary_patch),
            calculate_dice(pred_05, gt_binary_patch),
        )

    return best, metrics_at_05


def pointing_game(topo_mask, gt_binary_patch):
    max_idx = int(np.argmax(topo_mask.flatten()))
    return int(gt_binary_patch.flatten()[max_idx] == 1)


def topk_hit(topo_mask, gt_binary_patch, k=TOP_K):
    flat_scores = topo_mask.flatten()
    flat_gt = gt_binary_patch.flatten()
    k = min(k, len(flat_scores))
    top_indices = np.argsort(flat_scores)[-k:]
    return int(np.any(flat_gt[top_indices] == 1))


def safe_auc_ap(gt_binary_patch, topo_mask):
    y_true = gt_binary_patch.flatten().astype(np.uint8)
    y_score = topo_mask.flatten().astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def safe_spearman(gt_ratio, topo_mask):
    x = gt_ratio.flatten().astype(np.float32)
    y = topo_mask.flatten().astype(np.float32)

    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return np.nan

    corr = spearmanr(x, y).correlation
    return float(corr) if np.isfinite(corr) else np.nan


def red_excess_baseline(img_path, grid_size=GRID_SIZE):
    """
    Baseline màu rẻ để kiểm tra TDA có hơn heuristic red-excess không.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")

    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    red_excess = np.clip(r - 0.5 * g - 0.5 * b, 0.0, 1.0)
    patch_map = cv2.resize(red_excess, grid_size[::-1], interpolation=cv2.INTER_AREA)
    return normalize(patch_map)


def center_prior_baseline(grid_size=GRID_SIZE, sigma=0.35):
    h, w = grid_size
    ys, xs = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dist2 = ((ys - cy) / h) ** 2 + ((xs - cx) / w) ** 2
    prior = np.exp(-dist2 / (2.0 * sigma ** 2))
    return normalize(prior)


def random_baseline(grid_size=GRID_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random(grid_size).astype(np.float32)


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_val) / (max_val - min_val)


def find_mask_path(mask_dir, img_name):
    stem = Path(img_name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = Path(mask_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def evaluate_score_map(score_map, gt_ratio, gt_binary_patch):
    best, metrics_05 = calculate_best_threshold_metrics(score_map, gt_binary_patch)
    auc, ap = safe_auc_ap(gt_binary_patch, score_map)
    return {
        "iou_05": metrics_05[0],
        "dice_05": metrics_05[1],
        "best_iou": best["iou"],
        "best_dice": best["dice"],
        "best_threshold": best["threshold"],
        "pointing_hit": pointing_game(score_map, gt_binary_patch),
        "topk_hit": topk_hit(score_map, gt_binary_patch),
        "auc": auc,
        "ap": ap,
        "spearman": safe_spearman(gt_ratio, score_map),
    }


def write_csv(rows, output_csv):
    if not rows:
        return

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_metric_block(name, rows, prefix):
    print(f"\n{name}")
    print("-" * len(name))

    for metric in (
        "iou_05",
        "dice_05",
        "best_iou",
        "best_dice",
        "pointing_hit",
        "topk_hit",
        "auc",
        "ap",
        "spearman",
    ):
        values = [row[f"{prefix}_{metric}"] for row in rows]
        stats = summarize(values)
        print(
            f"{metric:>14}: "
            f"mean={stats['mean']:.4f} | "
            f"median={stats['median']:.4f} | "
            f"p75={stats['p75']:.4f} | "
            f"p90={stats['p90']:.4f}"
        )


def print_case_lists(rows, output_dir, k=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda x: x["tda_best_iou"])
    bottom = sorted_rows[:k]
    top = sorted_rows[-k:][::-1]
    median_start = max(0, len(sorted_rows) // 2 - k // 2)
    median = sorted_rows[median_start:median_start + k]

    for name, subset in (("bottom_cases.txt", bottom), ("median_cases.txt", median), ("top_cases.txt", top)):
        path = output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(
                    f"{row['image_name']} | "
                    f"tda_best_iou={row['tda_best_iou']:.4f} | "
                    f"tda_pointing={row['tda_pointing_hit']} | "
                    f"tda_topk={row['tda_topk_hit']}\n"
                )

    print(f"\n📝 Đã lưu top/bottom/median cases vào: {output_dir}")


def test_tda_batch(
    image_dir,
    mask_dir,
    output_csv="visuals/tda_morphy_iou_results.csv",
    output_case_dir="visuals/tda_morphy_iou_cases",
    max_images=None,
):
    print(f"🔄 Đang quét dữ liệu từ:\n- Ảnh: {image_dir}\n- Mask: {mask_dir}\n")

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_files = sorted(
        f.name for f in image_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if max_images is not None:
        image_files = image_files[:max_images]

    if not image_files:
        print("❌ Không tìm thấy ảnh nào trong thư mục!")
        return

    print(f"🚀 Bắt đầu xử lý {len(image_files)} ảnh...")
    extractor = TopologicalExtractor(grid_size=GRID_SIZE)
    center_prior = center_prior_baseline()

    rows = []
    failed_count = 0

    for idx, img_name in enumerate(tqdm(image_files, desc="TDA evaluation"), 1):
        img_path = image_dir / img_name
        mask_path = find_mask_path(mask_dir, img_name)

        if mask_path is None:
            failed_count += 1
            continue

        try:
            gt_ratio, gt_binary_patch = downsample_gt_mask(mask_path)
            lesion_patch_ratio = float(gt_binary_patch.mean())
            lesion_area_ratio = float(gt_ratio.mean())

            topo_output = extractor.fit_transform(str(img_path))
            tda_map = normalize(topo_output)
            red_map = red_excess_baseline(img_path)
            random_map = random_baseline(seed=idx)

            # Hybrid baselines: kiểm tra TDA có đóng góp thêm khi kết hợp màu/vị trí không.
            red_center_map = normalize(0.50 * red_map + 0.50 * center_prior)
            red_tda_map = normalize(0.70 * red_map + 0.30 * tda_map)
            center_tda_map = normalize(0.70 * center_prior + 0.30 * tda_map)
            red_center_tda_map = normalize(0.45 * red_map + 0.35 * center_prior + 0.20 * tda_map)

            tda_metrics = evaluate_score_map(tda_map, gt_ratio, gt_binary_patch)
            red_metrics = evaluate_score_map(red_map, gt_ratio, gt_binary_patch)
            center_metrics = evaluate_score_map(center_prior, gt_ratio, gt_binary_patch)
            random_metrics = evaluate_score_map(random_map, gt_ratio, gt_binary_patch)
            red_center_metrics = evaluate_score_map(red_center_map, gt_ratio, gt_binary_patch)
            red_tda_metrics = evaluate_score_map(red_tda_map, gt_ratio, gt_binary_patch)
            center_tda_metrics = evaluate_score_map(center_tda_map, gt_ratio, gt_binary_patch)
            red_center_tda_metrics = evaluate_score_map(red_center_tda_map, gt_ratio, gt_binary_patch)

            row = {
                "image_name": img_name,
                "mask_name": mask_path.name,
                "lesion_area_ratio": lesion_area_ratio,
                "lesion_patch_ratio": lesion_patch_ratio
            }

            for prefix, metrics in (
                ("tda", tda_metrics),
                ("red", red_metrics),
                ("center", center_metrics),
                ("random", random_metrics),
                ("red_center", red_center_metrics),
                ("red_tda", red_tda_metrics),
                ("center_tda", center_tda_metrics),
                ("red_center_tda", red_center_tda_metrics),
            ):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {img_name}: {e}")
            failed_count += 1

    if not rows:
        print("\n❌ Không có dữ liệu để đánh giá.")
        return

    write_csv(rows, output_csv)
    print_case_lists(rows, output_case_dir)

    print("\n======================================")
    print(" 🏆 BÁO CÁO NGHIỆM THU TDA PATCH-LEVEL")
    print("======================================")
    print(f"Tổng số ảnh quét:   {len(image_files)}")
    print(f"Số ảnh hợp lệ:      {len(rows)}")
    print(f"Số ảnh lỗi/bỏ qua:  {failed_count}")
    print(f"CSV kết quả:        {output_csv}")

    lesion_stats = summarize([row["lesion_area_ratio"] for row in rows])
    print("--------------------------------------")
    print(
        "Lesion area ratio: "
        f"mean={lesion_stats['mean']:.4f} | "
        f"median={lesion_stats['median']:.4f} | "
        f"p75={lesion_stats['p75']:.4f}"
    )

    print_metric_block("TDA Cubical Persistence", rows, "tda")
    print_metric_block("Red-excess Baseline", rows, "red")
    print_metric_block("Center Prior Baseline", rows, "center")
    print_metric_block("Random Baseline", rows, "random")
    print_metric_block("Hybrid Red + Center", rows, "red_center")
    print_metric_block("Hybrid Red + TDA", rows, "red_tda")
    print_metric_block("Hybrid Center + TDA", rows, "center_tda")
    print_metric_block("Hybrid Red + Center + TDA", rows, "red_center_tda")

    tda_mean = summarize([row["tda_best_iou"] for row in rows])["mean"]
    red_mean = summarize([row["red_best_iou"] for row in rows])["mean"]
    center_mean = summarize([row["center_best_iou"] for row in rows])["mean"]
    red_center_mean = summarize([row["red_center_best_iou"] for row in rows])["mean"]
    red_tda_mean = summarize([row["red_tda_best_iou"] for row in rows])["mean"]
    center_tda_mean = summarize([row["center_tda_best_iou"] for row in rows])["mean"]
    red_center_tda_mean = summarize([row["red_center_tda_best_iou"] for row in rows])["mean"]

    print("\n--------------------------------------")
    print(f"Δ best IoU TDA - Red baseline:              {tda_mean - red_mean:+.4f}")
    print(f"Δ best IoU TDA - Center baseline:           {tda_mean - center_mean:+.4f}")
    print(f"Δ best IoU Red+TDA - Red:                   {red_tda_mean - red_mean:+.4f}")
    print(f"Δ best IoU Center+TDA - Center:             {center_tda_mean - center_mean:+.4f}")
    print(f"Δ best IoU Red+Center+TDA - Red+Center:     {red_center_tda_mean - red_center_mean:+.4f}")
    print("======================================\n")


if __name__ == "__main__":
    IMAGE_DIR = "data/raw/Kvasir-SEG/images"
    MASK_DIR = "data/raw/Kvasir-SEG/masks"

    test_tda_batch(
        IMAGE_DIR,
        MASK_DIR,
        output_csv="visuals/tda_iou_results.csv",
        output_case_dir="visuals/tda_iou_cases",
        max_images=None,
    )