import csv
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.topology.lesion_prior import LesionPriorExtractor
from src.topology.tda_morphology import TopologicalExtractor as MorphoExtractor
from src.topology.tda_extractor import TopologicalExtractor as CubicalExtractor
from src.topology.tda_morphology_l import TopologicalExtractor as MorphoLExtractor


GRID_SIZE = (14, 14)
THRESHOLDS = np.linspace(0.1, 0.9, 9)
GT_PATCH_THRESHOLD = 0.10
TOP_K = 5
GRID_SEARCH_STEP = 0.10  # Step 0.1 để 4 biến không tốn quá nhiều thời gian (có 286 tổ hợp)



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


def generate_weight_grid(step=GRID_SEARCH_STEP):
    """Sinh các bộ trọng số tổng bằng 1 cho 4 cấu phần."""
    values = np.arange(0.0, 1.0 + 1e-8, step)
    weights = []
    for w_lesion in values:
        for w_morpho in values:
            for w_cubical in values:
                w_morpho_l = 1.0 - w_lesion - w_morpho - w_cubical
                if w_morpho_l < -1e-8:
                    continue
                if w_morpho_l < 0:
                    w_morpho_l = 0.0
                weights.append((float(w_lesion), float(w_morpho), float(w_cubical), float(w_morpho_l)))
    return weights


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


def write_grid_search_csv(grid_summary, output_csv):
    if not grid_summary:
        return
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_summary[0].keys()))
        writer.writeheader()
        writer.writerows(grid_summary)


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


def summarize_grid_search(grid_metric_records):
    grid_summary = []

    for weights, metric_list in grid_metric_records.items():
        if not metric_list:
            continue

        w_l, w_m, w_c, w_ml = weights
        row = {
            "w_lesion": w_l,
            "w_morpho": w_m,
            "w_cubical": w_c,
            "w_morpho_l": w_ml,
        }

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
            values = [item[metric] for item in metric_list]
            stats = summarize(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_p75"] = stats["p75"]
            row[f"{metric}_p90"] = stats["p90"]

        grid_summary.append(row)

    grid_summary.sort(key=lambda x: x["best_iou_mean"], reverse=True)
    return grid_summary


def print_grid_search_report(grid_summary, top_n=10):
    if not grid_summary:
        print("\nGrid Search: không có dữ liệu.")
        return

    print("\nGrid Search Lesion / Morpho / Cubical / Morpho_L")
    print("--------------------------------------")

    for metric in ("best_iou_mean", "auc_mean", "ap_mean", "topk_hit_mean", "pointing_hit_mean"):
        best = max(grid_summary, key=lambda x: -np.inf if not np.isfinite(x[metric]) else x[metric])
        print(
            f"Best {metric}: "
            f"{best[metric]:.4f} | "
            f"lesion={best['w_lesion']:.2f}, "
            f"morpho={best['w_morpho']:.2f}, "
            f"cubical={best['w_cubical']:.2f}, "
            f"morpho_l={best['w_morpho_l']:.2f}"
        )

    print(f"\nTop {top_n} by best_iou_mean:")
    for rank, row in enumerate(grid_summary[:top_n], 1):
        print(
            f"#{rank:02d} "
            f"IoU={row['best_iou_mean']:.4f}, "
            f"AUC={row['auc_mean']:.4f}, "
            f"AP={row['ap_mean']:.4f}, "
            f"TopK={row['topk_hit_mean']:.4f} | "
            f"lesion={row['w_lesion']:.2f}, "
            f"morpho={row['w_morpho']:.2f}, "
            f"cubical={row['w_cubical']:.2f}, "
            f"morpho_l={row['w_morpho_l']:.2f}"
        )


def print_case_lists(rows, output_dir, k=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda x: x["lesion_best_iou"])
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
                    f"lesion_best_iou={row['lesion_best_iou']:.4f} | "
                    f"lesion_pointing={row['lesion_pointing_hit']} | "
                    f"lesion_topk={row['lesion_topk_hit']}\n"
                )

    print(f"\n📝 Đã lưu top/bottom/median cases dựa trên lesion_prior vào: {output_dir}")


def test_tda_batch(
    image_dir,
    mask_dir,
    output_csv="visuals/tda_iou_results.csv",
    output_case_dir="visuals/tda_iou_cases",
    output_grid_csv="visuals/tda_weight_grid_search.csv",
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
    
    prior_extractor = LesionPriorExtractor(grid_size=GRID_SIZE, use_morphology=False)
    morpho_extractor = MorphoExtractor(grid_size=GRID_SIZE)
    cubical_extractor = CubicalExtractor(grid_size=GRID_SIZE)
    morpho_l_extractor = MorphoLExtractor(grid_size=GRID_SIZE)

    rows = []
    failed_count = 0
    weight_grid = generate_weight_grid()
    grid_metric_records = {weights: [] for weights in weight_grid}

    for idx, img_name in enumerate(tqdm(image_files, desc="Evaluation"), 1):
        img_path = image_dir / img_name
        mask_path = find_mask_path(mask_dir, img_name)

        if mask_path is None:
            failed_count += 1
            continue

        try:
            gt_ratio, gt_binary_patch = downsample_gt_mask(mask_path)
            lesion_patch_ratio = float(gt_binary_patch.mean())
            lesion_area_ratio = float(gt_ratio.mean())

            # Trích xuất 4 loại feature maps
            lesion_out = prior_extractor.extract_prior(str(img_path))
            lesion_map = normalize(lesion_out["prior_mask"]) # red + center
            morpho_map = normalize(morpho_extractor.fit_transform(str(img_path)))
            cubical_map = normalize(cubical_extractor.fit_transform(str(img_path)))
            morpho_l_map = normalize(morpho_l_extractor.fit_transform(str(img_path)))

            lesion_metrics = evaluate_score_map(lesion_map, gt_ratio, gt_binary_patch)
            morpho_metrics = evaluate_score_map(morpho_map, gt_ratio, gt_binary_patch)
            cubical_metrics = evaluate_score_map(cubical_map, gt_ratio, gt_binary_patch)
            morpho_l_metrics = evaluate_score_map(morpho_l_map, gt_ratio, gt_binary_patch)

            row = {
                "image_name": img_name,
                "mask_name": mask_path.name,
                "lesion_area_ratio": lesion_area_ratio,
                "lesion_patch_ratio": lesion_patch_ratio,
            }

            for prefix, metrics in (
                ("lesion", lesion_metrics),
                ("morpho", morpho_metrics),
                ("cubical", cubical_metrics),
                ("morpho_l", morpho_l_metrics),
            ):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value

            for weights in weight_grid:
                w_l, w_m, w_c, w_ml = weights
                grid_map = normalize(
                    (w_l * lesion_map) +
                    (w_m * morpho_map) +
                    (w_c * cubical_map) +
                    (w_ml * morpho_l_map)
                )
                grid_metric_records[weights].append(evaluate_score_map(grid_map, gt_ratio, gt_binary_patch))

            rows.append(row)

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {img_name}: {e}")
            failed_count += 1

    if not rows:
        print("\n❌ Không có dữ liệu để đánh giá.")
        return

    write_csv(rows, output_csv)
    print_case_lists(rows, output_case_dir)

    grid_summary = summarize_grid_search(grid_metric_records)
    write_grid_search_csv(grid_summary, output_grid_csv)

    print("\n======================================")
    print(" 🏆 BÁO CÁO NGHIỆM THU TDA PATCH-LEVEL")
    print("======================================")
    print(f"Tổng số ảnh quét:   {len(image_files)}")
    print(f"Số ảnh hợp lệ:      {len(rows)}")
    print(f"Số ảnh lỗi/bỏ qua:  {failed_count}")
    print(f"CSV kết quả:        {output_csv}")
    print(f"CSV grid search:    {output_grid_csv}")

    lesion_stats = summarize([row["lesion_area_ratio"] for row in rows])
    print("--------------------------------------")
    print(
        "Lesion area ratio: "
        f"mean={lesion_stats['mean']:.4f} | "
        f"median={lesion_stats['median']:.4f} | "
        f"p75={lesion_stats['p75']:.4f}"
    )

    print_metric_block("Lesion Prior (Red + Center)", rows, "lesion")
    print_metric_block("Morphology TDA", rows, "morpho")
    print_metric_block("Cubical TDA", rows, "cubical")
    print_metric_block("Morphology_L TDA", rows, "morpho_l")

    lesion_mean = summarize([row["lesion_best_iou"] for row in rows])["mean"]
    morpho_mean = summarize([row["morpho_best_iou"] for row in rows])["mean"]
    cubical_mean = summarize([row["cubical_best_iou"] for row in rows])["mean"]
    morpho_l_mean = summarize([row["morpho_l_best_iou"] for row in rows])["mean"]

    print("\n--------------------------------------")
    print(f"Δ Morpho - Lesion:     {morpho_mean - lesion_mean:+.4f}")
    print(f"Δ Cubical - Lesion:    {cubical_mean - lesion_mean:+.4f}")
    print(f"Δ Morpho_L - Lesion:   {morpho_l_mean - lesion_mean:+.4f}")

    print_grid_search_report(grid_summary)
    print("======================================\n")


if __name__ == "__main__":
    IMAGE_DIR = "data/raw/Kvasir-SEG/images"
    MASK_DIR = "data/raw/Kvasir-SEG/masks"

    test_tda_batch(
        IMAGE_DIR,
        MASK_DIR,
        output_csv="visuals/tda_iou_results.csv",
        output_case_dir="visuals/tda_iou_cases",
        output_grid_csv="visuals/tda_weight_grid_search.csv",
        max_images=None,
    )