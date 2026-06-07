"""Create paper figures for the CATA paper.

Outputs:
- cata_architecture.pdf/png: architecture overview.
- cata_qualitative_examples.pdf/png: original image, heatmap, and explanation cards.
- selected_examples.json: metadata for chosen examples.

Example:
    python scripts/make_paper_figures.py \
      --task2-jsonl hf_submission/cata_multitask_final/submission_task2_cata_final.jsonl \
      --visual-dir hf_submission/cata_multitask_final/visuals \
      --output-dir paper_assets/figures \
      --num-examples 4 \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["__line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def normalize_text(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def resolve_path(path_text: str, base_dir: Path, task2_dir: Path) -> Optional[Path]:
    if not path_text:
        return None
    p = Path(path_text)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            task2_dir / p,
            base_dir / p.name,
            PROJECT_ROOT / p,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def heatmap_path_from_row(row: Dict[str, Any], visual_dir: Path, task2_dir: Path) -> Optional[Path]:
    items = row.get("visual_explanation") or []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            data = str(item.get("data", ""))
            if "heatmap" in item_type or data.lower().endswith((".png", ".jpg", ".jpeg")):
                resolved = resolve_path(data, visual_dir, task2_dir)
                if resolved:
                    return resolved
    val_id = str(row.get("val_id", "")).strip()
    if val_id.isdigit():
        for name in (f"{int(val_id):04d}_heatmap.png", f"{int(val_id)}_heatmap.png"):
            cand = visual_dir / name
            if cand.exists():
                return cand
    return None


def load_task2_subset() -> List[Any]:
    from datasets import Image as HfImage, load_dataset

    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    val_set_task2 = (
        ds.filter(lambda x: x["complexity"] == 1)
        .shuffle(seed=42)
        .select(range(1500))
        .add_column("val_id", list(range(1500)))
        .remove_columns(["complexity", "answer", "original", "question_class"])
        .cast_column("image", HfImage())
    )
    return list(val_set_task2)


def image_for_row(row: Dict[str, Any], task2_examples: Sequence[Any]) -> Image.Image:
    val_id = row.get("val_id")
    try:
        idx = int(val_id)
    except Exception:
        idx = int(row.get("__line_number", 1)) - 1
    idx = max(0, min(idx, len(task2_examples) - 1))
    image = task2_examples[idx]["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return image.convert("RGB")


def choose_examples(rows: List[Dict[str, Any]], num_examples: int, seed: int, visual_dir: Path, task2_dir: Path) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if heatmap_path_from_row(row, visual_dir, task2_dir) is not None]
    if not candidates:
        candidates = rows
    # Prefer rows with non-empty explanation and moderate length.
    scored = []
    for row in candidates:
        exp = normalize_text(row.get("textual_explanation", ""), 2000)
        answer = normalize_text(row.get("answer", ""), 300)
        score = 0
        score += 2 if len(exp) > 80 else 0
        score += 1 if len(answer) > 0 else 0
        score += 1 if row.get("confidence_score") not in (None, "") else 0
        scored.append((score, row))
    rng = random.Random(seed)
    rng.shuffle(scored)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:num_examples]]


def draw_rounded_box(ax, xy, width, height, text, facecolor, edgecolor, fontsize=10):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        weight="bold",
        wrap=True,
    )


def draw_arrow(ax, start, end, color="#3b82f6"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0, shrinkA=6, shrinkB=6),
    )


def make_architecture_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    draw_rounded_box(ax, (0.04, 0.66), 0.15, 0.14, "Endoscopy\nImage", "#e0f2fe", "#0284c7")
    draw_rounded_box(ax, (0.26, 0.66), 0.17, 0.14, "Frozen\nViT/timm", "#dbeafe", "#2563eb")
    draw_rounded_box(ax, (0.49, 0.66), 0.18, 0.14, "Visual TDA\nFusion", "#ede9fe", "#7c3aed")
    draw_rounded_box(ax, (0.74, 0.66), 0.18, 0.14, "Qwen2.5\nDecoder", "#dcfce7", "#16a34a")

    draw_rounded_box(ax, (0.26, 0.28), 0.17, 0.14, "Patch-level\nTDA", "#fef3c7", "#d97706")
    draw_rounded_box(ax, (0.49, 0.28), 0.18, 0.14, "TDA Condition\nVector", "#ffedd5", "#ea580c")
    draw_rounded_box(ax, (0.74, 0.28), 0.18, 0.14, "Gated TDA\nAdapter", "#fce7f3", "#db2777")
    draw_rounded_box(ax, (0.74, 0.05), 0.18, 0.11, "Generated\nAnswer", "#f8fafc", "#475569")
    draw_rounded_box(ax, (0.49, 0.88), 0.18, 0.10, "Question", "#f1f5f9", "#64748b")

    draw_arrow(ax, (0.19, 0.73), (0.26, 0.73))
    draw_arrow(ax, (0.43, 0.73), (0.49, 0.73))
    draw_arrow(ax, (0.67, 0.73), (0.74, 0.73))
    draw_arrow(ax, (0.34, 0.66), (0.34, 0.42), color="#d97706")
    draw_arrow(ax, (0.43, 0.35), (0.49, 0.35), color="#d97706")
    draw_arrow(ax, (0.58, 0.42), (0.58, 0.66), color="#7c3aed")
    draw_arrow(ax, (0.67, 0.35), (0.74, 0.35), color="#db2777")
    draw_arrow(ax, (0.83, 0.42), (0.83, 0.66), color="#db2777")
    draw_arrow(ax, (0.58, 0.88), (0.74, 0.76), color="#64748b")
    draw_arrow(ax, (0.83, 0.66), (0.83, 0.16), color="#16a34a")

    ax.text(
        0.5,
        0.015,
        "CATA couples patch-level topology with visual fusion and decoder adaptation for generative medical VQA.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )

    for ext in ("pdf", "png"):
        path = output_dir / f"cata_architecture.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"Saved {path}")
    plt.close(fig)


def fit_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def wrap_label(prefix: str, text: str, width: int, max_chars: int) -> str:
    text = normalize_text(text, max_chars=max_chars)
    wrapped = textwrap.fill(text, width=width)
    return f"{prefix}: {wrapped}"


def make_qualitative_figure(rows: List[Dict[str, Any]], visual_dir: Path, task2_dir: Path, output_dir: Path, num_examples: int, seed: int) -> List[Dict[str, Any]]:
    task2_examples = load_task2_subset()
    selected = choose_examples(rows, num_examples, seed, visual_dir, task2_dir)

    # Compact 2x2 layout for paper page limits: each cell contains original,
    # heatmap, and a short evidence card. This avoids the tall 1-example-per-row
    # layout that tends to float onto a separate page.
    n_cols = 2
    n_rows = max(1, (len(selected) + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(12.0, 4.15 * n_rows))
    grid = fig.add_gridspec(n_rows, n_cols, wspace=0.08, hspace=0.20)

    selected_meta: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(selected):
        panel_row = row_idx // n_cols
        panel_col = row_idx % n_cols
        cell = grid[panel_row, panel_col].subgridspec(
            2,
            2,
            height_ratios=[1.0, 0.55],
            width_ratios=[1.0, 1.0],
            wspace=0.03,
            hspace=0.05,
        )

        original = fit_image(image_for_row(row, task2_examples), (320, 230))
        heat_path = heatmap_path_from_row(row, visual_dir, task2_dir)
        if heat_path and heat_path.exists():
            heatmap = fit_image(Image.open(heat_path), (320, 230))
        else:
            heatmap = original.copy()

        ax_img = fig.add_subplot(cell[0, 0])
        ax_heat = fig.add_subplot(cell[0, 1])
        ax_text = fig.add_subplot(cell[1, :])
        for ax in (ax_img, ax_heat, ax_text):
            ax.axis("off")

        ax_img.imshow(original)
        ax_img.set_title(f"Sample {row_idx + 1}: Original", fontsize=8.8, weight="bold", pad=2)
        ax_heat.imshow(heatmap)
        ax_heat.set_title("CATA heatmap", fontsize=8.8, weight="bold", pad=2)

        q = wrap_label("Q", row.get("question", ""), 58, 130)
        pred = wrap_label("Pred", row.get("answer", ""), 58, 120)
        exp = wrap_label("Exp", row.get("textual_explanation", ""), 58, 190)
        conf = normalize_text(row.get("confidence_score", ""), 60)
        text = f"{q}\n{pred}\nConf: {conf}\n{exp}"
        ax_text.text(
            0.01,
            0.98,
            text,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.36", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0),
            transform=ax_text.transAxes,
            wrap=True,
        )

        selected_meta.append(
            {
                "val_id": row.get("val_id"),
                "img_id": row.get("img_id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "confidence_score": row.get("confidence_score"),
                "heatmap": str(heat_path) if heat_path else None,
            }
        )

    for ext in ("pdf", "png"):
        path = output_dir / f"cata_qualitative_examples.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=240)
        print(f"Saved {path}")
    plt.close(fig)
    return selected_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Make CATA paper architecture and qualitative figures.")
    parser.add_argument("--task2-jsonl", required=True)
    parser.add_argument("--visual-dir", required=True)
    parser.add_argument("--output-dir", default="paper_assets/figures")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task2_path = Path(args.task2_jsonl)
    if not task2_path.is_absolute():
        task2_path = PROJECT_ROOT / task2_path
    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = PROJECT_ROOT / visual_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(task2_path)
    task2_dir = task2_path.parent

    make_architecture_figure(output_dir)
    selected_meta = make_qualitative_figure(
        rows=rows,
        visual_dir=visual_dir,
        task2_dir=task2_dir,
        output_dir=output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    selected_path = output_dir / "selected_examples.json"
    selected_path.write_text(json.dumps(selected_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {selected_path}")


if __name__ == "__main__":
    main()
