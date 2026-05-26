"""Lightweight heatmap visualizer for Medico 2026 Task 2."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def normalize_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + eps)).astype(np.float32)


def resize_map(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a heatmap to PIL size=(width, height)."""

    heat = normalize_map(values)
    img = Image.fromarray(np.uint8(heat * 255), mode="L")
    img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    """Small dependency-free blue-cyan-yellow-red colormap."""

    x = normalize_map(values)
    r = np.clip(1.8 * x - 0.35, 0.0, 1.0)
    g = np.clip(1.8 - np.abs(2.4 * x - 1.2), 0.0, 1.0)
    b = np.clip(1.35 - 1.9 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.42,
    draw_peak: bool = True,
) -> Path:
    """Save an image with a semi-transparent heatmap overlay."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGB")
    heat = resize_map(heatmap, base.size)
    color = turbo_like_colormap(heat)
    base_arr = np.asarray(base).astype(np.float32) / 255.0
    mixed = (1.0 - alpha * heat[..., None]) * base_arr + (alpha * heat[..., None]) * color
    mixed = np.uint8(np.clip(mixed, 0.0, 1.0) * 255)
    out = Image.fromarray(mixed, mode="RGB")

    if draw_peak and float(heat.max()) > 0:
        y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
        draw = ImageDraw.Draw(out)
        radius = max(6, min(base.size) // 32)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255), width=2)
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), outline=(220, 70, 45), width=2)

    out.save(output_path)
    return output_path


def quadrant_from_heatmap(heatmap: np.ndarray) -> str:
    heat = normalize_map(heatmap)
    if float(heat.sum()) <= 1e-8:
        return "the central field"
    h, w = heat.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = float(heat.sum())
    cy = float((ys * heat).sum() / total) / max(1, h - 1)
    cx = float((xs * heat).sum() / total) / max(1, w - 1)

    vertical = "upper" if cy < 0.38 else "lower" if cy > 0.62 else "central"
    horizontal = "left" if cx < 0.38 else "right" if cx > 0.62 else "central"
    if vertical == "central" and horizontal == "central":
        return "central field"
    if vertical == "central":
        return f"{horizontal} field"
    if horizontal == "central":
        return f"{vertical} field"
    return f"{vertical}-{horizontal} field"
