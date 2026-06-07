"""Create a reliability/calibration diagram for CATA confidence scores.

This script uses *real* confidence scores from a Task 2 JSONL submission and
empirical correctness from an automatic judge result file. By default, a sample
is treated as correct when judge `correctness >= 3` (acceptable or better on the
1--5 judge scale). This is a reliability-style calibration analysis for the
paper, not a clinical calibration study.

Outputs:
- paper_assets/figures/calibration_curve.pdf
- paper_assets/figures/calibration_curve.png
- paper_assets/figures/calibration_scores.json
- paper_assets/tables/calibration_macros.tex

Example:
    py -3 scripts/make_calibration_curve.py

Optional baseline comparison is supported only when you provide a baseline
submission with `confidence_score` and a matching judge-result JSONL. The script
will not invent baseline confidence values.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CalibrationPoint:
    val_id: str
    confidence: float
    accuracy: float
    correctness_score: float


@dataclass
class BinStats:
    index: int
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    gap: float


@dataclass
class CalibrationStats:
    label: str
    points: List[CalibrationPoint]
    bins: List[BinStats]
    ece: float
    mean_confidence: float
    mean_accuracy: float


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            row["__line_number"] = line_number
            rows.append(row)
    return rows


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def normalize_val_id(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text


def load_submission_confidence(path: Path, confidence_key: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    by_val: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if "val_id" not in row:
            continue
        val_id = normalize_val_id(row.get("val_id"))
        conf = to_float(row.get(confidence_key))
        if conf is None:
            continue
        row = dict(row)
        row[confidence_key] = max(0.0, min(1.0, conf))
        by_val[val_id] = row
    return by_val


def correctness_to_accuracy(score: float, mode: str, threshold: float) -> float:
    if mode == "binary":
        return 1.0 if score >= threshold else 0.0
    if mode == "soft_1_5":
        # Map judge score 1..5 to [0, 1].
        return max(0.0, min(1.0, (score - 1.0) / 4.0))
    if mode == "soft_div5":
        return max(0.0, min(1.0, score / 5.0))
    raise ValueError(f"Unknown accuracy mode: {mode}")


def collect_points(
    submission_path: Path,
    judge_path: Path,
    confidence_key: str,
    accuracy_mode: str,
    correct_threshold: float,
) -> List[CalibrationPoint]:
    submissions = load_submission_confidence(submission_path, confidence_key)
    judge_rows = load_jsonl(judge_path)
    points: List[CalibrationPoint] = []

    for row in judge_rows:
        status = str(row.get("status", "ok")).lower()
        if status not in {"", "ok", "success", "done"}:
            continue
        if "val_id" not in row:
            continue
        val_id = normalize_val_id(row.get("val_id"))
        sub = submissions.get(val_id)
        if sub is None:
            continue
        correctness = to_float(row.get("correctness"))
        confidence = to_float(sub.get(confidence_key))
        if correctness is None or confidence is None:
            continue
        acc = correctness_to_accuracy(correctness, accuracy_mode, correct_threshold)
        points.append(
            CalibrationPoint(
                val_id=val_id,
                confidence=max(0.0, min(1.0, confidence)),
                accuracy=acc,
                correctness_score=correctness,
            )
        )
    if not points:
        raise ValueError(
            f"No matched calibration points between submission={submission_path} "
            f"and judge={judge_path}. Check val_id and confidence_score."
        )
    return points


def compute_bins(points: Sequence[CalibrationPoint], num_bins: int) -> Tuple[List[BinStats], float]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    conf = np.array([p.confidence for p in points], dtype=float)
    acc = np.array([p.accuracy for p in points], dtype=float)
    n = len(points)
    bins: List[BinStats] = []
    ece = 0.0

    for i in range(num_bins):
        lower = float(edges[i])
        upper = float(edges[i + 1])
        if i == num_bins - 1:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                BinStats(
                    index=i,
                    lower=lower,
                    upper=upper,
                    count=0,
                    mean_confidence=float("nan"),
                    empirical_accuracy=float("nan"),
                    gap=float("nan"),
                )
            )
            continue
        mean_conf = float(conf[mask].mean())
        mean_acc = float(acc[mask].mean())
        gap = abs(mean_acc - mean_conf)
        ece += (count / n) * gap
        bins.append(
            BinStats(
                index=i,
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_conf,
                empirical_accuracy=mean_acc,
                gap=gap,
            )
        )
    return bins, float(ece)


def compute_stats(label: str, points: Sequence[CalibrationPoint], num_bins: int) -> CalibrationStats:
    bins, ece = compute_bins(points, num_bins)
    return CalibrationStats(
        label=label,
        points=list(points),
        bins=bins,
        ece=ece,
        mean_confidence=float(np.mean([p.confidence for p in points])),
        mean_accuracy=float(np.mean([p.accuracy for p in points])),
    )


def finite_bin_xy(stats: CalibrationStats) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    counts: List[int] = []
    for b in stats.bins:
        if b.count <= 0 or not math.isfinite(b.mean_confidence) or not math.isfinite(b.empirical_accuracy):
            continue
        xs.append(b.mean_confidence)
        ys.append(b.empirical_accuracy)
        counts.append(b.count)
    return np.array(xs), np.array(ys), np.array(counts)


def plot_calibration(stats_list: Sequence[CalibrationStats], output_dir: Path, accuracy_label: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.4, label="Perfect calibration", zorder=1)

    palette = ["#D55E00", "#0072B2", "#1f2937", "#009E73"]
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "-.", "--", ":"]

    for idx, stats in enumerate(stats_list):
        xs, ys, counts = finite_bin_xy(stats)
        if len(xs) == 0:
            continue
        color = palette[idx % len(palette)]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        counts = counts[order]
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            markersize=6.5,
            linewidth=2.3 if idx == 0 else 1.9,
            linestyle=linestyles[idx % len(linestyles)],
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=f"{stats.label} (ECE={stats.ece:.3f}, n={len(stats.points)})",
            zorder=3 + idx,
        )
        # Visualize calibration gap without hiding the curves.
        ax.vlines(xs, np.minimum(xs, ys), np.maximum(xs, ys), color=color, alpha=0.20, linewidth=4.0, zorder=2)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel(accuracy_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(color="#e5e7eb", linestyle="--", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.96, edgecolor="#374151")
    ax.set_title("Reliability and Calibration of CATA Confidence")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = output_dir / f"calibration_curve.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved {out}")
    plt.close(fig)


def stats_to_json(stats: CalibrationStats) -> Dict[str, Any]:
    return {
        "label": stats.label,
        "n": len(stats.points),
        "ece": round(stats.ece, 6),
        "mean_confidence": round(stats.mean_confidence, 6),
        "mean_accuracy": round(stats.mean_accuracy, 6),
        "bins": [
            {
                "index": b.index,
                "range": [round(b.lower, 4), round(b.upper, 4)],
                "count": b.count,
                "mean_confidence": None if not math.isfinite(b.mean_confidence) else round(b.mean_confidence, 6),
                "empirical_accuracy": None if not math.isfinite(b.empirical_accuracy) else round(b.empirical_accuracy, 6),
                "gap": None if not math.isfinite(b.gap) else round(b.gap, 6),
            }
            for b in stats.bins
        ],
    }


def write_outputs(
    stats_list: Sequence[CalibrationStats],
    output_dir: Path,
    table_dir: Path,
    num_bins: int,
    accuracy_mode: str,
    correct_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_bins": num_bins,
        "accuracy_mode": accuracy_mode,
        "correct_threshold": correct_threshold,
        "accuracy_definition": (
            f"judge correctness >= {correct_threshold:g}"
            if accuracy_mode == "binary"
            else "normalized judge correctness score"
        ),
        "models": [stats_to_json(s) for s in stats_list],
        "note": (
            "Calibration is computed against automatic judge correctness for Task 2. "
            "This is a reliability-style benchmark analysis, not clinical probability calibration."
        ),
    }
    json_path = output_dir / "calibration_scores.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")

    # Convenience macros for LaTeX after the script has been run.
    first = stats_list[0]
    macros = [
        "% Auto-generated by scripts/make_calibration_curve.py",
        f"\\newcommand{{\\CalibrationN}}{{{len(first.points)}}}",
        f"\\newcommand{{\\CATAECE}}{{{first.ece:.3f}}}",
        f"\\newcommand{{\\CATAMeanConfidence}}{{{first.mean_confidence:.3f}}}",
        f"\\newcommand{{\\CATAMeanAccuracy}}{{{first.mean_accuracy:.3f}}}",
    ]
    if len(stats_list) > 1:
        second = stats_list[1]
        macros.extend(
            [
                f"\\newcommand{{\\BaselineECE}}{{{second.ece:.3f}}}",
                f"\\newcommand{{\\BaselineMeanConfidence}}{{{second.mean_confidence:.3f}}}",
                f"\\newcommand{{\\BaselineMeanAccuracy}}{{{second.mean_accuracy:.3f}}}",
            ]
        )
    macro_path = table_dir / "calibration_macros.tex"
    macro_path.write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"Saved {macro_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a publication-ready calibration curve for CATA confidence.")
    parser.add_argument("--cata-submission", default="hf_submission/cata_multitask_final/submission_task2.jsonl")
    parser.add_argument("--cata-judge-results", default="paper_assets/task2_judge_qwen72b_4bit/task2_qwen_judge_results.jsonl")
    parser.add_argument("--cata-label", default="CATA-Final")
    parser.add_argument("--baseline-submission", default="", help="Optional baseline Task 2 JSONL with confidence_score.")
    parser.add_argument("--baseline-judge-results", default="", help="Optional baseline judge JSONL matched by val_id.")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--confidence-key", default="confidence_score")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument(
        "--accuracy-mode",
        choices=["binary", "soft_1_5", "soft_div5"],
        default="binary",
        help="binary uses correctness >= threshold; soft modes use normalized judge correctness.",
    )
    parser.add_argument("--correct-threshold", type=float, default=3.0)
    parser.add_argument("--output-dir", default="paper_assets/figures")
    parser.add_argument("--table-dir", default="paper_assets/tables")
    args = parser.parse_args()

    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2")

    output_dir = resolve_path(args.output_dir)
    table_dir = resolve_path(args.table_dir)

    cata_points = collect_points(
        submission_path=resolve_path(args.cata_submission),
        judge_path=resolve_path(args.cata_judge_results),
        confidence_key=args.confidence_key,
        accuracy_mode=args.accuracy_mode,
        correct_threshold=args.correct_threshold,
    )
    stats_list: List[CalibrationStats] = [compute_stats(args.cata_label, cata_points, args.num_bins)]

    if args.baseline_submission or args.baseline_judge_results:
        if not args.baseline_submission or not args.baseline_judge_results:
            raise ValueError("Provide both --baseline-submission and --baseline-judge-results, or neither.")
        baseline_points = collect_points(
            submission_path=resolve_path(args.baseline_submission),
            judge_path=resolve_path(args.baseline_judge_results),
            confidence_key=args.confidence_key,
            accuracy_mode=args.accuracy_mode,
            correct_threshold=args.correct_threshold,
        )
        stats_list.append(compute_stats(args.baseline_label, baseline_points, args.num_bins))
    else:
        print("No baseline confidence/judge files supplied; plotting CATA only.")
        print("The script does not invent baseline confidence values.")

    accuracy_label = (
        f"Empirical Acceptable Rate (Correctness ≥ {args.correct_threshold:g})"
        if args.accuracy_mode == "binary"
        else "Mean Judge Correctness (Normalized)"
    )
    plot_calibration(stats_list, output_dir, accuracy_label)
    write_outputs(stats_list, output_dir, table_dir, args.num_bins, args.accuracy_mode, args.correct_threshold)

    for stats in stats_list:
        print(
            f"{stats.label}: n={len(stats.points)} | ECE={stats.ece:.4f} | "
            f"mean_conf={stats.mean_confidence:.4f} | mean_acc={stats.mean_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
