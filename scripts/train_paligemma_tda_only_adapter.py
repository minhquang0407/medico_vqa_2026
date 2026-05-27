"""Run PaliGemma Structural Adapter in TDA-only mode.

This wrapper calls train_paligemma_structural_adapter.py with --structural-mode
tda_only, so the adapter receives only topology-derived patch features. Prior
mask, red map, center map, morpho prior, and global priors are zeroed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="PaliGemma TDA-only structural adapter wrapper")
    p.add_argument("--output-dir", default="outputs/paligemma_tda_only_adapter")
    p.add_argument("--max-train-samples", default=None)
    p.add_argument("--eval-samples", default="1500")
    p.add_argument("--epochs", default="2")
    p.add_argument("--batch-size", default="1")
    p.add_argument("--gradient-accumulation-steps", default="8")
    p.add_argument("--learning-rate", default="2e-5")
    p.add_argument("--num-workers", default="2")
    p.add_argument("--use-augmentation", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()

    cmd = [
        sys.executable, "scripts/train_paligemma_structural_adapter.py",
        "--mode", "train_eval",
        "--output-dir", a.output_dir,
        "--epochs", str(a.epochs),
        "--batch-size", str(a.batch_size),
        "--gradient-accumulation-steps", str(a.gradient_accumulation_steps),
        "--learning-rate", str(a.learning_rate),
        "--eval-samples", str(a.eval_samples),
        "--max-length", "512",
        "--max-new-tokens", "48",
        "--num-workers", str(a.num_workers),
        "--structural-root", ".",
        "--train-structural-manifest", "data/processed/structural_features/train_original_manifest.csv",
        "--eval-structural-manifest", "data/processed/structural_features/test_original_manifest.csv",
        "--structural-mode", "tda_only",
    ]
    if a.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(a.max_train_samples)])
    if a.use_augmentation:
        cmd.append("--use-augmentation")
    cmd.extend(a.extra)
    print("Running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
