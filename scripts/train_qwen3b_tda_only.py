"""Train Qwen3B Structural VQA in TDA-only mode.

This wrapper intentionally disables prior masks and global priors while keeping
only topology-derived patch features (`topo_features`). It calls the existing
structural generative trainer with settings that zero/remove prior supervision.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Qwen3B TDA-only structural VQA wrapper")
    p.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    p.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    p.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    p.add_argument("--output-dir", default="outputs/qwen3b_tda_only")
    p.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-samples", default=None)
    p.add_argument("--epochs", default="2")
    p.add_argument("--batch-size", default="1")
    p.add_argument("--gradient-accumulation-steps", default="8")
    p.add_argument("--lr", default="3e-5")
    p.add_argument("--seed", default="42")
    p.add_argument("--num-workers", default="2")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()

    cmd = [
        sys.executable, "scripts/train_structural_vqa_generative.py",
        "--jsonl", a.jsonl,
        "--images-dir", a.images_dir,
        "--structural-manifest", a.structural_manifest,
        "--output-dir", a.output_dir,
        "--llm-name-or-path", a.llm_name_or_path,
        "--epochs", str(a.epochs),
        "--batch-size", str(a.batch_size),
        "--gradient-accumulation-steps", str(a.gradient_accumulation_steps),
        "--lr", str(a.lr),
        "--seed", str(a.seed),
        "--num-workers", str(a.num_workers),
        "--vision-pretrained", "true",
        "--freeze-vision-backbone", "true",
        "--freeze-llm", "true",
        "--use-lora", "true",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-target-modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        # TDA-only: no prior mask as OT target and no prior/global auxiliary losses.
        "--use-ot", "true",
        "--use-ot-fusion", "true",
        "--use-prior-as-ot-target", "false",
        "--ot-loss-weight", "0.02",
        "--use-topological-loss", "true",
        "--use-prior-align-loss", "false",
        "--use-global-topo-loss", "false",
        "--use-patch-topo-loss", "true",
        "--prior-loss-weight", "0.0",
        "--global-topo-loss-weight", "0.0",
        "--patch-topo-loss-weight", "0.005",
        "--lr-scheduler", "cosine",
        "--warmup-steps", "1000",
    ]
    if a.max_samples is not None:
        cmd.extend(["--max-samples", str(a.max_samples)])
    cmd.extend(a.extra)
    print("Running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
