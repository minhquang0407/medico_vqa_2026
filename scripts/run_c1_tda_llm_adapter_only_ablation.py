"""Run C1 clean ablation: TDA-conditioned LLM TopoAdapter only.

C1 disables all visual-side structural paths and base structural losses. Real TDA
features are used only to condition gated TopoAdapters inside the Qwen decoder.
Training optimizes LM loss plus the adapter gate loss.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def bool_arg(value: bool) -> str:
    return "true" if bool(value) else "false"


def optional_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: List[str], log_path: Path, cwd: Path, env: Dict[str, str], dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print("\n" + "=" * 120)
    print(printable)
    print("=" * 120)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(printable + "\n" + "=" * 120 + "\n")

    if dry_run:
        return 0

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)
    return process.wait()


def build_train_eval_command(args: argparse.Namespace, run_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train_qwen3b_curriculum_topo_adapter",
        "--mode",
        "train_eval",
        "--jsonl",
        args.train_jsonl,
        "--eval-jsonl",
        args.test_jsonl,
        "--images-dir",
        args.images_dir,
        "--structural-manifest",
        args.train_structural_manifest,
        "--eval-structural-manifest",
        args.test_structural_manifest,
        "--output-dir",
        str(run_dir),
        "--llm-name-or-path",
        args.llm_name_or_path,
        "--topo-mode",
        "tda_only",
        "--visual-structural-mode",
        "none",
        "--use-global-structural-token",
        "false",
        "--use-base-ot",
        "false",
        "--use-base-ot-fusion",
        "false",
        "--use-base-prior-as-ot-target",
        "false",
        "--use-base-topological-loss",
        "false",
        "--use-base-prior-align-loss",
        "false",
        "--use-base-global-topo-loss",
        "false",
        "--use-base-patch-topo-loss",
        "false",
        "--vision-backend",
        args.vision_backend,
        "--vision-pretrained",
        "true",
        "--freeze-vision-backbone",
        bool_arg(args.freeze_vision_backbone),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--bottleneck-dim",
        str(args.bottleneck_dim),
        "--adapter-last-n-layers",
        str(args.adapter_last_n_layers),
        "--adapter-every-n-layers",
        str(args.adapter_every_n_layers),
        "--align-warmup-epochs",
        str(args.align_warmup_epochs),
        "--warmup-text-only",
        bool_arg(args.warmup_text_only),
        "--keep-patch-topo-during-warmup",
        bool_arg(args.keep_patch_topo_during_warmup),
        "--gate-loss-weight",
        str(args.gate_loss_weight),
        "--val-ratio",
        str(args.val_ratio),
    ]
    max_samples = optional_positive(args.max_samples)
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    eval_samples = optional_positive(args.eval_max_samples)
    if eval_samples is not None:
        command.extend(["--eval-samples", str(eval_samples)])
    if args.val_jsonl:
        command.extend(["--val-jsonl", args.val_jsonl])
        if args.val_structural_manifest:
            command.extend(["--val-structural-manifest", args.val_structural_manifest])
    if args.val_max_samples > 0:
        command.extend(["--val-max-samples", str(args.val_max_samples)])
    if args.save_full_checkpoint:
        command.extend(["--save-full-checkpoint", "true"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate C1 clean TDA LLM adapter-only ablation.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="outputs/ablations/c1_tda_llm_adapter_only")
    parser.add_argument("--train-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--test-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--train-structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--val-structural-manifest", default=None)
    parser.add_argument("--test-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")

    parser.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=30000, help="Use 0 for full training set.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--val-max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Use 0 for full test set.")
    parser.add_argument("--max-new-tokens", type=int, default=48)

    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--adapter-last-n-layers", type=int, default=8)
    parser.add_argument("--adapter-every-n-layers", type=int, default=0)
    parser.add_argument("--align-warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-text-only", type=str2bool, default=False)
    parser.add_argument("--keep-patch-topo-during-warmup", type=str2bool, default=False)
    parser.add_argument("--gate-loss-weight", type=float, default=0.001)
    parser.add_argument("--save-full-checkpoint", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    run_name = args.run_name or f"c1_tda_llm_adapter_only_{timestamp()}"
    run_dir = Path(args.output_root) / run_name
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ablation_group": "C1",
        "ablation_name": "TDA LLM adapter-only with gate loss",
        "description": "Visual structural mode is none. Real TDA only conditions gated Qwen TopoAdapters. Loss is LM + gate loss.",
        "structural_features": {
            "topo_mode": "tda_only",
            "visual_structural_mode": "none",
            "use_global_structural_token": False,
            "use_base_ot": False,
            "use_base_ot_fusion": False,
            "use_base_topological_loss": False,
            "use_base_prior_align_loss": False,
            "use_base_global_topo_loss": False,
            "use_base_patch_topo_loss": False,
            "llm_topo_adapter": True,
            "gate_loss_weight": args.gate_loss_weight,
        },
        "args": vars(args),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    train_eval_cmd = build_train_eval_command(args, run_dir)
    paper_metrics_cmd = [sys.executable, "-m", "scripts.evaluate_paper_metrics", "--input", str(eval_dir / "predictions.jsonl")]
    (run_dir / "train_eval_command.json").write_text(json.dumps(train_eval_cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "paper_metrics_command.json").write_text(json.dumps(paper_metrics_cmd, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: Dict[str, Any] = {"config": config, "train_eval_returncode": None, "metrics_returncode": None}
    train_eval_rc = run_command(train_eval_cmd, run_dir / "train_eval.log", repo_dir, env, dry_run=args.dry_run)
    summary["train_eval_returncode"] = train_eval_rc
    if train_eval_rc != 0:
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Training/eval failed with return code {train_eval_rc}. See {run_dir / 'train_eval.log'}")

    metrics_rc = run_command(paper_metrics_cmd, run_dir / "paper_metrics.log", repo_dir, env, dry_run=args.dry_run)
    summary["metrics_returncode"] = metrics_rc
    summary["predictions_path"] = str(eval_dir / "predictions.jsonl")
    summary["paper_metrics_path"] = str(eval_dir / "paper_metrics.json")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if metrics_rc != 0:
        raise SystemExit(f"Metrics failed with return code {metrics_rc}. See {run_dir / 'paper_metrics.log'}")

    print("\n✅ C1 TDA LLM adapter-only ablation complete")
    print(f"Run dir:     {run_dir}")
    print(f"Predictions: {eval_dir / 'predictions.jsonl'}")
    print(f"Metrics:     {eval_dir / 'paper_metrics.json'}")


if __name__ == "__main__":
    main()
