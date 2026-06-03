"""Run B3 clean ablation: TDA visual-token fusion only.

B3 keeps the same pretrained ViT + Qwen LoRA budget as A0, but enables only
post-ViT TDA patch-token fusion and the patch TDA auxiliary loss. It disables
prior/global structural inputs, OT, and LLM TopoAdapter.
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


def build_train_command(args: argparse.Namespace, checkpoint_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train_structural_vqa_generative",
        "--jsonl",
        args.train_jsonl,
        "--images-dir",
        args.images_dir,
        "--structural-manifest",
        args.train_structural_manifest,
        "--output-dir",
        str(checkpoint_dir),
        "--llm-name-or-path",
        args.llm_name_or_path,
        "--vision-backend",
        args.vision_backend,
        "--vision-pretrained",
        "true",
        "--freeze-vision-backbone",
        bool_arg(args.freeze_vision_backbone),
        "--freeze-llm",
        "true",
        "--use-lora",
        "true",
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--max-question-length",
        str(args.max_question_length),
        "--max-answer-length",
        str(args.max_answer_length),
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
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--val-ratio",
        str(args.val_ratio),
        "--device",
        args.device,
        "--structural-mode",
        "tda_only",
        "--visual-structural-mode",
        "tda_only",
        "--use-global-structural-token",
        "false",
        "--use-ot",
        "false",
        "--use-ot-fusion",
        "false",
        "--ot-fusion-mode",
        "none",
        "--use-prior-as-ot-target",
        "false",
        "--use-topological-loss",
        "true",
        "--use-prior-align-loss",
        "false",
        "--use-global-topo-loss",
        "false",
        "--use-patch-topo-loss",
        "true",
        "--prior-loss-weight",
        "0.0",
        "--global-topo-loss-weight",
        "0.0",
        "--patch-topo-loss-weight",
        str(args.patch_topo_loss_weight),
        "--ot-loss-weight",
        "0.0",
        "--disable-tqdm",
        bool_arg(args.disable_tqdm),
        "--log-every-steps",
        str(args.log_every_steps),
        "--log-every-seconds",
        str(args.log_every_seconds),
    ]
    max_samples = optional_positive(args.max_samples)
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if args.save_full_checkpoint:
        command.extend(["--save-full-checkpoint", "true"])
    return command


def build_eval_command(args: argparse.Namespace, checkpoint_dir: Path, eval_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_structural_vqa_generative",
        "--checkpoint",
        str(checkpoint_dir / "last.pt"),
        "--jsonl",
        args.test_jsonl,
        "--images-dir",
        args.images_dir,
        "--structural-manifest",
        args.test_structural_manifest,
        "--output-dir",
        str(eval_dir),
        "--llm-name-or-path",
        args.llm_name_or_path,
        "--vision-backend",
        args.vision_backend,
        "--vision-pretrained",
        "true",
        "--freeze-vision-backbone",
        bool_arg(args.freeze_vision_backbone),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
    ]
    eval_max_samples = optional_positive(args.eval_max_samples)
    if eval_max_samples is not None:
        command.extend(["--max-samples", str(eval_max_samples)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate B3 clean TDA visual-only ablation.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="outputs/ablations/b3_tda_visual_only")
    parser.add_argument("--train-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--test-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--train-structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--test-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")

    parser.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--max-question-length", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=30000, help="Use 0 for full training set.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--patch-topo-loss-weight", type=float, default=0.005)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Use 0 for full test set.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--disable-tqdm", type=str2bool, default=False)
    parser.add_argument("--log-every-steps", type=int, default=5000)
    parser.add_argument("--log-every-seconds", type=int, default=600)
    parser.add_argument("--save-full-checkpoint", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    run_name = args.run_name or f"b3_tda_visual_only_{timestamp()}"
    run_dir = Path(args.output_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ablation_group": "B3",
        "ablation_name": "TDA visual-token fusion only",
        "description": "Post-ViT TDA fusion and patch TDA loss only. No prior/global inputs, OT, or LLM TopoAdapter.",
        "structural_features": {
            "structural_mode": "tda_only",
            "visual_structural_mode": "tda_only",
            "use_global_structural_token": False,
            "use_ot": False,
            "use_ot_fusion": False,
            "use_prior_as_ot_target": False,
            "use_topological_loss": True,
            "use_prior_align_loss": False,
            "use_global_topo_loss": False,
            "use_patch_topo_loss": True,
            "llm_topo_adapter": False,
        },
        "args": vars(args),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "eval_dir": str(eval_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    train_cmd = build_train_command(args, checkpoint_dir)
    eval_cmd = build_eval_command(args, checkpoint_dir, eval_dir)
    paper_metrics_cmd = [sys.executable, "-m", "scripts.evaluate_paper_metrics", "--input", str(eval_dir / "predictions.jsonl")]
    (run_dir / "train_command.json").write_text(json.dumps(train_cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "eval_command.json").write_text(json.dumps(eval_cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "paper_metrics_command.json").write_text(json.dumps(paper_metrics_cmd, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: Dict[str, Any] = {"config": config, "train_returncode": None, "eval_returncode": None, "metrics_returncode": None}
    train_rc = run_command(train_cmd, run_dir / "train.log", repo_dir, env, dry_run=args.dry_run)
    summary["train_returncode"] = train_rc
    if train_rc != 0:
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Training failed with return code {train_rc}. See {run_dir / 'train.log'}")

    eval_rc = run_command(eval_cmd, run_dir / "eval.log", repo_dir, env, dry_run=args.dry_run)
    summary["eval_returncode"] = eval_rc
    if eval_rc != 0:
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Evaluation failed with return code {eval_rc}. See {run_dir / 'eval.log'}")

    metrics_rc = run_command(paper_metrics_cmd, run_dir / "paper_metrics.log", repo_dir, env, dry_run=args.dry_run)
    summary["metrics_returncode"] = metrics_rc
    summary["predictions_path"] = str(eval_dir / "predictions.jsonl")
    summary["paper_metrics_path"] = str(eval_dir / "paper_metrics.json")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if metrics_rc != 0:
        raise SystemExit(f"Metrics failed with return code {metrics_rc}. See {run_dir / 'paper_metrics.log'}")

    print("\n✅ B3 TDA visual-only ablation complete")
    print(f"Run dir:     {run_dir}")
    print(f"Predictions: {eval_dir / 'predictions.jsonl'}")
    print(f"Metrics:     {eval_dir / 'paper_metrics.json'}")


if __name__ == "__main__":
    main()
