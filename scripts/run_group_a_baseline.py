"""Run Group A ablation: ViT pretrained + Qwen baseline.

This runner trains and evaluates a non-structural baseline for the Medico VQA
paper/ablation table. It intentionally disables OT fusion and all topological
losses while keeping the pretrained ViT image encoder and Qwen LoRA setup.

Outputs per run:
  - <output-root>/<run-id>/config.json
  - <output-root>/<run-id>/train_command.json
  - <output-root>/<run-id>/eval_command.json
  - <output-root>/<run-id>/paper_metrics_command.json
  - <output-root>/<run-id>/train.log
  - <output-root>/<run-id>/eval.log
  - <output-root>/<run-id>/checkpoints/{epoch_*.pt,last.pt,train_log.jsonl}
  - <output-root>/<run-id>/eval/predictions.jsonl
  - <output-root>/<run-id>/summary.json

Example:
  python scripts/run_group_a_baseline.py --epochs 1 --max-samples 30000

Full train + full-test generation:
  python scripts/run_group_a_baseline.py --epochs 1 --max-samples 0 --eval-max-samples 0

After generation, compute paper metrics separately:
  python scripts/evaluate_paper_metrics.py --input <output-root>/<run-id>/eval/predictions.jsonl
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
        # Group A baseline: no structural OT/fusion/losses.
        "--use-ot",
        "false",
        "--use-ot-fusion",
        "false",
        "--ot-fusion-mode",
        "none",
        "--use-prior-as-ot-target",
        "false",
        "--use-topological-loss",
        "false",
        "--use-prior-align-loss",
        "false",
        "--use-global-topo-loss",
        "false",
        "--use-patch-topo-loss",
        "false",
        "--prior-loss-weight",
        "0.0",
        "--global-topo-loss-weight",
        "0.0",
        "--patch-topo-loss-weight",
        "0.0",
        "--ot-loss-weight",
        "0.0",
        "--structural-mode",
        "all",
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


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Group A ViT-pretrained + Qwen baseline.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="outputs/ablations/group_a_baseline")
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

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=30000, help="Use 0 for full training set.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation inference.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="Number of test samples to generate. Use 0 for the full test set. Use 1500 only for the official HF public subset.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--disable-tqdm", type=str2bool, default=False)
    parser.add_argument("--log-every-steps", type=int, default=5000)
    parser.add_argument("--log-every-seconds", type=int, default=600)
    parser.add_argument("--save-full-checkpoint", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    run_name = args.run_name or f"group_a_vit_pretrained_qwen_{timestamp()}"
    run_dir = Path(args.output_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ablation_group": "A",
        "ablation_name": "ViT pretrained + Qwen LoRA baseline",
        "description": "No OT fusion and no topological losses. Structural tensors are loaded by the dataset but disabled in model config.",
        "structural_features": {
            "use_ot": False,
            "use_ot_fusion": False,
            "use_prior_as_ot_target": False,
            "use_topological_loss": False,
            "use_prior_align_loss": False,
            "use_global_topo_loss": False,
            "use_patch_topo_loss": False,
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
    paper_metrics_cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_paper_metrics",
        "--input",
        str(eval_dir / "predictions.jsonl"),
    ]
    (run_dir / "train_command.json").write_text(json.dumps(train_cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "eval_command.json").write_text(json.dumps(eval_cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "paper_metrics_command.json").write_text(json.dumps(paper_metrics_cmd, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions_path = eval_dir / "predictions.jsonl"
    summary: Dict[str, Any] = {
        "config": config,
        "train_returncode": None,
        "eval_returncode": None,
        "predictions_path": str(predictions_path),
        "paper_metrics_command": paper_metrics_cmd,
    }

    train_rc = run_command(train_cmd, run_dir / "train.log", repo_dir, env, dry_run=args.dry_run)
    summary["train_returncode"] = train_rc
    if train_rc != 0:
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Training failed with return code {train_rc}. See {run_dir / 'train.log'}")

    eval_rc = run_command(eval_cmd, run_dir / "eval.log", repo_dir, env, dry_run=args.dry_run)
    summary["eval_returncode"] = eval_rc
    summary["predictions_exists"] = False if args.dry_run else predictions_path.exists()
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if eval_rc != 0:
        raise SystemExit(f"Evaluation failed with return code {eval_rc}. See {run_dir / 'eval.log'}")

    print("\n✅ Group A baseline complete")
    print(f"Run dir:      {run_dir}")
    print(f"Config:       {run_dir / 'config.json'}")
    print(f"Predictions:  {predictions_path}")
    print("Next metrics command:")
    print(" ".join(paper_metrics_cmd))


if __name__ == "__main__":
    main()
