import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


A3_PRIOR_OT_FUSION = {
    "use_ot": "true",
    "use_ot_fusion": "true",
    "use_prior_as_ot_target": "true",
    "ot_loss_weight": "0.05",
}


def timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def torch_load_epoch(checkpoint_path: str) -> int:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(checkpoint.get("epoch", 0))


def should_echo_line(line: str, *, quiet: bool, last_progress_time: list[float], progress_interval_sec: int) -> bool:
    if not quiet:
        return True

    stripped = line.lstrip("\r").strip()
    if not stripped:
        return False

    important_prefixes = (
        "===",
        "---",
        "Device:",
        "LLM:",
        "Max samples:",
        "Freeze LLM:",
        "LR scheduler:",
        "trainable params:",
        "Epoch ",
        "Train:",
        "Val:",
        "Preview:",
        "Q:",
        "GT:",
        "GEN:",
        "✅",
        "❌",
        "⚠️",
        "Warning:",
        "Traceback",
        "RuntimeError",
        "FileNotFoundError",
        "ImportError",
        "{",
        "}",
        '"',
    )
    if stripped.startswith(important_prefixes):
        return True

    is_progress = stripped.startswith(("Train epoch ", "Evaluating:", "Loading weights:"))
    if is_progress:
        now = dt.datetime.now().timestamp()
        if now - last_progress_time[0] >= progress_interval_sec or "100%" in stripped:
            last_progress_time[0] = now
            return True
    return False


def run_command(command, log_file: Path, *, quiet: bool = True, progress_interval_sec: int = 300):
    printable = " ".join(map(str, command))
    print("\n" + "=" * 120)
    print(printable)
    print("=" * 120 + "\n")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(printable + "\n")
        f.write("=" * 120 + "\n")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if quiet:
        env["DISABLE_TQDM"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    last_progress_time = [0.0]
    with log_file.open("a", encoding="utf-8") as f:
        for line in process.stdout:
            if should_echo_line(line, quiet=quiet, last_progress_time=last_progress_time, progress_interval_sec=progress_interval_sec):
                print(line, end="")
            f.write(line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {printable}")


def maybe_copy_tree(src: Path, dst_root: Path | None):
    if dst_root is None:
        return None
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return str(dst)


def train_command(args, checkpoint_dir: Path):
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
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--llm-name-or-path",
        args.llm_name_or_path,
        "--vision-backend",
        args.vision_backend,
        "--vision-pretrained",
        "true",
        "--freeze-vision-backbone",
        "true",
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
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--use-ot",
        A3_PRIOR_OT_FUSION["use_ot"],
        "--use-ot-fusion",
        A3_PRIOR_OT_FUSION["use_ot_fusion"],
        "--ot-fusion-mode",
        "prefix",
        "--use-prior-as-ot-target",
        A3_PRIOR_OT_FUSION["use_prior_as_ot_target"],
        "--ot-loss-weight",
        A3_PRIOR_OT_FUSION["ot_loss_weight"],
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-steps",
        str(args.warmup_steps),
        "--use-patch-topo-loss",
        "false",
        "--log-every-steps",
        str(args.log_every_steps),
        "--log-every-seconds",
        str(args.log_every_seconds),
        "--tqdm-mininterval",
        str(args.tqdm_mininterval),
        "--tqdm-miniters",
        str(args.tqdm_miniters),
    ]
    if args.resume_checkpoint:
        command.extend(["--resume-checkpoint", args.resume_checkpoint])
        command.extend(["--resume-reset-scheduler", args.resume_reset_scheduler])
    if args.max_samples and args.max_samples > 0:
        command.extend(["--max-samples", str(args.max_samples)])
    return command


def eval_command(args, checkpoint_path: Path, eval_dir: Path):
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_structural_vqa_generative",
        "--checkpoint",
        str(checkpoint_path),
        "--jsonl",
        args.test_jsonl,
        "--images-dir",
        args.images_dir,
        "--structural-manifest",
        args.test_structural_manifest,
        "--batch-size",
        str(args.eval_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(eval_dir),
    ]
    if args.eval_samples and args.eval_samples > 0:
        command.extend(["--max-samples", str(args.eval_samples)])
    return command


def load_metrics(eval_dir: Path):
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_report(report_path: Path, summary: dict):
    rows = []
    for item in summary["epochs"]:
        metrics = item.get("metrics", {})
        rows.append(
            [
                str(item["epoch"]),
                f"{metrics.get('exact_match', 0):.4f}",
                f"{metrics.get('token_f1', 0):.4f}",
                f"{metrics.get('rouge_l', 0):.4f}",
                f"{metrics.get('bleu_1', 0):.4f}",
                f"{metrics.get('qtype_yes_no_answer_acc', 0):.4f}",
                f"{metrics.get('qtype_numerical_count_answer_acc', 0):.4f}",
                item["checkpoint"],
                item["eval_dir"],
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Full A3 Epoch Evaluation Report\n\n")
        f.write(f"Run ID: `{summary['run_id']}`\n\n")
        f.write("| Epoch | EM | Token F1 | ROUGE-L | BLEU-1 | Yes/No | Count | Checkpoint | Eval |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Train A3 full data and evaluate every saved epoch checkpoint.")
    parser.add_argument("--run-prefix", default="colab_a100_qwen3b_a3_full_2ep_epoch_eval")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--log-dir", default="logs/overnight")
    parser.add_argument("--backup-dir", default=None, help="Optional Drive directory to copy checkpoints/eval/logs after completion.")
    parser.add_argument("--quiet", default="true", choices=["true", "false"], help="Reduce child-process console output; full logs are still written to file.")
    parser.add_argument("--progress-interval-sec", type=int, default=300, help="When quiet=true, print progress lines at most once per this many seconds.")
    parser.add_argument("--log-every-steps", type=int, default=5000, help="Training progress line interval when tqdm is disabled.")
    parser.add_argument("--log-every-seconds", type=int, default=600, help="Training progress time interval when tqdm is disabled.")
    parser.add_argument("--tqdm-mininterval", type=float, default=60.0, help="Minimum seconds between tqdm refreshes when quiet=false.")
    parser.add_argument("--tqdm-miniters", type=int, default=1000, help="Minimum iterations between tqdm refreshes when quiet=false.")

    parser.add_argument("--train-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--test-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--train-structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--test-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")

    parser.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--resume-checkpoint", default=None, help="Resume training from a checkpoint created by train_structural_vqa_generative.py.")
    parser.add_argument("--resume-reset-scheduler", default="false", choices=["true", "false"], help="When resuming, reset LR scheduler instead of loading it from checkpoint.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full training set.")
    parser.add_argument("--eval-samples", type=int, default=2000, help="0 means full test set.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-question-length", type=int, default=256)
    parser.add_argument("--max-answer-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_samples <= 0:
        args.max_samples = None
    if args.eval_samples <= 0:
        args.eval_samples = None

    run_id = f"{args.run_prefix}_{timestamp()}"
    checkpoint_dir = Path(args.output_root) / f"{run_id}_A3_prior_ot_fusion"
    log_file = Path(args.log_dir) / f"{run_id}.log"
    summary_path = Path(args.log_dir) / f"{run_id}_summary.json"
    report_path = Path(args.log_dir) / f"{run_id}_report.md"

    print("\n🚀 FULL A3 TRAIN + PER-EPOCH EVAL")
    print(f"Run ID: {run_id}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Eval root: {args.eval_root}")
    print(f"Log: {log_file}")

    summary = {"run_id": run_id, "args": vars(args), "checkpoint_dir": str(checkpoint_dir), "epochs": []}

    quiet = args.quiet.lower() == "true"

    run_command(train_command(args, checkpoint_dir), log_file, quiet=quiet, progress_interval_sec=args.progress_interval_sec)

    eval_start_epoch = 1
    if args.resume_checkpoint:
        resumed = torch_load_epoch(args.resume_checkpoint)
        eval_start_epoch = resumed + 1

    for epoch in range(eval_start_epoch, args.epochs + 1):
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing epoch checkpoint: {checkpoint_path}")
        eval_dir = Path(args.eval_root) / f"{run_id}_epoch_{epoch}_test{args.eval_samples or 'full'}"
        run_command(eval_command(args, checkpoint_path, eval_dir), log_file, quiet=quiet, progress_interval_sec=args.progress_interval_sec)
        item = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "eval_dir": str(eval_dir),
            "metrics": load_metrics(eval_dir),
        }
        summary["epochs"].append(item)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(report_path, summary)

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir is not None:
        copied = {
            "checkpoints": maybe_copy_tree(checkpoint_dir, backup_dir / "checkpoints"),
            "logs": None,
            "eval": [],
        }
        (backup_dir / "logs").mkdir(parents=True, exist_ok=True)
        for path in [log_file, summary_path, report_path]:
            if path.exists():
                shutil.copy2(path, backup_dir / "logs" / path.name)
        copied["logs"] = str(backup_dir / "logs")
        for item in summary["epochs"]:
            copied["eval"].append(maybe_copy_tree(Path(item["eval_dir"]), backup_dir / "eval"))
        summary["backup"] = copied
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n✅ Full train + per-epoch eval complete")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Log:     {log_file}")


if __name__ == "__main__":
    main()
