import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


A3_PRIOR_OT_FUSION = {
    "name": "A3_prior_ot_fusion",
    "use_ot": True,
    "use_ot_fusion": True,
    "use_prior_as_ot_target": True,
    "ot_loss_weight": 0.05,
}

A0_NO_OT = {
    "name": "A0_no_ot",
    "use_ot": False,
    "use_ot_fusion": False,
    "use_prior_as_ot_target": True,
    "ot_loss_weight": 0.05,
}

A3B_PRIOR_OT_FUSION_NO_OT_LOSS = {
    "name": "A3b_prior_ot_fusion_no_ot_loss",
    "use_ot": True,
    "use_ot_fusion": True,
    "use_prior_as_ot_target": True,
    "ot_loss_weight": 0.0,
}

A3_PRIOR_OT_FUSION_LORA16 = {
    **A3_PRIOR_OT_FUSION,
    "name": "A3_prior_ot_fusion_lora16",
    "lora_r": 16,
    "lora_alpha": 32,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_progress_line(line: str) -> bool:
    stripped = line.lstrip("\r").strip()
    progress_prefixes = (
        "Train epoch ",
        "Evaluating:",
        "Structural precompute",
        "Loading weights:",
    )
    return any(stripped.startswith(prefix) for prefix in progress_prefixes)


def compact_progress_text(line: str) -> str:
    return line.lstrip("\r").strip()


def run_command(command, log_file: Path, dry_run=False):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print("\n" + "=" * 120)
    print(printable)
    print("=" * 120 + "\n")
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(printable + "\n")
        f.write("=" * 120 + "\n")

    if dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    last_progress_at = 0.0
    progress_active = False
    with log_file.open("a", encoding="utf-8") as f:
        for line in process.stdout:
            f.write(line)
            if is_progress_line(line):
                now = time.monotonic()
                text = compact_progress_text(line)
                is_final_progress = "100%" in text
                if is_final_progress or now - last_progress_at >= 5.0:
                    print("\r" + text[:180], end="", flush=True)
                    last_progress_at = now
                    progress_active = True
                continue

            if progress_active:
                print()
                progress_active = False
            print(line, end="")
    return process.wait()


def add_optional_max_samples(command, flag_name, value):
    if value is not None and value > 0:
        command.extend([flag_name, str(value)])
    return command


def train_command(args, cfg, output_dir: Path, max_samples, epochs):
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
        str(output_dir),
        "--epochs",
        str(epochs),
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
        bool_arg(args.freeze_vision_backbone),
        "--freeze-llm",
        "true",
        "--use-lora",
        "true",
        "--lora-r",
        str(cfg.get("lora_r", args.lora_r)),
        "--lora-alpha",
        str(cfg.get("lora_alpha", args.lora_alpha)),
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
        bool_arg(cfg["use_ot"]),
        "--use-ot-fusion",
        bool_arg(cfg["use_ot_fusion"]),
        "--ot-fusion-mode",
        "prefix",
        "--use-prior-as-ot-target",
        bool_arg(cfg["use_prior_as_ot_target"]),
        "--ot-loss-weight",
        str(cfg["ot_loss_weight"]),
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-steps",
        str(args.warmup_steps),
        "--use-patch-topo-loss",
        bool_arg(args.use_patch_topo_loss),
    ]
    return add_optional_max_samples(command, "--max-samples", max_samples)


def eval_command(args, checkpoint_dir: Path, eval_dir: Path, max_samples):
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
        "--batch-size",
        str(args.eval_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(eval_dir),
    ]
    return add_optional_max_samples(command, "--max-samples", max_samples)


def load_metrics(eval_dir: Path):
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_markdown_report(summary, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in summary["jobs"]:
        metrics = job.get("metrics") or {}
        rows.append(
            [
                job["stage"],
                job["config"],
                str(job["train_samples"] or "full"),
                str(job["eval_samples"] or "full"),
                f"{metrics.get('exact_match', 0):.4f}",
                f"{metrics.get('token_f1', 0):.4f}",
                f"{metrics.get('rouge_l', 0):.4f}",
                f"{metrics.get('bleu_1', 0):.4f}",
                f"{metrics.get('concept_yes_no_acc', 0):.4f}",
                f"{metrics.get('concept_count_acc', 0):.4f}",
            ]
        )

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Overnight Full Evaluation Report\n\n")
        f.write(f"Run ID: `{summary['run_id']}`\n\n")
        f.write("| Stage | Config | Train samples | Eval samples | EM | Token F1 | ROUGE-L | BLEU-1 | Yes/No | Count |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n## Output Directories\n\n")
        for job in summary["jobs"]:
            f.write(f"- **{job['stage']} / {job['config']}**\n")
            f.write(f"  - Checkpoint: `{job['checkpoint_dir']}`\n")
            f.write(f"  - Eval: `{job['eval_dir']}`\n")


def selected_configs(plan):
    if plan == "best_only":
        return [A3_PRIOR_OT_FUSION]
    if plan == "a3b_only":
        return [A3B_PRIOR_OT_FUSION_NO_OT_LOSS]
    if plan == "a3_lora16_only":
        return [A3_PRIOR_OT_FUSION_LORA16]
    if plan == "option_a_b":
        return [A3B_PRIOR_OT_FUSION_NO_OT_LOSS, A3_PRIOR_OT_FUSION_LORA16]
    if plan == "compare_a0_a3":
        return [A0_NO_OT, A3_PRIOR_OT_FUSION]
    raise ValueError(f"Unknown plan: {plan}")


def selected_stages(args):
    stages = []
    if args.stage in {"all", "tenk"}:
        stages.append(
            {
                "stage": "10k",
                "train_samples": args.tenk_samples,
                "epochs": args.tenk_epochs,
                "eval_samples": args.tenk_eval_samples,
            }
        )
    if args.stage in {"all", "full"}:
        stages.append(
            {
                "stage": "full",
                "train_samples": None,
                "epochs": args.full_epochs,
                "eval_samples": args.full_eval_samples,
            }
        )
    return stages


def main():
    parser = argparse.ArgumentParser(
        description="One-command overnight runner: train 10k -> eval, then full data -> eval."
    )
    parser.add_argument(
        "--plan",
        choices=["best_only", "compare_a0_a3", "a3b_only", "a3_lora16_only", "option_a_b"],
        default="best_only",
    )
    parser.add_argument("--stage", choices=["all", "tenk", "full"], default="all")
    parser.add_argument("--run-prefix", default="overnight_qwen3b_a3_10k_then_full")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--log-dir", default="logs/overnight")

    parser.add_argument("--train-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--test-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--train-structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--test-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")

    parser.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-question-length", type=int, default=256)
    parser.add_argument("--max-answer-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-patch-topo-loss", type=str2bool, default=False)

    parser.add_argument("--tenk-samples", type=int, default=10000)
    parser.add_argument("--tenk-epochs", type=int, default=2)
    parser.add_argument("--tenk-eval-samples", type=int, default=1000, help="Use 0 for full test after 10k stage.")
    parser.add_argument("--full-epochs", type=int, default=1)
    parser.add_argument("--full-eval-samples", type=int, default=0, help="Use 0 for full test after full-data stage.")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tenk_eval_samples <= 0:
        args.tenk_eval_samples = None
    if args.full_eval_samples <= 0:
        args.full_eval_samples = None

    run_id = f"{args.run_prefix}_{timestamp()}"
    log_file = Path(args.log_dir) / f"{run_id}.log"
    summary_path = Path(args.log_dir) / f"{run_id}_summary.json"
    report_path = Path(args.log_dir) / f"{run_id}_report.md"
    configs = selected_configs(args.plan)
    stages = selected_stages(args)

    summary = {
        "run_id": run_id,
        "plan": args.plan,
        "stage": args.stage,
        "args": vars(args),
        "jobs": [],
    }

    print("\n🌙 OVERNIGHT 10K → FULL GENERATIVE VQA RUNNER")
    print(f"Run ID: {run_id}")
    print(f"Plan: {args.plan}")
    print(f"Stages: {[s['stage'] for s in stages]}")
    print(f"Log: {log_file}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")

    stop = False
    for stage_cfg in stages:
        for cfg in configs:
            stage = stage_cfg["stage"]
            train_samples = stage_cfg["train_samples"]
            eval_samples = stage_cfg["eval_samples"]
            checkpoint_dir = Path(args.output_root) / f"{run_id}_{stage}_{cfg['name']}"
            eval_name = f"{run_id}_{stage}_{cfg['name']}_test{eval_samples or 'full'}"
            eval_dir = Path(args.eval_root) / eval_name
            job = {
                "stage": stage,
                "config": cfg["name"],
                "train_samples": train_samples,
                "epochs": stage_cfg["epochs"],
                "eval_samples": eval_samples,
                "checkpoint_dir": str(checkpoint_dir),
                "eval_dir": str(eval_dir),
                "train_returncode": None,
                "eval_returncode": None,
                "metrics": None,
            }

            rc = run_command(
                train_command(args, cfg, checkpoint_dir, train_samples, stage_cfg["epochs"]),
                log_file,
                dry_run=args.dry_run,
            )
            job["train_returncode"] = rc
            if rc != 0:
                print(f"❌ Train failed: {stage}/{cfg['name']} return code={rc}")
                summary["jobs"].append(job)
                stop = True
                break

            rc = run_command(
                eval_command(args, checkpoint_dir, eval_dir, eval_samples),
                log_file,
                dry_run=args.dry_run,
            )
            job["eval_returncode"] = rc
            if rc != 0:
                print(f"❌ Eval failed: {stage}/{cfg['name']} return code={rc}")
                summary["jobs"].append(job)
                stop = True
                break

            if not args.dry_run:
                job["metrics"] = load_metrics(eval_dir)
            summary["jobs"].append(job)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            make_markdown_report(summary, report_path)

        if stop:
            break

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    make_markdown_report(summary, report_path)

    print("\n✅ Overnight 10k/full runner finished.")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Log:     {log_file}")


if __name__ == "__main__":
    main()
