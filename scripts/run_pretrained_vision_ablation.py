import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = {
    "A0_no_ot": {
        "use_ot": False,
        "use_ot_fusion": False,
        "use_prior_as_ot_target": True,
        "ot_loss_weight": 0.05,
        "description": "No OT baseline",
    },
    "A1_ot_loss_only": {
        "use_ot": True,
        "use_ot_fusion": False,
        "use_prior_as_ot_target": True,
        "ot_loss_weight": 0.05,
        "description": "OT loss only, no OT prefix fusion",
    },
    "A2_ot_fusion_no_prior": {
        "use_ot": True,
        "use_ot_fusion": True,
        "use_prior_as_ot_target": False,
        "ot_loss_weight": 0.05,
        "description": "OT prefix fusion without prior target + OT loss",
    },
    "A2b_ot_fusion_no_prior_no_loss": {
        "use_ot": True,
        "use_ot_fusion": True,
        "use_prior_as_ot_target": False,
        "ot_loss_weight": 0.0,
        "description": "OT prefix fusion without prior target, no OT loss",
    },
    "A3_prior_ot_fusion": {
        "use_ot": True,
        "use_ot_fusion": True,
        "use_prior_as_ot_target": True,
        "ot_loss_weight": 0.05,
        "description": "Prior-guided OT prefix fusion + OT loss",
    },
    "A3b_prior_ot_fusion_no_loss": {
        "use_ot": True,
        "use_ot_fusion": True,
        "use_prior_as_ot_target": True,
        "ot_loss_weight": 0.0,
        "description": "Prior-guided OT prefix fusion, no OT loss",
    },
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


def run_command(command, dry_run=False):
    print("\n" + "=" * 100)
    print(" ".join(command))
    print("=" * 100 + "\n")
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_train_command(args, ablation_name, cfg):
    output_dir = Path(args.output_root) / f"{args.run_prefix}_{ablation_name}"
    command = [
        sys.executable,
        "-m",
        "scripts.train_structural_vqa_generative",
        "--jsonl",
        args.jsonl,
        "--images-dir",
        args.images_dir,
        "--structural-manifest",
        args.structural_manifest,
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(args.max_samples),
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
        bool_arg(args.freeze_vision_backbone),
        "--freeze-llm",
        bool_arg(args.freeze_llm),
        "--use-lora",
        bool_arg(args.use_lora),
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
        "--use-patch-topo-loss",
        bool_arg(args.use_patch_topo_loss),
    ]
    return command, output_dir


def build_eval_command(args, ablation_name, checkpoint_dir):
    eval_dir = Path(args.eval_root) / f"{args.run_prefix}_{ablation_name}_test{args.eval_max_samples or 'full'}"
    return [
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
        "--output-dir",
        str(eval_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ] + (["--max-samples", str(args.eval_max_samples)] if args.eval_max_samples is not None else [])


def main():
    parser = argparse.ArgumentParser(
        description="Run A0/A1/A2/A2b/A3/A3b pretrained-vision ablations for generative Structural VQA."
    )
    parser.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--test-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--test-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--run-prefix", default="ablation_qwen05b_lora_1k_pretrained_vision")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()), choices=list(ABLATIONS.keys()))

    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--freeze-llm", type=str2bool, default=True)
    parser.add_argument("--use-lora", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-question-length", type=int, default=256)
    parser.add_argument("--max-answer-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-patch-topo-loss", type=str2bool, default=False)

    parser.add_argument("--evaluate", type=str2bool, default=True)
    parser.add_argument("--eval-max-samples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n🧪 Running pretrained-vision ablations:")
    for name in args.ablations:
        print(f"- {name}: {ABLATIONS[name]['description']}")
    print("\nNote: this script forces --vision-pretrained true.\n")

    for ablation_name in args.ablations:
        cfg = ABLATIONS[ablation_name]
        train_command, checkpoint_dir = build_train_command(args, ablation_name, cfg)
        run_command(train_command, dry_run=args.dry_run)

        if args.evaluate:
            eval_command = build_eval_command(args, ablation_name, checkpoint_dir)
            run_command(eval_command, dry_run=args.dry_run)

    print("\n✅ All requested pretrained-vision ablations completed.")


if __name__ == "__main__":
    main()
