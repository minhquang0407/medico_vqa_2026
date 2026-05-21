import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data_pipeline.dataset import MedicoVQADataset, medico_vqa_collate_fn
from src.models.structural_vqa_generative import build_structural_generative_vqa


try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for this script
    np = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Không parse được bool: {value}")


def set_reproducible_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def batch_to_device(batch, device):
    for key in ["image", "prior_mask", "topo_features", "global_features"]:
        batch[key] = batch[key].to(device)
    return batch


def scalar(value):
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def build_lr_scheduler(optimizer, scheduler_name: str, warmup_steps: int, total_update_steps: int):
    scheduler_name = str(scheduler_name or "none").lower()
    warmup_steps = max(int(warmup_steps), 0)
    total_update_steps = max(int(total_update_steps), 1)

    if scheduler_name == "none":
        return None

    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(float(current_step + 1) / float(warmup_steps), 1e-8)
        progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "linear":
            return max(1.0 - progress, 0.0)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown lr scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(model, dataloader, optimizer, device, epoch, train=True, gradient_accumulation_steps=1, scheduler=None):
    model.train(train)
    totals = {
        "loss": 0.0,
        "lm_loss": 0.0,
        "ot_cost": 0.0,
        "prior_align_loss": 0.0,
        "global_topo_loss": 0.0,
        "patch_topo_loss": 0.0,
        "topological_loss": 0.0,
    }
    steps = 0
    desc = f"Train epoch {epoch}" if train else f"Eval epoch {epoch}"
    iterator = tqdm(dataloader, desc=desc)

    for batch in iterator:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            out = model(
                image=batch["image"],
                prior_mask=batch["prior_mask"],
                topo_features=batch["topo_features"],
                global_features=batch["global_features"],
                question_text=batch["question_text"],
                answer_text=batch["answer_text"],
                return_diagnostics=True,
            )
            loss = out["loss"]
            if train:
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                if steps % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        step_values = {key: scalar(out.get(key)) for key in totals}
        for key, value in step_values.items():
            totals[key] += value
        steps += 1
        iterator.set_postfix(loss=step_values["loss"], lm=step_values["lm_loss"], lr=current_lr(optimizer))

    if train and steps % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def preview_generation(model, dataloader, device, max_new_tokens=48):
    model.eval()
    batch = next(iter(dataloader))
    batch = batch_to_device(batch, device)
    generated = model.generate(
        image=batch["image"],
        prior_mask=batch["prior_mask"],
        topo_features=batch["topo_features"],
        global_features=batch["global_features"],
        question_text=batch["question_text"],
        max_new_tokens=max_new_tokens,
    )
    return {
        "question": batch["question_text"][0],
        "answer": batch["answer_text"][0],
        "generated": generated[0] if generated else "",
    }


def trainable_state_dict(model):
    """Return only trainable tensors to keep checkpoints small.

    For LoRA/frozen-base training this includes LoRA weights plus trainable
    projectors/heads, while excluding frozen multi-GB LLM base weights.
    """
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable_names
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_full = bool(getattr(args, "save_full_checkpoint", False))
    payload = {
        "model_state_dict": model.state_dict() if save_full else trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if save_full else None,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "checkpoint_format": "full" if save_full else "trainable_only",
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Train generative Structural VQA model.")
    parser.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--output-dir", default="checkpoints/structural_vqa_generative")
    parser.add_argument("--llm-name-or-path", default="distilgpt2")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--save-full-checkpoint",
        type=str2bool,
        default=False,
        help="Save full model+optimizer checkpoint. Default false saves only trainable weights for LoRA/frozen-base runs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--vision-pretrained", type=str2bool, default=False)
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--freeze-llm", type=str2bool, default=True)
    parser.add_argument("--max-question-length", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=128)
    parser.add_argument("--use-lora", type=str2bool, default=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")

    parser.add_argument("--ot-loss-weight", type=float, default=0.05)
    parser.add_argument("--use-ot", type=str2bool, default=True)
    parser.add_argument("--use-ot-fusion", type=str2bool, default=True)
    parser.add_argument("--ot-fusion-mode", default="prefix", choices=["none", "prefix"])
    parser.add_argument("--ot-fusion-dropout", type=float, default=0.10)
    parser.add_argument("--use-prior-as-ot-target", type=str2bool, default=True)
    parser.add_argument("--prior-ot-global-mass", type=float, default=0.05)
    parser.add_argument("--use-topological-loss", type=str2bool, default=True)
    parser.add_argument("--use-prior-align-loss", type=str2bool, default=True)
    parser.add_argument("--use-global-topo-loss", type=str2bool, default=True)
    parser.add_argument("--use-patch-topo-loss", type=str2bool, default=False)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--global-topo-loss-weight", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--patch-topo-loss-weight", type=float, default=0.005)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    print("\n======================================")
    print("🧠 TRAIN GENERATIVE STRUCTURAL VQA")
    print("======================================")
    print(f"Device: {device}")
    print(f"LLM: {args.llm_name_or_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Freeze LLM: {args.freeze_llm}")
    print("======================================\n")

    dataset = MedicoVQADataset(
        jsonl_path=args.jsonl,
        images_dir=args.images_dir,
        structural_manifest_csv=args.structural_manifest,
        strict_structural=False,
        max_samples=args.max_samples,
    )
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=medico_vqa_collate_fn,
        generator=loader_generator,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=medico_vqa_collate_fn,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

    model = build_structural_generative_vqa(
        llm_name_or_path=args.llm_name_or_path,
        vision_pretrained=args.vision_pretrained,
        vision_backend=args.vision_backend,
        freeze_vision_backbone=args.freeze_vision_backbone,
        freeze_llm=args.freeze_llm,
        max_question_length=args.max_question_length,
        max_answer_length=args.max_answer_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        ot_loss_weight=args.ot_loss_weight,
        use_ot=args.use_ot,
        use_ot_fusion=args.use_ot_fusion,
        ot_fusion_mode=args.ot_fusion_mode,
        ot_fusion_dropout=args.ot_fusion_dropout,
        use_prior_as_ot_target=args.use_prior_as_ot_target,
        prior_ot_global_mass=args.prior_ot_global_mass,
        use_topological_loss=args.use_topological_loss,
        use_prior_align_loss=args.use_prior_align_loss,
        use_global_topo_loss=args.use_global_topo_loss,
        use_patch_topo_loss=args.use_patch_topo_loss,
        prior_loss_weight=args.prior_loss_weight,
        global_topo_loss_weight=args.global_topo_loss_weight,
        patch_topo_loss_weight=args.patch_topo_loss_weight,
    ).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_update_steps = max(updates_per_epoch * args.epochs, 1)
    scheduler = build_lr_scheduler(optimizer, args.lr_scheduler, args.warmup_steps, total_update_steps)
    print(f"LR scheduler: {args.lr_scheduler} | warmup_steps={args.warmup_steps} | total_update_steps={total_update_steps}")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            train=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, optimizer, device, epoch, train=False) if val_loader is not None else {}
        preview = preview_generation(model, train_loader, device)
        elapsed = time.time() - start
        row = {"epoch": epoch, "elapsed_sec": elapsed, "train": train_metrics, "val": val_metrics, "preview": preview, "args": vars(args)}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("\n--------------------------------------")
        print(f"Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s")
        print(f"Train: {train_metrics}")
        if val_metrics:
            print(f"Val:   {val_metrics}")
        print("Preview:")
        print(f"Q: {preview['question']}")
        print(f"GT: {preview['answer']}")
        print(f"GEN: {preview['generated']}")
        print("--------------------------------------\n")

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, row, args)

    model.tokenizer.save_pretrained(output_dir / "tokenizer")
    if args.use_lora and hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(output_dir / "lora_adapter")
    print(f"✅ Generative training complete. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
