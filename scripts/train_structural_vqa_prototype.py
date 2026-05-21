import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data_pipeline.dataset import MedicoVQADataset, medico_vqa_collate_fn
from src.models.structural_vqa import build_structural_vqa_prototype


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Không parse được bool: {value}")


def collect_texts(dataset):
    questions = []
    answers = []
    for i in tqdm(range(len(dataset)), desc="Build vocab texts"):
        sample = dataset[i]
        questions.append(sample["question_text"])
        answers.append(sample["answer_text"])
    return questions, answers


def save_answer_vocab(answer_vocab, path):
    path = Path(path)
    payload = {
        "answer_to_id": answer_vocab.answer_to_id,
        "id_to_answer": {str(k): v for k, v in answer_vocab.id_to_answer.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_to_device(batch, device):
    tensor_keys = [
        "image",
        "prior_mask",
        "topo_features",
        "global_features",
    ]
    for key in tensor_keys:
        batch[key] = batch[key].to(device)
    return batch


def compute_accuracy(logits, answer_ids):
    pred = logits.argmax(dim=-1)
    return (pred == answer_ids).float().mean().item()


def train_one_epoch(model, dataloader, optimizer, answer_vocab, device, epoch):
    model.train()
    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "ot_cost": 0.0,
        "prior_align_loss": 0.0,
        "global_topo_loss": 0.0,
        "patch_topo_loss": 0.0,
        "topological_loss": 0.0,
        "accuracy": 0.0,
    }
    steps = 0

    progress = tqdm(dataloader, desc=f"Train epoch {epoch}")
    for batch in progress:
        batch = batch_to_device(batch, device)
        answer_ids = answer_vocab.batch_encode(batch["answer_text"]).to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            image=batch["image"],
            prior_mask=batch["prior_mask"],
            topo_features=batch["topo_features"],
            global_features=batch["global_features"],
            question_text=batch["question_text"],
            answer_ids=answer_ids,
            return_diagnostics=True,
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = compute_accuracy(out["logits"].detach(), answer_ids)
        step_values = {
            "loss": float(out["loss"].detach().cpu()),
            "ce_loss": float(out["ce_loss"].detach().cpu()),
            "ot_cost": float(out["ot_cost"].detach().cpu()),
            "prior_align_loss": float(out["prior_align_loss"].detach().cpu()),
            "global_topo_loss": float(out["global_topo_loss"].detach().cpu()),
            "patch_topo_loss": float(out["patch_topo_loss"].detach().cpu()),
            "topological_loss": float(out["topological_loss"].detach().cpu()),
            "accuracy": acc,
        }
        for key, value in step_values.items():
            totals[key] += value
        steps += 1
        progress.set_postfix(loss=step_values["loss"], acc=acc)

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, dataloader, answer_vocab, device, epoch):
    model.eval()
    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "ot_cost": 0.0,
        "prior_align_loss": 0.0,
        "global_topo_loss": 0.0,
        "patch_topo_loss": 0.0,
        "topological_loss": 0.0,
        "accuracy": 0.0,
    }
    steps = 0

    for batch in tqdm(dataloader, desc=f"Eval epoch {epoch}"):
        batch = batch_to_device(batch, device)
        answer_ids = answer_vocab.batch_encode(batch["answer_text"]).to(device)
        out = model(
            image=batch["image"],
            prior_mask=batch["prior_mask"],
            topo_features=batch["topo_features"],
            global_features=batch["global_features"],
            question_text=batch["question_text"],
            answer_ids=answer_ids,
            return_diagnostics=True,
        )
        acc = compute_accuracy(out["logits"], answer_ids)
        step_values = {
            "loss": float(out["loss"].cpu()),
            "ce_loss": float(out["ce_loss"].cpu()),
            "ot_cost": float(out["ot_cost"].cpu()),
            "prior_align_loss": float(out["prior_align_loss"].cpu()),
            "global_topo_loss": float(out["global_topo_loss"].cpu()),
            "patch_topo_loss": float(out["patch_topo_loss"].cpu()),
            "topological_loss": float(out["topological_loss"].cpu()),
            "accuracy": acc,
        }
        for key, value in step_values.items():
            totals[key] += value
        steps += 1

    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    path = Path(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Structural VQA prototype classification model.")
    parser.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    parser.add_argument("--output-dir", default="checkpoints/structural_vqa_prototype")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--vision-backend", default="timm", choices=["auto", "timm", "torchvision"])
    parser.add_argument("--vision-pretrained", type=str2bool, default=False)
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--max-text-vocab-size", type=int, default=30000)
    parser.add_argument("--max-answer-vocab-size", type=int, default=None)

    parser.add_argument("--ot-loss-weight", type=float, default=0.01)
    parser.add_argument("--use-topological-loss", type=str2bool, default=True)
    parser.add_argument("--use-prior-align-loss", type=str2bool, default=True)
    parser.add_argument("--use-global-topo-loss", type=str2bool, default=True)
    parser.add_argument("--use-patch-topo-loss", type=str2bool, default=False)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--global-topo-loss-weight", type=float, default=0.01)
    parser.add_argument("--patch-topo-loss-weight", type=float, default=0.005)
    parser.add_argument("--use-prior-as-ot-target", type=str2bool, default=True)
    parser.add_argument("--prior-ot-global-mass", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    print("\n======================================")
    print("🧠 TRAIN STRUCTURAL VQA PROTOTYPE")
    print("======================================")
    print(f"Device: {device}")
    print(f"JSONL: {args.jsonl}")
    print(f"Max samples: {args.max_samples}")
    print(f"Topo loss: {args.use_topological_loss}")
    print("======================================\n")

    dataset = MedicoVQADataset(
        jsonl_path=args.jsonl,
        images_dir=args.images_dir,
        structural_manifest_csv=args.structural_manifest,
        strict_structural=False,
        max_samples=args.max_samples,
    )
    questions, answers = collect_texts(dataset)

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=medico_vqa_collate_fn,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=medico_vqa_collate_fn,
        )

    model, tokenizer, answer_vocab = build_structural_vqa_prototype(
        questions=questions,
        answers=answers,
        d_model=args.d_model,
        text_max_length=args.text_max_length,
        vision_pretrained=args.vision_pretrained,
        vision_backend=args.vision_backend,
        freeze_vision_backbone=args.freeze_vision_backbone,
        max_text_vocab_size=args.max_text_vocab_size,
        max_answer_vocab_size=args.max_answer_vocab_size,
        ot_loss_weight=args.ot_loss_weight,
        use_topological_loss=args.use_topological_loss,
        use_prior_align_loss=args.use_prior_align_loss,
        use_global_topo_loss=args.use_global_topo_loss,
        use_patch_topo_loss=args.use_patch_topo_loss,
        prior_loss_weight=args.prior_loss_weight,
        global_topo_loss_weight=args.global_topo_loss_weight,
        patch_topo_loss_weight=args.patch_topo_loss_weight,
        use_prior_as_ot_target=args.use_prior_as_ot_target,
        prior_ot_global_mass=args.prior_ot_global_mass,
    )
    model.to(device)

    save_answer_vocab(answer_vocab, output_dir / "answer_vocab.json")
    (output_dir / "tokenizer_vocab.json").write_text(
        json.dumps(tokenizer.vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, answer_vocab, device, epoch)
        val_metrics = evaluate(model, val_loader, answer_vocab, device, epoch) if val_loader is not None else {}
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "train": train_metrics,
            "val": val_metrics,
            "args": vars(args),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("\n--------------------------------------")
        print(f"Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s")
        print(f"Train: {train_metrics}")
        if val_metrics:
            print(f"Val:   {val_metrics}")
        print("--------------------------------------\n")

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, row, args)
        current_val_acc = val_metrics.get("accuracy", train_metrics["accuracy"])
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, row, args)

    print(f"✅ Training complete. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
