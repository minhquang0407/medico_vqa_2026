import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_pipeline.dataset import MedicoVQADataset, medico_vqa_collate_fn
from src.models.structural_vqa_generative import build_structural_generative_vqa


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Không parse được bool: {value}")


def batch_to_device(batch, device):
    for key in ["image", "prior_mask", "topo_features", "global_features"]:
        batch[key] = batch[key].to(device)
    return batch


def load_model_from_checkpoint(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    config = {**train_args}
    config.update(
        {
            "llm_name_or_path": args.llm_name_or_path or train_args.get("llm_name_or_path", "distilgpt2"),
            "vision_pretrained": args.vision_pretrained,
            "vision_backend": args.vision_backend or train_args.get("vision_backend", "timm"),
            "freeze_vision_backbone": args.freeze_vision_backbone,
            "freeze_llm": train_args.get("freeze_llm", True),
        }
    )
    if args.visual_structural_mode is not None:
        config["visual_structural_mode"] = args.visual_structural_mode
    if args.use_global_structural_token is not None:
        config["use_global_structural_token"] = args.use_global_structural_token
    allowed = {
        "llm_name_or_path",
        "vision_pretrained",
        "vision_backend",
        "freeze_vision_backbone",
        "freeze_llm",
        "max_question_length",
        "max_answer_length",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "ot_loss_weight",
        "use_ot",
        "use_ot_fusion",
        "ot_fusion_mode",
        "ot_fusion_dropout",
        "use_prior_as_ot_target",
        "prior_ot_global_mass",
        "use_topological_loss",
        "use_prior_align_loss",
        "use_global_topo_loss",
        "use_patch_topo_loss",
        "prior_loss_weight",
        "global_topo_loss_weight",
        "patch_topo_loss_weight",
        "visual_structural_mode",
        "use_global_structural_token",
    }
    build_kwargs = {key: value for key, value in config.items() if key in allowed}
    model = build_structural_generative_vqa(**build_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative Structural VQA model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    parser.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    parser.add_argument("--structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")
    parser.add_argument("--output-dir", default="eval/structural_vqa_generative")
    parser.add_argument("--llm-name-or-path", default=None)
    parser.add_argument("--vision-backend", default=None)
    parser.add_argument("--vision-pretrained", type=str2bool, default=False)
    parser.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    parser.add_argument("--visual-structural-mode", default=None, choices=["none", "tda_only", "all"])
    parser.add_argument("--use-global-structural-token", type=str2bool, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structural_manifest = Path(args.structural_manifest) if args.structural_manifest else None
    if structural_manifest is not None and not structural_manifest.exists():
        print(
            f"⚠️ Structural manifest not found: {structural_manifest}. "
            "Falling back to default structural tensors for smoke evaluation. "
            "For official evaluation, precompute structural features for the test set."
        )
        structural_manifest = None

    dataset = MedicoVQADataset(
        jsonl_path=args.jsonl,
        images_dir=args.images_dir,
        structural_manifest_csv=structural_manifest,
        strict_structural=False,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=medico_vqa_collate_fn)
    model = load_model_from_checkpoint(args, device)

    prediction_path = output_dir / "predictions.jsonl"
    num_predictions = 0

    with prediction_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch_to_device(batch, device)
            predictions = model.generate(
                image=batch["image"],
                prior_mask=batch["prior_mask"],
                topo_features=batch["topo_features"],
                global_features=batch["global_features"],
                question_text=batch["question_text"],
                max_new_tokens=args.max_new_tokens,
            )
            for idx, pred in enumerate(predictions):
                question = batch["question_text"][idx]
                gold = batch["answer_text"][idx]
                row = {
                    "record_index": int(batch["record_index"][idx].cpu()),
                    "image_ref": batch["image_ref"][idx],
                    "image_path": batch["image_path"][idx],
                    "question": question,
                    "ground_truth": gold,
                    "prediction": pred,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_predictions += 1

    print(f"✅ Generation complete. Wrote {num_predictions} rows to: {prediction_path}")
    print("Next: run scripts/evaluate_paper_metrics.py on this predictions.jsonl file.")


if __name__ == "__main__":
    main()
