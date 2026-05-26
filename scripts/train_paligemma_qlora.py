"""Train/evaluate a PaliGemma QLoRA + structural-hint baseline for Medico VQA.

This is a challenger Task 1 experiment, not a replacement for the current
submission unless it scores better. It can use the structural feature ZIP that
was uploaded to Drive, avoiding online TDA/prior computation.

Colab example:

  !unzip -q /content/drive/MyDrive/structural_features.zip -d /content/structural_features
  !python scripts/train_paligemma_qlora.py \
    --mode train_eval \
    --output-dir outputs/paligemma_struct_hints \
    --epochs 2 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --max-train-samples 30000 \
    --eval-samples 1500 \
    --use-structural-hints \
    --structural-root /content/structural_features \
    --use-augmentation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Image as HfImage
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


_SPACE_RE = re.compile(r"\s+")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)+", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    text = _SPACE_RE.sub(" ", str(text or "")).strip()
    text = _ANSWER_PREFIX_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _basename_no_ext(value: str) -> str:
    return Path(str(value or "")).stem


def _image_key(sample: Dict[str, Any]) -> str:
    if "img_id" in sample and sample["img_id"]:
        return _basename_no_ext(str(sample["img_id"]))
    image = sample.get("image")
    if isinstance(image, dict):
        for key in ("path", "filename"):
            if image.get(key):
                return _basename_no_ext(str(image[key]))
    path = getattr(image, "filename", None)
    if path:
        return _basename_no_ext(str(path))
    return ""


class StructuralHintStore:
    """Load precomputed structural NPZ files and convert them to short text hints."""

    def __init__(self, root: str | Path, train_manifest: str = "auto", eval_manifest: str = "auto"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"structural root not found: {self.root}")
        self.index: Dict[str, Path] = {}
        manifests: List[Path] = []
        for manifest in [train_manifest, eval_manifest]:
            if manifest == "auto":
                continue
            path = Path(manifest)
            if not path.is_absolute():
                path = self.root / path
            manifests.append(path)
        if train_manifest == "auto" and eval_manifest == "auto":
            manifests = sorted(self.root.rglob("*manifest*.csv"))
        if manifests:
            for manifest in manifests:
                self._load_manifest(manifest)
        else:
            for npz_path in self.root.rglob("*.npz"):
                self.index[npz_path.stem.split("_")[0]] = npz_path
        print(f"Loaded structural hint index: {len(self.index)} images from {self.root}")

    def _load_manifest(self, manifest: Path) -> None:
        if not manifest.exists():
            print(f"Warning: structural manifest not found: {manifest}")
            return
        with manifest.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") not in {"processed", "skipped_existing", ""}:
                    continue
                image_ref = row.get("image_ref") or row.get("image_path") or ""
                cache = row.get("cache_path") or ""
                if not cache:
                    continue
                cache_path = Path(cache)
                if not cache_path.is_absolute():
                    cache_path = self.root / cache_path
                    if not cache_path.exists():
                        cache_path = self.root / Path(cache).name
                key = _basename_no_ext(image_ref)
                if key and cache_path.exists():
                    self.index[key] = cache_path

    def get_hint(self, sample: Dict[str, Any]) -> str:
        key = _image_key(sample)
        path = self.index.get(key)
        if path is None or not path.exists():
            return ""
        try:
            with np.load(path, allow_pickle=True) as data:
                return structural_hint_from_npz(data)
        except Exception:
            return ""


def _bucket(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _quadrant_from_map(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2 or float(arr.max(initial=0.0)) <= 1e-8:
        return "central"
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    h, w = arr.shape
    vertical = "upper" if y < h / 3 else "lower" if y > 2 * h / 3 else "central"
    horizontal = "left" if x < w / 3 else "right" if x > 2 * w / 3 else "central"
    if vertical == "central" and horizontal == "central":
        return "central"
    if vertical == "central":
        return horizontal
    if horizontal == "central":
        return vertical
    return f"{vertical} {horizontal}"


def structural_hint_from_npz(data) -> str:
    prior = np.asarray(data.get("prior_mask", np.zeros((14, 14))), dtype=np.float32)
    red = np.asarray(data.get("red_map", prior), dtype=np.float32)
    topo = np.asarray(data.get("topo_mask", prior), dtype=np.float32)
    topo_features = np.asarray(data.get("topo_features", np.zeros((14, 14, 12))), dtype=np.float32)
    global_features = np.asarray(data.get("global_features", np.zeros(8)), dtype=np.float32).reshape(-1)

    prior_strength = float(np.mean(prior))
    red_strength = float(np.mean(red))
    topo_strength = float(np.mean(topo))
    concentration = 0.0
    flat = prior.reshape(-1)
    if float(flat.sum()) > 1e-8:
        k = max(1, int(0.10 * flat.size))
        concentration = float(np.sort(flat)[::-1][:k].sum() / flat.sum())

    specular = 0.0
    tissue = 0.0
    if topo_features.ndim == 3 and topo_features.shape[-1] > 8:
        specular = float(np.mean(np.clip(topo_features[..., 7], 0.0, 1.0)))
        tissue = float(np.mean(np.clip(topo_features[..., 8], 0.0, 1.0)))
    elif global_features.size >= 2:
        tissue = float(np.clip(global_features[0], 0.0, 1.0))
        specular = float(np.clip(global_features[-1], 0.0, 1.0))

    location = _quadrant_from_map(prior)
    focality = "focal" if concentration >= 0.35 else "diffuse"
    hints = [
        f"structural evidence {location}",
        f"{focality} prior",
        f"redness {_bucket(red_strength, 0.08, 0.22)}",
        f"morphology {_bucket(topo_strength, 0.08, 0.22)}",
        f"tissue {_bucket(tissue, 0.35, 0.70)}",
        f"specular artifact {_bucket(specular, 0.10, 0.30)}",
    ]
    return "; ".join(hints)


def build_prompt(question: str, structural_hint: str = "") -> str:
    question = str(question).strip()
    if structural_hint:
        return f"answer en {question} structural hints: {structural_hint}"
    return f"answer en {question}"


def load_kvasir_splits(train_samples: Optional[int] = None, eval_samples: Optional[int] = 1500, seed: int = 42):
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")
    train = ds["train"].cast_column("image", HfImage())
    test = ds["test"].cast_column("image", HfImage())
    if train_samples is not None:
        train = train.shuffle(seed=seed).select(range(min(train_samples, len(train))))
    if eval_samples is not None:
        test = test.shuffle(seed=seed).select(range(min(eval_samples, len(test))))
    return train, test


def build_albumentations_transform(enabled: bool):
    if not enabled:
        return None
    import albumentations as A

    return A.Compose([
        A.Affine(scale=(0.90, 1.10), translate_percent=(-0.10, 0.10), rotate=(-10, 10), p=0.75),
        A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.03, p=0.60),
    ])


def apply_augmentation(image: Image.Image, transform) -> Image.Image:
    if transform is None:
        return image.convert("RGB")
    arr = np.asarray(image.convert("RGB"))
    out = transform(image=arr)["image"]
    return Image.fromarray(out).convert("RGB")


@dataclass
class PaligemmaCollator:
    processor: Any
    max_length: int = 224
    train: bool = True
    transform: Any = None
    structural_store: Optional[StructuralHintStore] = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [apply_augmentation(ex["image"], self.transform if self.train else None) for ex in examples]
        hints = [self.structural_store.get_hint(ex) if self.structural_store is not None else "" for ex in examples]
        prompts = [build_prompt(ex["question"], hint) for ex, hint in zip(examples, hints)]
        answers = [normalize_answer(ex.get("answer", "")) for ex in examples]
        batch = self.processor(
            text=prompts,
            images=images,
            suffix=answers if self.train else None,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.train and "labels" not in batch:
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def load_model_and_processor(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "right"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    return model, processor


def make_structural_store(args) -> Optional[StructuralHintStore]:
    if not args.use_structural_hints:
        return None
    return StructuralHintStore(args.structural_root, args.train_structural_manifest, args.eval_structural_manifest)


def train(args) -> Path:
    from transformers import get_cosine_schedule_with_warmup

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, eval_ds = load_kvasir_splits(args.max_train_samples, args.eval_samples, args.seed)
    model, processor = load_model_and_processor(args)
    transform = build_albumentations_transform(args.use_augmentation)
    structural_store = make_structural_store(args)
    collator = PaligemmaCollator(processor, args.max_length, True, transform, structural_store)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator, pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Training epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(loss=running_loss / max(1, step + 1), step=global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
        epoch_dir = output_dir / f"epoch-{epoch + 1}"
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "training_metadata.json").write_text(json.dumps({
        "model_name": args.model_name,
        "epochs": args.epochs,
        "global_step": global_step,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "elapsed_sec": round(time.time() - start, 2),
        "args": vars(args),
    }, indent=2), encoding="utf-8")
    return output_dir


@torch.inference_mode()
def generate_answer(model, processor, image: Image.Image, question: str, max_new_tokens: int, structural_hint: str = "") -> str:
    prompt = build_prompt(question, structural_hint)
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=processor.tokenizer.pad_token_id)
    input_len = inputs["input_ids"].shape[-1]
    text = processor.tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return normalize_answer(text)


def evaluate(args, checkpoint_dir: Optional[str] = None) -> Path:
    from evaluate import load as load_metric
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir or args.output_dir
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True) if args.load_in_4bit else None
    processor = AutoProcessor.from_pretrained(checkpoint)
    base = PaliGemmaForConditionalGeneration.from_pretrained(args.model_name, quantization_config=quant_config, torch_dtype=torch_dtype, device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    structural_store = make_structural_store(args)

    _, eval_ds = load_kvasir_splits(None, args.eval_samples, args.seed)
    predictions = []
    refs = []
    for idx, sample in enumerate(tqdm(eval_ds, desc="Evaluating")):
        hint = structural_store.get_hint(sample) if structural_store is not None else ""
        pred = generate_answer(model, processor, sample["image"], sample["question"], args.max_new_tokens, hint)
        ref = normalize_answer(sample["answer"])
        predictions.append({
            "index": idx,
            "img_id": str(sample.get("img_id", "")),
            "question": str(sample["question"]),
            "structural_hint": hint,
            "answer": pred,
            "reference": ref,
        })
        refs.append([ref])

    pred_text = [p["answer"] for p in predictions]
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")
    bleu_result = bleu.compute(predictions=pred_text, references=refs)
    rouge_result = rouge.compute(predictions=pred_text, references=[r[0] for r in refs])
    meteor_result = meteor.compute(predictions=pred_text, references=[r[0] for r in refs])
    scores = {
        "bleu": round(float(bleu_result["bleu"]), 4),
        "rouge1": round(float(rouge_result["rouge1"]), 4),
        "rouge2": round(float(rouge_result["rouge2"]), 4),
        "rougeL": round(float(rouge_result["rougeL"]), 4),
        "meteor": round(float(meteor_result["meteor"]), 4),
    }
    print("Scores:", scores)
    suffix = "struct_hints" if args.use_structural_hints else "plain"
    out_path = output_dir / f"paligemma_eval_predictions_{suffix}.json"
    out_path.write_text(json.dumps({"scores": scores, "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma QLoRA with optional offline structural hints")
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--model-name", default="google/paligemma-3b-pt-224")
    parser.add_argument("--output-dir", default="outputs/paligemma_qlora_medico")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=-1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--use-structural-hints", action="store_true")
    parser.add_argument("--structural-root", default="data/processed/structural_features")
    parser.add_argument("--train-structural-manifest", default="auto")
    parser.add_argument("--eval-structural-manifest", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = None
    if args.mode in {"train", "train_eval"}:
        checkpoint_dir = str(train(args))
    if args.mode in {"eval", "train_eval"}:
        evaluate(args, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
