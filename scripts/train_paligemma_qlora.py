"""Train/evaluate a PaliGemma QLoRA baseline for Medico/Kvasir VQA.

Designed for Colab GPU experiments. This is a challenger baseline inspired by
Medico 2025 winning reports:

  google/paligemma-3b-pt-224 + QLoRA 4-bit NF4 + LoRA r16/alpha32
  + light geometric/color augmentation.

Example Colab usage:

  !pip install -U transformers datasets accelerate peft bitsandbytes evaluate \
      rouge_score sacrebleu nltk albumentations opencv-python-headless pillow tqdm

  !python scripts/train_paligemma_qlora.py \
      --mode train_eval \
      --output-dir outputs/paligemma_qlora_medico \
      --epochs 2 \
      --batch-size 2 \
      --gradient-accumulation-steps 8 \
      --learning-rate 2e-5 \
      --max-train-samples 30000 \
      --eval-samples 1500 \
      --use-augmentation

Notes:
- This script does not modify the current Task 1 submission.
- It is meant to quickly test whether a native VLM baseline can beat the current
  structural Qwen model.
- For serious comparison, evaluate on the same public 1500-sample subset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def build_prompt(question: str) -> str:
    # PaliGemma was trained on short task prefixes. Keep prompt compact.
    return f"answer en {str(question).strip()}"


def load_kvasir_splits(
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = 1500,
    seed: int = 42,
):
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
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError("Install albumentations to use --use-augmentation") from exc

    return A.Compose(
        [
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.10, 0.10),
                rotate=(-10, 10),
                p=0.75,
            ),
            A.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.10,
                hue=0.03,
                p=0.60,
            ),
        ]
    )


def apply_augmentation(image: Image.Image, transform) -> Image.Image:
    if transform is None:
        return image.convert("RGB")
    arr = np.asarray(image.convert("RGB"))
    out = transform(image=arr)["image"]
    return Image.fromarray(out).convert("RGB")


@dataclass
class PaligemmaCollator:
    processor: Any
    max_length: int = 192
    train: bool = True
    transform: Any = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [apply_augmentation(ex["image"], self.transform if self.train else None) for ex in examples]
        prompts = [build_prompt(ex["question"]) for ex in examples]
        answers = [normalize_answer(ex.get("answer", "")) for ex in examples]

        # PaliGemmaProcessor supports `suffix`; this automatically creates labels
        # with prompt/image tokens masked when available in recent transformers.
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
            # Fallback for older transformers: train on all text tokens, masking pad.
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def load_model_and_processor(args):
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.padding_side != "right":
        processor.tokenizer.padding_side = "right"

    quant_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def train(args) -> Path:
    from transformers import get_cosine_schedule_with_warmup

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = load_kvasir_splits(args.max_train_samples, args.eval_samples, args.seed)
    model, processor = load_model_and_processor(args)
    transform = build_albumentations_transform(args.use_augmentation)

    collator = PaligemmaCollator(
        processor=processor,
        max_length=args.max_length,
        train=True,
        transform=transform,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

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
            batch = {k: v.to(model.device) if hasattr(model, "device") else v.cuda() for k, v in batch.items()}
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
    metadata = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "global_step": global_step,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "elapsed_sec": round(time.time() - start, 2),
        "args": vars(args),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


@torch.inference_mode()
def generate_answer(model, processor, image: Image.Image, question: str, max_new_tokens: int) -> str:
    prompt = build_prompt(question)
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    # Decode only newly generated tokens when possible.
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, input_len:]
    text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
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
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(checkpoint)
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()

    _, eval_ds = load_kvasir_splits(None, args.eval_samples, args.seed)
    predictions = []
    refs = []
    for idx, sample in enumerate(tqdm(eval_ds, desc="Evaluating")):
        pred = generate_answer(model, processor, sample["image"], sample["question"], args.max_new_tokens)
        ref = normalize_answer(sample["answer"])
        predictions.append(
            {
                "index": idx,
                "img_id": str(sample.get("img_id", "")),
                "question": str(sample["question"]),
                "answer": pred,
                "reference": ref,
            }
        )
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

    out_path = output_dir / "paligemma_eval_predictions.json"
    out_path.write_text(
        json.dumps({"scores": scores, "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma QLoRA baseline for Medico/Kvasir VQA")
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
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--use-augmentation", action="store_true")

    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
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
