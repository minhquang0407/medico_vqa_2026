"""Qwen3B Curriculum-Gated Topological Adapter training.

RunB++ deep-fusion adapter with PEFT curriculum:
- epoch warmup can disable OT/topological alignment losses
- explicit sigmoid gate regularization encourages decisive topology gates
- checkpoints/eval use project layout under checkpoints/ and eval/
"""
from __future__ import annotations

import argparse, json, math, os, random, types, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.dataset import (
    MedicoVQADataset,
    get_image_ref,
    medico_vqa_collate_fn,
    parse_question_answer,
    path_basename_any,
)
from src.models.structural_vqa_generative import build_structural_generative_vqa
from scripts.train_structural_vqa_generative import (
    build_lr_scheduler,
    current_lr,
    load_trainable_or_full_state,
    preview_generation,
    save_checkpoint,
    seed_worker,
    set_reproducible_seed,
    str2bool,
)


class TopoAdapter(nn.Module):
    def __init__(self, hidden_dim: int, topo_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.topo_projector = nn.Sequential(
            nn.Linear(topo_dim, bottleneck_dim), nn.SiLU(), nn.Linear(bottleneck_dim, bottleneck_dim)
        )
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.gating = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.up_proj.weight)  # exact no-op at init
        self.last_gate_loss: Optional[torch.Tensor] = None
        self.last_gate_mean: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, topo_condition: torch.Tensor) -> torch.Tensor:
        if topo_condition.ndim == 2:
            topo_condition = topo_condition.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        topo_condition = topo_condition.to(device=hidden_states.device, dtype=self.topo_projector[0].weight.dtype)
        hs = hidden_states.to(dtype=self.down_proj.weight.dtype)
        t_emb = self.topo_projector(topo_condition)
        gate = torch.sigmoid(self.gating(t_emb))
        self.last_gate_loss = (gate * (1.0 - gate)).mean()
        self.last_gate_mean = gate.detach().mean()
        delta = self.up_proj(self.down_proj(hs) * gate).to(dtype=hidden_states.dtype)
        return hidden_states + delta


class DecoderLayerWithTopoAdapter(nn.Module):
    def __init__(self, base_layer: nn.Module, adapter: TopoAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.attention_type = getattr(base_layer, "attention_type", "full_attention")
        self.topo_adapter = adapter
        self.topo_condition: Optional[torch.Tensor] = None

    def set_topo_condition(self, condition: Optional[torch.Tensor]):
        self.topo_condition = condition

    def forward(self, *args, **kwargs):
        out = self.base_layer(*args, **kwargs)
        if self.topo_condition is None:
            return out
        if isinstance(out, tuple):
            hs = self.topo_adapter(out[0], self.topo_condition)
            return (hs,) + out[1:]
        return self.topo_adapter(out, self.topo_condition)


def find_decoder_layers(llm) -> nn.ModuleList:
    candidates = [
        "model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
        "model.model.layers",
    ]
    for path in candidates:
        obj = llm
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part): ok = False; break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("Cannot locate Qwen decoder layers for TopoAdapter injection")


def install_topo_adapters(model, topo_dim: int, bottleneck_dim: int, last_n_layers: int, every_n_layers: int) -> List[DecoderLayerWithTopoAdapter]:
    layers = find_decoder_layers(model.llm)
    hidden_dim = int(model.llm.get_input_embeddings().embedding_dim)
    n = len(layers)
    selected = set(range(max(0, n - last_n_layers), n)) if last_n_layers > 0 else set()
    if every_n_layers > 0:
        selected.update(range(0, n, every_n_layers))
    wrappers: List[DecoderLayerWithTopoAdapter] = []
    for i in sorted(selected):
        if isinstance(layers[i], DecoderLayerWithTopoAdapter):
            wrappers.append(layers[i]); continue
        wrapper = DecoderLayerWithTopoAdapter(layers[i], TopoAdapter(hidden_dim, topo_dim, bottleneck_dim))
        layers[i] = wrapper
        wrappers.append(wrapper)
    print(f"Installed {len(wrappers)} TopoAdapters / {n} decoder layers | hidden={hidden_dim} topo_dim={topo_dim}")
    return wrappers


def topo_condition(prior_mask, topo_features, global_features, mode: str) -> torch.Tensor:
    # topo_features: [B,14,14,C]
    tf = topo_features.float()
    flat = tf.flatten(1, 2)
    parts = [flat.mean(1), flat.std(1), flat.amax(1)]  # 3*C = 36 when C=12
    if mode == "all":
        pm = prior_mask.float().flatten(1)
        parts.extend([pm.mean(1, keepdim=True), pm.std(1, keepdim=True), pm.amax(1, keepdim=True)])
        parts.append(global_features.float())
    return torch.cat(parts, dim=-1)


def set_condition(wrappers: List[DecoderLayerWithTopoAdapter], cond: Optional[torch.Tensor]):
    for w in wrappers:
        w.set_topo_condition(cond)


def collect_gate_stats(wrappers: List[DecoderLayerWithTopoAdapter], device: torch.device):
    losses = [w.topo_adapter.last_gate_loss for w in wrappers if w.topo_adapter.last_gate_loss is not None]
    means = [w.topo_adapter.last_gate_mean for w in wrappers if w.topo_adapter.last_gate_mean is not None]
    gate_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    gate_mean = torch.stack([m.to(device) for m in means]).mean() if means else torch.tensor(0.0, device=device)
    return gate_loss, gate_mean


def patch_model_forward(model, wrappers, mode: str):
    original_forward = model.forward
    original_generate = model.generate

    def forward_with_topo(self, image, prior_mask, topo_features, global_features, question_text, answer_text=None, return_diagnostics=True):
        cond = topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        set_condition(wrappers, cond)
        try:
            return original_forward(image, prior_mask, topo_features, global_features, question_text, answer_text, return_diagnostics)
        finally:
            set_condition(wrappers, None)

    @torch.no_grad()
    def generate_with_topo(self, image, prior_mask, topo_features, global_features, question_text, max_new_tokens=64):
        cond = topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        set_condition(wrappers, cond)
        try:
            return original_generate(image, prior_mask, topo_features, global_features, question_text, max_new_tokens)
        finally:
            set_condition(wrappers, None)

    model.forward = types.MethodType(forward_with_topo, model)
    model.generate = types.MethodType(generate_with_topo, model)


def resolve_feature_zeroing(args) -> tuple[bool, bool]:
    """Resolve effective prior/global zeroing while preserving historical topo_mode behavior."""
    zero_prior_mask = args.zero_prior_mask if args.zero_prior_mask is not None else args.topo_mode == "tda_only"
    zero_global_features = (
        args.zero_global_features if args.zero_global_features is not None else args.topo_mode == "tda_only"
    )
    return bool(zero_prior_mask), bool(zero_global_features)


def batch_to_device(batch, device, zero_prior_mask: bool = False, zero_global_features: bool = False):
    for key in ["image", "prior_mask", "topo_features", "global_features"]:
        batch[key] = batch[key].to(device)
    if zero_prior_mask:
        batch["prior_mask"] = torch.zeros_like(batch["prior_mask"])
    if zero_global_features:
        batch["global_features"] = torch.zeros_like(batch["global_features"])
    return batch


def resolve_base_structural_flags(args) -> Dict[str, bool]:
    """Resolve base-model structural flags, keeping historical auto defaults."""
    zero_prior_mask, zero_global_features = resolve_feature_zeroing(args)
    prior_as_ot_target = (
        args.use_base_prior_as_ot_target
        if args.use_base_prior_as_ot_target is not None
        else args.topo_mode == "all" and not zero_prior_mask
    )
    prior_align_loss = (
        args.use_base_prior_align_loss
        if args.use_base_prior_align_loss is not None
        else args.topo_mode == "all" and not zero_prior_mask
    )
    global_topo_loss = (
        args.use_base_global_topo_loss
        if args.use_base_global_topo_loss is not None
        else args.topo_mode == "all" and not zero_global_features
    )
    return {
        "use_ot": bool(args.use_base_ot),
        "use_ot_fusion": bool(args.use_base_ot_fusion),
        "use_prior_as_ot_target": bool(prior_as_ot_target),
        "use_topological_loss": bool(args.use_base_topological_loss),
        "use_prior_align_loss": bool(prior_align_loss),
        "use_global_topo_loss": bool(global_topo_loss),
        "use_patch_topo_loss": bool(args.use_base_patch_topo_loss),
    }


def apply_curriculum_schedule(model, epoch: int, args, base_cfg: Dict[str, object]):
    for key, value in base_cfg.items():
        setattr(model.config, key, value)
    warmup = epoch <= args.align_warmup_epochs and args.warmup_text_only
    if warmup:
        model.config.ot_loss_weight = 0.0
        model.config.prior_loss_weight = 0.0
        model.config.global_topo_loss_weight = 0.0
        if not args.keep_patch_topo_during_warmup:
            model.config.patch_topo_loss_weight = 0.0
        model.config.use_prior_as_ot_target = False
        model.config.use_prior_align_loss = False
        model.config.use_global_topo_loss = False
        if not args.keep_patch_topo_during_warmup:
            model.config.use_patch_topo_loss = False
    return warmup


def run_epoch(model, loader, optimizer, device, epoch, train=True, grad_accum=1, scheduler=None, zero_prior_mask=False, zero_global_features=False, wrappers=None, gate_loss_weight=0.0):
    model.train(train); totals = {"loss":0.0,"lm_loss":0.0,"ot_cost":0.0,"topological_loss":0.0,"gate_loss":0.0,"gate_mean":0.0}; steps=0
    iterator = tqdm(loader, desc=("Train" if train else "Eval") + f" epoch {epoch}")
    for batch in iterator:
        batch = batch_to_device(batch, device, zero_prior_mask=zero_prior_mask, zero_global_features=zero_global_features)
        with torch.set_grad_enabled(train):
            out = model(image=batch["image"], prior_mask=batch["prior_mask"], topo_features=batch["topo_features"], global_features=batch["global_features"], question_text=batch["question_text"], answer_text=batch["answer_text"], return_diagnostics=True)
            loss = out["loss"]
            gate_loss, gate_mean = collect_gate_stats(wrappers or [], device)
            total_loss = loss + (gate_loss_weight * gate_loss if train and gate_loss_weight > 0 else 0.0)
            if train:
                (total_loss / grad_accum).backward()
                if steps % grad_accum == grad_accum - 1:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None: scheduler.step()
        for k in ["loss","lm_loss","ot_cost","topological_loss"]: totals[k] += float(out.get(k, torch.tensor(0.0)).detach().cpu())
        totals["gate_loss"] += float(gate_loss.detach().cpu())
        totals["gate_mean"] += float(gate_mean.detach().cpu())
        steps += 1
        iterator.set_postfix(loss=totals["loss"]/steps, lr=current_lr(optimizer))
    if train and steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if scheduler is not None: scheduler.step()
    return {k:v/max(steps,1) for k,v in totals.items()}



def _clean_question_for_compare(text: Any) -> str:
    return str(text or "").replace("<image>", "").strip()


def _clean_answer_for_compare(text: Any) -> str:
    return str(text or "").strip()


def _image_stem_for_compare(text: Any) -> str:
    return Path(path_basename_any(str(text or ""))).stem.lower()


def apply_hf_submission_subset(ds: MedicoVQADataset, args, eval_dir: Path) -> None:
    """Reorder/filter local JSONL to match submission_task1.py public subset exactly.

    submission_task1.py uses:
        load_dataset("SimulaMet/Kvasir-VQA-x1")["test"].shuffle(seed=42).select(range(1500))

    This helper asks HuggingFace Datasets for the original shuffled indices, then
    applies those indices to the local JSONL-backed dataset. This preserves both
    subset membership and order for metric comparison.
    """
    subset_size = int(args.eval_samples or 1500)
    from datasets import load_dataset

    hf_test = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    if subset_size > len(hf_test):
        raise ValueError(f"Requested {subset_size} HF samples, but HF test split has only {len(hf_test)} rows")
    if len(ds.raw_records) < len(hf_test):
        raise ValueError(
            f"Local eval JSONL has {len(ds.raw_records)} rows, fewer than HF test split {len(hf_test)} rows. "
            "Cannot map HF shuffled indices safely."
        )

    hf_test = hf_test.add_column("__hf_original_index", list(range(len(hf_test))))
    hf_subset = hf_test.shuffle(seed=args.eval_shuffle_seed).select(range(subset_size))
    original_indices = [int(i) for i in hf_subset["__hf_original_index"]]
    ds.raw_records = [ds.raw_records[i] for i in original_indices]

    mismatch_examples: List[Dict[str, Any]] = []
    for local_pos, hf_sample in enumerate(hf_subset):
        _, record = ds.raw_records[local_pos]
        local_q, local_a = parse_question_answer(record)
        local_img = _image_stem_for_compare(get_image_ref(record))
        hf_img = _image_stem_for_compare(hf_sample.get("img_id", ""))
        q_ok = _clean_question_for_compare(local_q) == _clean_question_for_compare(hf_sample.get("question", ""))
        a_ok = _clean_answer_for_compare(local_a) == _clean_answer_for_compare(hf_sample.get("answer", ""))
        img_ok = (not hf_img) or local_img == hf_img
        if not (q_ok and a_ok and img_ok):
            mismatch_examples.append(
                {
                    "subset_position": local_pos,
                    "hf_original_index": original_indices[local_pos],
                    "local_image": local_img,
                    "hf_img_id": hf_img,
                    "local_question": _clean_question_for_compare(local_q),
                    "hf_question": _clean_question_for_compare(hf_sample.get("question", "")),
                    "local_answer": local_a,
                    "hf_answer": hf_sample.get("answer", ""),
                }
            )
            if len(mismatch_examples) >= 5:
                break

    metadata = {
        "enabled": True,
        "hf_dataset": "SimulaMet/Kvasir-VQA-x1",
        "split": "test",
        "shuffle_seed": args.eval_shuffle_seed,
        "subset_size": subset_size,
        "local_jsonl": args.eval_jsonl,
        "indices_are_zero_based_hf_original_indices": True,
        "original_indices": original_indices,
        "verification_mismatch_count_first_5": len(mismatch_examples),
        "verification_mismatch_examples": mismatch_examples,
    }
    (eval_dir / "hf_submission_subset_indices.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if mismatch_examples:
        raise ValueError(
            "HF submission subset verification failed: local JSONL order/content does not match HF test split. "
            f"See {eval_dir / 'hf_submission_subset_indices.json'}"
        )
    print(
        f"Using HF submission subset: {subset_size} rows | "
        f"shuffle_seed={args.eval_shuffle_seed} | metadata={eval_dir / 'hf_submission_subset_indices.json'}"
    )


@torch.no_grad()
def evaluate_model(model, args, device):
    eval_dir = Path(args.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ds = MedicoVQADataset(
        args.eval_jsonl,
        args.images_dir,
        args.eval_structural_manifest,
        strict_structural=False,
        max_samples=None if args.eval_hf_submission_subset else args.eval_samples,
    )
    if args.eval_hf_submission_subset:
        apply_hf_submission_subset(ds, args, eval_dir)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=medico_vqa_collate_fn)
    print(f"Eval dataset: {len(ds)} rows | batch_size={args.eval_batch_size} | batches={len(loader)}")
    model.eval()
    zero_prior_mask, zero_global_features = resolve_feature_zeroing(args)
    num_predictions = 0
    pred_path = eval_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Full eval"):
            batch = batch_to_device(batch, device, zero_prior_mask=zero_prior_mask, zero_global_features=zero_global_features)
            preds = model.generate(
                image=batch["image"],
                prior_mask=batch["prior_mask"],
                topo_features=batch["topo_features"],
                global_features=batch["global_features"],
                question_text=batch["question_text"],
                max_new_tokens=args.max_new_tokens,
            )
            for i, pred in enumerate(preds):
                q = batch["question_text"][i]
                gt = batch["answer_text"][i]
                row = {
                    "record_index": int(batch["record_index"][i].detach().cpu()),
                    "image_ref": batch["image_ref"][i],
                    "image_path": batch["image_path"][i],
                    "question": q,
                    "ground_truth": gt,
                    "prediction": pred,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_predictions += 1
    print(f"✅ Generation complete. Wrote {num_predictions} rows to: {pred_path}")
    print("Next: run scripts/evaluate_paper_metrics.py on this predictions.jsonl file.")
    return {"num_examples": num_predictions, "predictions_path": str(pred_path)}


def validation_selection_score(metrics: Dict[str, object], metric_name: str) -> Optional[float]:
    """Return the checkpoint-selection score from generation metrics.

    Higher is better. The default composite metric averages METEOR, ROUGE-L, and
    chrF++ because these are closer to generation quality than teacher-forcing
    loss for short medical VQA answers.
    """
    if not metrics:
        return None
    if metric_name == "composite":
        keys = ["meteor", "rougeL", "chrf_pp"]
        vals = []
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return (sum(vals) / len(vals)) if vals else None

    value = metrics.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


@torch.no_grad()
def generate_validation_metrics(model, loader, args, device, epoch: int, outdir: Path) -> Dict[str, object]:
    """Generate validation answers and compute paper-style metrics for checkpoint selection."""
    val_dir = outdir / "validation" / f"epoch_{epoch}"
    val_dir.mkdir(parents=True, exist_ok=True)
    pred_path = val_dir / "predictions.jsonl"
    model.eval()
    zero_prior_mask, zero_global_features = resolve_feature_zeroing(args)
    predictions: List[str] = []
    references: List[str] = []
    num_predictions = 0

    with pred_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"Val generation epoch {epoch}"):
            batch = batch_to_device(batch, device, zero_prior_mask=zero_prior_mask, zero_global_features=zero_global_features)
            preds = model.generate(
                image=batch["image"],
                prior_mask=batch["prior_mask"],
                topo_features=batch["topo_features"],
                global_features=batch["global_features"],
                question_text=batch["question_text"],
                max_new_tokens=args.max_new_tokens,
            )
            for i, pred in enumerate(preds):
                gt = batch["answer_text"][i]
                row = {
                    "record_index": int(batch["record_index"][i].detach().cpu()),
                    "image_ref": batch["image_ref"][i],
                    "image_path": batch["image_path"][i],
                    "question": batch["question_text"][i],
                    "ground_truth": gt,
                    "prediction": pred,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                predictions.append(pred)
                references.append(gt)
                num_predictions += 1

    try:
        from scripts.evaluate_paper_metrics import compute_metrics

        metrics = compute_metrics(
            predictions,
            references,
            use_bertscore=args.val_compute_bertscore,
            bertscore_model=args.val_bertscore_model,
            bertscore_batch_size=args.val_bertscore_batch_size,
        )
    except Exception as exc:
        metrics = {"num_examples": num_predictions, "metric_error": repr(exc)}

    score = validation_selection_score(metrics, args.val_selection_metric)
    metrics.update(
        {
            "epoch": epoch,
            "predictions_path": str(pred_path),
            "selection_metric": args.val_selection_metric,
            "selection_score": score,
        }
    )
    (val_dir / "paper_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    p.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    p.add_argument("--val-jsonl", default=None, help="Optional internal validation JSONL. Prefer an image-level split from official train for paper model selection.")
    p.add_argument("--eval-jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl")
    p.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    p.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    p.add_argument("--val-structural-manifest", default=None, help="Structural manifest for --val-jsonl. Defaults to --structural-manifest.")
    p.add_argument("--eval-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv")
    p.add_argument("--output-dir", default="outputs/qwen3b_topo_adapter")
    p.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--topo-mode", choices=["tda_only","all"], default="tda_only")
    p.add_argument("--visual-structural-mode", choices=["none", "tda_only", "all"], default="all")
    p.add_argument("--zero-prior-mask", type=str2bool, default=None, help="Zero prior_mask before model forward/generation. Default preserves topo_mode behavior: true for tda_only, false for all.")
    p.add_argument("--zero-global-features", type=str2bool, default=None, help="Zero global_features before model forward/generation. Default preserves topo_mode behavior: true for tda_only, false for all.")
    p.add_argument("--use-global-structural-token", type=str2bool, default=True)
    p.add_argument("--use-base-ot", type=str2bool, default=True)
    p.add_argument("--use-base-ot-fusion", type=str2bool, default=True)
    p.add_argument("--use-base-prior-as-ot-target", type=str2bool, default=None)
    p.add_argument("--use-base-topological-loss", type=str2bool, default=True)
    p.add_argument("--use-base-prior-align-loss", type=str2bool, default=None)
    p.add_argument("--use-base-global-topo-loss", type=str2bool, default=None)
    p.add_argument("--use-base-patch-topo-loss", type=str2bool, default=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--val-max-samples", type=int, default=None)
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument(
        "--eval-hf-submission-subset",
        type=str2bool,
        default=False,
        help="Use the exact public subset from submission_task1.py: HF test shuffle(seed) then select(range(eval_samples or 1500)).",
    )
    p.add_argument("--eval-shuffle-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--val-ratio", type=float, default=0.0)
    p.add_argument("--bottleneck-dim", type=int, default=32); p.add_argument("--adapter-last-n-layers", type=int, default=8); p.add_argument("--adapter-every-n-layers", type=int, default=0)
    p.add_argument("--vision-pretrained", type=str2bool, default=True); p.add_argument("--vision-backend", default="timm"); p.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument(
        "--val-selection-metric",
        default="composite",
        choices=["composite", "meteor", "rougeL", "chrf_pp", "bleu", "sacrebleu"],
        help="Generation metric used to save checkpoints/best.pt. Higher is better.",
    )
    p.add_argument("--val-compute-bertscore", type=str2bool, default=False)
    p.add_argument("--val-bertscore-model", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--val-bertscore-batch-size", type=int, default=16)
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--align-warmup-epochs", type=int, default=1)
    p.add_argument("--gate-loss-weight", type=float, default=0.001)
    p.add_argument("--warmup-text-only", type=str2bool, default=True)
    p.add_argument("--keep-patch-topo-during-warmup", type=str2bool, default=False)
    p.add_argument("--save-full-checkpoint", type=str2bool, default=False)
    p.add_argument(
        "--resume-model-only",
        type=str2bool,
        default=False,
        help="When resuming, load model weights only and reset optimizer/scheduler/RNG state.",
    )
    args = p.parse_args()

    set_reproducible_seed(args.seed); outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "eval":
        topo_dim = 36 if args.topo_mode == "tda_only" else 47
        zero_prior_mask, zero_global_features = resolve_feature_zeroing(args)
        base_flags = resolve_base_structural_flags(args)
        use_global_token = args.use_global_structural_token and args.visual_structural_mode == "all"
        print(f"Effective structural zeroing: zero_prior_mask={zero_prior_mask}; zero_global_features={zero_global_features}; use_global_token={use_global_token}")
        model=build_structural_generative_vqa(
            llm_name_or_path=args.llm_name_or_path, vision_pretrained=args.vision_pretrained, vision_backend=args.vision_backend,
            freeze_vision_backbone=args.freeze_vision_backbone, freeze_llm=True, use_lora=True, lora_r=16, lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", visual_structural_mode=args.visual_structural_mode,
            use_global_structural_token=use_global_token, use_ot=base_flags["use_ot"], use_ot_fusion=base_flags["use_ot_fusion"],
            use_prior_as_ot_target=base_flags["use_prior_as_ot_target"], use_topological_loss=base_flags["use_topological_loss"],
            use_prior_align_loss=base_flags["use_prior_align_loss"], use_global_topo_loss=base_flags["use_global_topo_loss"],
            use_patch_topo_loss=base_flags["use_patch_topo_loss"], prior_loss_weight=0.05 if base_flags["use_prior_align_loss"] else 0.0,
            global_topo_loss_weight=0.01 if base_flags["use_global_topo_loss"] else 0.0, patch_topo_loss_weight=0.005 if base_flags["use_patch_topo_loss"] else 0.0,
        ).to(device)
        if hasattr(model, "tokenizer"):
            model.tokenizer.padding_side = "left"
        wrappers=install_topo_adapters(model, topo_dim, args.bottleneck_dim, args.adapter_last_n_layers, args.adapter_every_n_layers)
        for wrapper in wrappers:
            wrapper.to(device)
        patch_model_forward(model, wrappers, args.topo_mode)
        ckpt_path = args.resume_checkpoint or str(outdir / "checkpoints" / "last.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_trainable_or_full_state(model, ckpt.get("model_state_dict", ckpt))
        print(f"Loaded eval checkpoint: {ckpt_path}")
        evaluate_model(model, args, device)
        print(f"Done: {outdir}")
        return

    train_ds=MedicoVQADataset(args.jsonl, args.images_dir, args.structural_manifest, strict_structural=False, max_samples=args.max_samples)
    if args.val_jsonl:
        val_manifest = args.val_structural_manifest or args.structural_manifest
        val_ds = MedicoVQADataset(
            args.val_jsonl,
            args.images_dir,
            val_manifest,
            strict_structural=False,
            max_samples=args.val_max_samples,
        )
        train_loader=DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=medico_vqa_collate_fn,
            worker_init_fn=seed_worker if args.num_workers>0 else None,
        )
        val_loader=DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=medico_vqa_collate_fn,
        )
        val_generation_loader=DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=medico_vqa_collate_fn,
        )
        print(
            f"Using explicit validation set: train={len(train_ds)} rows from {args.jsonl}; "
            f"val={len(val_ds)} rows from {args.val_jsonl}; val_manifest={val_manifest}"
        )
    else:
        val_size=int(len(train_ds)*args.val_ratio); train_size=len(train_ds)-val_size
        train_ds,val_ds=random_split(train_ds,[train_size,val_size],generator=torch.Generator().manual_seed(args.seed))
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=medico_vqa_collate_fn,worker_init_fn=seed_worker if args.num_workers>0 else None)
        val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=medico_vqa_collate_fn) if val_size else None
        val_generation_loader=DataLoader(val_ds,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=medico_vqa_collate_fn) if val_size else None
        if val_loader is not None:
            print(f"Using row-level random validation split: train={train_size} rows; val={val_size} rows; val_ratio={args.val_ratio}")
        else:
            print(f"No validation set configured: train={len(train_ds)} rows; val_ratio={args.val_ratio}")

    topo_dim = 36 if args.topo_mode == "tda_only" else 47  # 3*12 + (prior stats 3) + global 8
    zero_prior_mask, zero_global_features = resolve_feature_zeroing(args)
    base_flags = resolve_base_structural_flags(args)
    use_global_token = args.use_global_structural_token and args.visual_structural_mode == "all" and not zero_global_features
    print(f"Effective structural zeroing: zero_prior_mask={zero_prior_mask}; zero_global_features={zero_global_features}; use_global_token={use_global_token}")
    model=build_structural_generative_vqa(
        llm_name_or_path=args.llm_name_or_path, vision_pretrained=args.vision_pretrained, vision_backend=args.vision_backend,
        freeze_vision_backbone=args.freeze_vision_backbone, freeze_llm=True, use_lora=True, lora_r=16, lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", visual_structural_mode=args.visual_structural_mode,
        use_global_structural_token=use_global_token, use_ot=base_flags["use_ot"], use_ot_fusion=base_flags["use_ot_fusion"],
        use_prior_as_ot_target=base_flags["use_prior_as_ot_target"], use_topological_loss=base_flags["use_topological_loss"],
        use_prior_align_loss=base_flags["use_prior_align_loss"], use_global_topo_loss=base_flags["use_global_topo_loss"],
        use_patch_topo_loss=base_flags["use_patch_topo_loss"], prior_loss_weight=0.05 if base_flags["use_prior_align_loss"] else 0.0,
        global_topo_loss_weight=0.01 if base_flags["use_global_topo_loss"] else 0.0, patch_topo_loss_weight=0.005 if base_flags["use_patch_topo_loss"] else 0.0,
    ).to(device)
    wrappers=install_topo_adapters(model, topo_dim, args.bottleneck_dim, args.adapter_last_n_layers, args.adapter_every_n_layers)
    for wrapper in wrappers:
        wrapper.to(device)
    patch_model_forward(model, wrappers, args.topo_mode)
    base_cfg = {
        "ot_loss_weight": model.config.ot_loss_weight,
        "prior_loss_weight": model.config.prior_loss_weight,
        "global_topo_loss_weight": model.config.global_topo_loss_weight,
        "patch_topo_loss_weight": model.config.patch_topo_loss_weight,
        "use_ot": model.config.use_ot,
        "use_ot_fusion": model.config.use_ot_fusion,
        "use_topological_loss": model.config.use_topological_loss,
        "use_prior_as_ot_target": model.config.use_prior_as_ot_target,
        "use_prior_align_loss": model.config.use_prior_align_loss,
        "use_global_topo_loss": model.config.use_global_topo_loss,
        "use_patch_topo_loss": model.config.use_patch_topo_loss,
    }
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_updates=max(1, math.ceil(len(train_loader)/args.gradient_accumulation_steps)*args.epochs)
    sched=build_lr_scheduler(opt,"cosine",min(1000,total_updates//10),total_updates)
    start_epoch = 1
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        load_trainable_or_full_state(model, ckpt.get("model_state_dict", ckpt))
        if args.resume_model_only:
            start_epoch = 1
            print(
                f"Loaded model weights only from {args.resume_checkpoint}; "
                "optimizer/scheduler/RNG reset; start_epoch=1"
            )
        else:
            if ckpt.get("optimizer_state_dict") is not None:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                sched.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"Resumed from {args.resume_checkpoint}; start_epoch={start_epoch}")
    log_path=outdir/"train_log.jsonl"; opt.zero_grad(set_to_none=True)
    best_val_score = float("-inf")
    best_epoch = None
    if args.mode in {"train", "train_eval"}:
        for ep in range(start_epoch,args.epochs+1):
            warmup_active = apply_curriculum_schedule(model, ep, args, base_cfg)
            print(f"Epoch {ep}: curriculum_warmup_active={warmup_active}; gate_loss_weight={args.gate_loss_weight}")
            t0=time.time(); train_m=run_epoch(model,train_loader,opt,device,ep,True,args.gradient_accumulation_steps,sched,zero_prior_mask,zero_global_features,wrappers,args.gate_loss_weight)
            val_m=run_epoch(model,val_loader,opt,device,ep,False,1,None,zero_prior_mask,zero_global_features,wrappers,0.0) if val_loader else {}
            val_gen_m=generate_validation_metrics(model,val_generation_loader,args,device,ep,outdir) if val_generation_loader else {}
            preview=preview_generation(model,train_loader,device)
            val_score = validation_selection_score(val_gen_m, args.val_selection_metric) if val_gen_m else None
            is_best = val_score is not None and val_score > best_val_score
            if is_best:
                best_val_score = float(val_score)
                best_epoch = ep
            row={
                "epoch":ep,
                "elapsed_sec":time.time()-t0,
                "train":train_m,
                "val":val_m,
                "val_generation":val_gen_m,
                "val_selection_metric":args.val_selection_metric,
                "val_selection_score":val_score,
                "is_best":is_best,
                "best_epoch":best_epoch,
                "best_val_score":None if best_epoch is None else best_val_score,
                "preview":preview,
                "args":vars(args),
            }
            log_path.open("a",encoding="utf-8").write(json.dumps(row,ensure_ascii=False)+"\n")
            print("Validation loss diagnostics:", val_m if val_m else "not configured")
            print("Validation generation metrics:", json.dumps(val_gen_m, ensure_ascii=False, indent=2) if val_gen_m else "not configured")
            if is_best:
                print(f"✅ New best validation generation score ({args.val_selection_metric}): {best_val_score:.6f} at epoch {ep}")
            print("Preview:", preview)
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_dir/f"epoch_{ep}.pt", model, opt, sched, ep, row, args)
            save_checkpoint(ckpt_dir/"last.pt", model, opt, sched, ep, row, args)
            if is_best:
                save_checkpoint(ckpt_dir/"best.pt", model, opt, sched, ep, row, args)
        model.tokenizer.save_pretrained(outdir/"tokenizer")
        if hasattr(model.llm,"save_pretrained"): model.llm.save_pretrained(outdir/"lora_adapter")
    if args.mode in {"eval", "train_eval"}:
        if args.mode == "eval":
            ckpt_path = args.resume_checkpoint or str(outdir / "checkpoints" / "last.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            load_trainable_or_full_state(model, ckpt.get("model_state_dict", ckpt))
            print(f"Loaded eval checkpoint: {ckpt_path}")
        evaluate_model(model, args, device)
    print(f"Done: {outdir}")

if __name__ == "__main__":
    main()
