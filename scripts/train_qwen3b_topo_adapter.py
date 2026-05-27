"""Qwen3B Topological Adapter training.

New deep-fusion adapter: topology/global structural conditions modulate hidden
states inside selected Qwen decoder blocks through zero-init gated bottlenecks.
Supports both strict TDA-only and full-structural modes.
"""
from __future__ import annotations

import argparse, json, math, os, random, types, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.dataset import MedicoVQADataset, medico_vqa_collate_fn
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

    def forward(self, hidden_states: torch.Tensor, topo_condition: torch.Tensor) -> torch.Tensor:
        if topo_condition.ndim == 2:
            topo_condition = topo_condition.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        topo_condition = topo_condition.to(device=hidden_states.device, dtype=self.topo_projector[0].weight.dtype)
        hs = hidden_states.to(dtype=self.down_proj.weight.dtype)
        t_emb = self.topo_projector(topo_condition)
        gate = torch.sigmoid(self.gating(t_emb))
        delta = self.up_proj(self.down_proj(hs) * gate).to(dtype=hidden_states.dtype)
        return hidden_states + delta


class DecoderLayerWithTopoAdapter(nn.Module):
    def __init__(self, base_layer: nn.Module, adapter: TopoAdapter):
        super().__init__()
        self.base_layer = base_layer
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


def batch_to_device(batch, device, topo_mode: str):
    for key in ["image", "prior_mask", "topo_features", "global_features"]:
        batch[key] = batch[key].to(device)
    if topo_mode == "tda_only":
        batch["prior_mask"] = torch.zeros_like(batch["prior_mask"])
        batch["global_features"] = torch.zeros_like(batch["global_features"])
    return batch


def run_epoch(model, loader, optimizer, device, epoch, train=True, grad_accum=1, scheduler=None, topo_mode="tda_only"):
    model.train(train); totals = {"loss":0.0,"lm_loss":0.0,"ot_cost":0.0,"topological_loss":0.0}; steps=0
    iterator = tqdm(loader, desc=("Train" if train else "Eval") + f" epoch {epoch}")
    for batch in iterator:
        batch = batch_to_device(batch, device, topo_mode)
        with torch.set_grad_enabled(train):
            out = model(image=batch["image"], prior_mask=batch["prior_mask"], topo_features=batch["topo_features"], global_features=batch["global_features"], question_text=batch["question_text"], answer_text=batch["answer_text"], return_diagnostics=True)
            loss = out["loss"]
            if train:
                (loss / grad_accum).backward()
                if steps % grad_accum == grad_accum - 1:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None: scheduler.step()
        for k in totals: totals[k] += float(out.get(k, torch.tensor(0.0)).detach().cpu())
        steps += 1
        iterator.set_postfix(loss=totals["loss"]/steps, lr=current_lr(optimizer))
    if train and steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if scheduler is not None: scheduler.step()
    return {k:v/max(steps,1) for k,v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/raw/Kvasir-VQA-x1/Kvasir-VQA-x1-train.jsonl")
    p.add_argument("--images-dir", default="data/raw/Kvasir-VQA-x1/images")
    p.add_argument("--structural-manifest", default="data/processed/structural_features/train_original_manifest.csv")
    p.add_argument("--output-dir", default="outputs/qwen3b_topo_adapter")
    p.add_argument("--llm-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--topo-mode", choices=["tda_only","all"], default="tda_only")
    p.add_argument("--max-samples", type=int, default=None); p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--val-ratio", type=float, default=0.0)
    p.add_argument("--bottleneck-dim", type=int, default=32); p.add_argument("--adapter-last-n-layers", type=int, default=8); p.add_argument("--adapter-every-n-layers", type=int, default=0)
    p.add_argument("--vision-pretrained", type=str2bool, default=True); p.add_argument("--vision-backend", default="timm"); p.add_argument("--freeze-vision-backbone", type=str2bool, default=True)
    p.add_argument("--save-full-checkpoint", type=str2bool, default=False)
    args = p.parse_args()

    set_reproducible_seed(args.seed); outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=MedicoVQADataset(args.jsonl, args.images_dir, args.structural_manifest, strict_structural=False, max_samples=args.max_samples)
    val_size=int(len(ds)*args.val_ratio); train_size=len(ds)-val_size
    train_ds,val_ds=random_split(ds,[train_size,val_size],generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=medico_vqa_collate_fn,worker_init_fn=seed_worker if args.num_workers>0 else None)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=medico_vqa_collate_fn) if val_size else None

    topo_dim = 36 if args.topo_mode == "tda_only" else 47  # 3*12 + (prior stats 3) + global 8
    model=build_structural_generative_vqa(
        llm_name_or_path=args.llm_name_or_path, vision_pretrained=args.vision_pretrained, vision_backend=args.vision_backend,
        freeze_vision_backbone=args.freeze_vision_backbone, freeze_llm=True, use_lora=True, lora_r=16, lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", use_ot=True, use_ot_fusion=True,
        use_prior_as_ot_target=(args.topo_mode=="all"), use_topological_loss=True, use_prior_align_loss=(args.topo_mode=="all"),
        use_global_topo_loss=(args.topo_mode=="all"), use_patch_topo_loss=True, prior_loss_weight=0.05 if args.topo_mode=="all" else 0.0,
        global_topo_loss_weight=0.01 if args.topo_mode=="all" else 0.0, patch_topo_loss_weight=0.005,
    ).to(device)
    wrappers=install_topo_adapters(model, topo_dim, args.bottleneck_dim, args.adapter_last_n_layers, args.adapter_every_n_layers)
    patch_model_forward(model, wrappers, args.topo_mode)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_updates=max(1, math.ceil(len(train_loader)/args.gradient_accumulation_steps)*args.epochs)
    sched=build_lr_scheduler(opt,"cosine",min(1000,total_updates//10),total_updates)
    log_path=outdir/"train_log.jsonl"; opt.zero_grad(set_to_none=True)
    for ep in range(1,args.epochs+1):
        t0=time.time(); train_m=run_epoch(model,train_loader,opt,device,ep,True,args.gradient_accumulation_steps,sched,args.topo_mode)
        val_m=run_epoch(model,val_loader,opt,device,ep,False,1,None,args.topo_mode) if val_loader else {}
        preview=preview_generation(model,train_loader,device)
        row={"epoch":ep,"elapsed_sec":time.time()-t0,"train":train_m,"val":val_m,"preview":preview,"args":vars(args)}
        log_path.open("a",encoding="utf-8").write(json.dumps(row,ensure_ascii=False)+"\n")
        print("Preview:", preview)
        save_checkpoint(outdir/f"epoch_{ep}.pt", model, opt, sched, ep, row, args)
        save_checkpoint(outdir/"last.pt", model, opt, sched, ep, row, args)
    model.tokenizer.save_pretrained(outdir/"tokenizer")
    if hasattr(model.llm,"save_pretrained"): model.llm.save_pretrained(outdir/"lora_adapter")
    print(f"Done: {outdir}")

if __name__ == "__main__":
    main()
