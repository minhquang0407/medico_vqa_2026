"""PaliGemma QLoRA + Structural Adapter for Medico/Kvasir VQA.

Injects offline lesion-prior/TDA features as learned soft tokens in the embedding
stream instead of textual hints.
"""
from __future__ import annotations

import argparse, csv, json, math, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Image as HfImage, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_SPACE_RE = re.compile(r"\s+")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)+", re.I)


def normalize_answer(x: str) -> str:
    x = _SPACE_RE.sub(" ", str(x or "")).strip()
    x = _ANSWER_PREFIX_RE.sub("", x)
    x = re.sub(r"\s+([,.;:!?])", r"\1", x)
    return _SPACE_RE.sub(" ", x).strip()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def image_key(sample: Dict[str, Any]) -> str:
    if sample.get("img_id"):
        return Path(str(sample["img_id"])).stem
    image = sample.get("image")
    path = getattr(image, "filename", None)
    return Path(str(path)).stem if path else ""


def prompt(question: str) -> str:
    return f"<image>answer en {str(question).strip()}"


class StructuralTensorStore:
    """Preloads compact structural tensors into RAM using manifest CSVs."""
    def __init__(self, root: str | Path, train_manifest: str, eval_manifest: str, max_topo_channels: int = 12, mode: str = "all"):
        self.root = Path(root)
        self.max_topo_channels = max_topo_channels
        self.mode = str(mode or "all").lower()
        if self.mode not in {"all", "tda_only"}:
            raise ValueError("structural mode must be 'all' or 'tda_only'")
        self.index: Dict[str, Dict[str, np.ndarray]] = {}
        for m in [train_manifest, eval_manifest]:
            p = Path(m)
            if not p.is_absolute():
                p = self.root / p
            self._load_manifest(p)
        print(f"Loaded structural tensor index: {len(self.index)} images")

    def _load_manifest(self, manifest: Path):
        if not manifest.exists():
            print(f"Warning: manifest missing {manifest}"); return
        manifest_dir = manifest.parent
        by_name = {p.name: p for p in manifest_dir.rglob("*.npz")}
        loaded = errors = 0
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") not in {"processed", "skipped_existing", ""}: continue
                image_ref = row.get("image_ref") or row.get("image_path") or ""
                cache = (row.get("cache_path") or "").replace("\\", "/")
                if not cache: continue
                cp = Path(cache); name = cp.name
                if cp.is_absolute() and cp.exists(): resolved = cp
                elif cp.exists(): resolved = cp.resolve()
                elif (self.root / cp).exists(): resolved = self.root / cp
                elif name in by_name: resolved = by_name[name]
                else: continue
                key = Path(str(image_ref)).stem
                try:
                    self.index[key] = self._read_npz(resolved); loaded += 1
                except Exception:
                    errors += 1
        print(f"  {manifest.name}: {loaded} structural tensors cached, {errors} errors")

    def _read_npz(self, path: Path) -> Dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as d:
            if self.mode == "tda_only":
                # Only topology-derived features. No prior mask, red map, center map,
                # morpho prior, or global priors are allowed in this ablation.
                base_maps = [np.zeros((14, 14, 1), dtype=np.float32) for _ in range(5)]
                glob = np.zeros(16, dtype=np.float32)
            else:
                base_maps = []
                for k in ["prior_mask", "red_map", "center_map", "morpho_prior_map", "topo_mask"]:
                    base_maps.append(np.asarray(d.get(k, np.zeros((14, 14))), dtype=np.float32)[..., None])
                glob = np.asarray(d.get("global_features", np.zeros(16)), dtype=np.float32).reshape(-1)
                if glob.size < 16: glob = np.pad(glob, (0, 16-glob.size))
                glob = glob[:16].astype(np.float32)

            topo = np.asarray(d.get("topo_features", np.zeros((14, 14, self.max_topo_channels))), dtype=np.float32)
            if topo.ndim != 3: topo = np.zeros((14, 14, self.max_topo_channels), dtype=np.float32)
            if topo.shape[-1] < self.max_topo_channels:
                pad = np.zeros((14, 14, self.max_topo_channels - topo.shape[-1]), dtype=np.float32)
                topo = np.concatenate([topo, pad], axis=-1)
            topo = topo[..., :self.max_topo_channels]
            grid = np.concatenate(base_maps + [topo], axis=-1).astype(np.float32)  # 14,14,C
        return {"grid": grid, "global": glob}

    def get(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return self.index.get(image_key(sample), {"grid": np.zeros((14,14,17), np.float32), "global": np.zeros(16, np.float32)})


class StructuralAdapter(nn.Module):
    def __init__(self, in_channels: int, global_dim: int, hidden_size: int, struct_tokens: int = 8, adapter_hidden: int = 512):
        super().__init__()
        self.struct_tokens = struct_tokens
        self.patch_proj = nn.Sequential(nn.Linear(in_channels, adapter_hidden), nn.GELU(), nn.Linear(adapter_hidden, hidden_size))
        self.query = nn.Parameter(torch.randn(struct_tokens, hidden_size) * 0.02)
        self.global_proj = nn.Sequential(nn.Linear(global_dim, adapter_hidden), nn.GELU(), nn.Linear(adapter_hidden, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, grid: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        b = grid.shape[0]
        patches = self.patch_proj(grid.reshape(b, -1, grid.shape[-1]))  # B,196,H
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        attn = torch.softmax(torch.matmul(q, patches.transpose(1,2)) / math.sqrt(patches.shape[-1]), dim=-1)
        pooled = torch.matmul(attn, patches)  # B,K,H
        g = self.global_proj(global_features).unsqueeze(1)
        return self.norm(torch.cat([pooled, g], dim=1))


class PaliGemmaStructuralWrapper(nn.Module):
    def __init__(self, model, adapter: StructuralAdapter, image_token_id: Optional[int]):
        super().__init__(); self.model = model; self.adapter = adapter; self.image_token_id = image_token_id

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _insert(self, input_ids, attention_mask, labels, grid, global_features):
        embed_layer = self.model.get_input_embeddings()
        embeds = embed_layer(input_ids)
        st = self.adapter(grid.to(embeds.dtype), global_features.to(embeds.dtype)).to(embeds.device)
        b, k, h = st.shape
        out_e, out_m, out_l = [], [], []
        for i in range(b):
            ids = input_ids[i]
            if self.image_token_id is not None:
                pos = (ids == self.image_token_id).nonzero(as_tuple=False).flatten()
                ins = int(pos[-1].item() + 1) if len(pos) else 1
            else:
                ins = 1
            out_e.append(torch.cat([embeds[i, :ins], st[i], embeds[i, ins:]], dim=0))
            out_m.append(torch.cat([attention_mask[i, :ins], torch.ones(k, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask[i, ins:]], dim=0))
            if labels is not None:
                pad = torch.full((k,), -100, device=labels.device, dtype=labels.dtype)
                out_l.append(torch.cat([labels[i, :ins], pad, labels[i, ins:]], dim=0))
        return torch.stack(out_e), torch.stack(out_m), torch.stack(out_l) if labels is not None else None

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, grid=None, global_features=None, **kw):
        inputs_embeds, attention_mask, labels = self._insert(input_ids, attention_mask, labels, grid, global_features)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, **kw)

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, grid=None, global_features=None, **kw):
        inputs_embeds, attention_mask, _ = self._insert(input_ids, attention_mask, None, grid, global_features)
        return self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, pixel_values=pixel_values, **kw)


@dataclass
class Collator:
    processor: Any; store: StructuralTensorStore; max_length: int = 512; train: bool = True; transform: Any = None
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        imgs = [apply_aug(ex["image"], self.transform if self.train else None) for ex in examples]
        texts = [prompt(ex["question"]) for ex in examples]
        answers = [normalize_answer(ex.get("answer", "")) for ex in examples]
        batch = self.processor(text=texts, images=imgs, suffix=answers if self.train else None, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt")
        if self.train and "labels" not in batch:
            labels = batch["input_ids"].clone(); labels[labels == self.processor.tokenizer.pad_token_id] = -100; batch["labels"] = labels
        structs = [self.store.get(ex) for ex in examples]
        batch["grid"] = torch.tensor(np.stack([s["grid"] for s in structs]), dtype=torch.float32)
        batch["global_features"] = torch.tensor(np.stack([s["global"] for s in structs]), dtype=torch.float32)
        return batch


def apply_aug(image: Image.Image, transform) -> Image.Image:
    if transform is None: return image.convert("RGB")
    arr = np.asarray(image.convert("RGB")); return Image.fromarray(transform(image=arr)["image"]).convert("RGB")


def aug(enabled: bool):
    if not enabled: return None
    import albumentations as A
    return A.Compose([A.Affine(scale=(0.9,1.1), translate_percent=(-0.1,0.1), rotate=(-10,10), p=.75), A.ColorJitter(brightness=.12, contrast=.12, saturation=.10, hue=.03, p=.60)])


def data(args):
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")
    tr = ds["train"].cast_column("image", HfImage()); te = ds["test"].cast_column("image", HfImage())
    if args.max_train_samples: tr = tr.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(tr))))
    if args.eval_samples: te = te.shuffle(seed=args.seed).select(range(min(args.eval_samples, len(te))))
    return tr, te


def load(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model_name); proc.tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    base = PaliGemmaForConditionalGeneration.from_pretrained(args.model_name, quantization_config=qcfg, torch_dtype=dtype, device_map="auto")
    base = prepare_model_for_kbit_training(base)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=[x.strip() for x in args.lora_target_modules.split(",")], bias="none", task_type="CAUSAL_LM")
    base = get_peft_model(base, lora); base.print_trainable_parameters()
    hidden = base.get_input_embeddings().embedding_dim
    adapter = StructuralAdapter(17, 16, hidden, args.struct_tokens, args.adapter_hidden).to(base.device)
    image_token_id = getattr(proc, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<image>")
    print("image_token_id:", image_token_id, "hidden:", hidden)
    return PaliGemmaStructuralWrapper(base, adapter, image_token_id), proc


def save_all(wrapper, proc, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    wrapper.model.save_pretrained(out); proc.save_pretrained(out)
    torch.save(wrapper.adapter.state_dict(), out / "structural_adapter.pt")


def train(args):
    from transformers import get_cosine_schedule_with_warmup
    set_seed(args.seed); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tr, te = data(args); model, proc = load(args)
    store = StructuralTensorStore(args.structural_root, args.train_structural_manifest, args.eval_structural_manifest, mode=args.structural_mode)
    coll = Collator(proc, store, args.max_length, True, aug(args.use_augmentation))
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=coll, pin_memory=torch.cuda.is_available())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total = max(1, math.ceil(len(dl)/args.gradient_accumulation_steps) * args.epochs)
    warm = args.warmup_steps if args.warmup_steps >= 0 else int(.03 * total)
    sch = get_cosine_schedule_with_warmup(opt, warm, total)
    model.train(); opt.zero_grad(set_to_none=True); gstep = 0; start=time.time()
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"Training epoch {ep+1}/{args.epochs}")
        loss_sum = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k,v in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward(); loss_sum += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            if (step+1) % args.gradient_accumulation_steps == 0 or (step+1) == len(dl):
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); gstep += 1
                pbar.set_postfix(loss=loss_sum/max(1, step+1), step=gstep)
        save_all(model, proc, out / f"epoch-{ep+1}")
    save_all(model, proc, out)
    (out/"training_metadata.json").write_text(json.dumps({"args":vars(args), "global_step":gstep, "elapsed_sec":round(time.time()-start,2)}, indent=2), encoding="utf-8")
    return str(out)


@torch.inference_mode()
def gen(model, proc, image, question, struct, max_new):
    batch = proc(text=prompt(question), images=image.convert("RGB"), return_tensors="pt")
    batch["grid"] = torch.tensor(struct["grid"][None], dtype=torch.float32)
    batch["global_features"] = torch.tensor(struct["global"][None], dtype=torch.float32)
    batch = {k:v.to(model.device) for k,v in batch.items()}
    out = model.generate(**batch, max_new_tokens=max_new, do_sample=False, num_beams=1, pad_token_id=proc.tokenizer.pad_token_id)
    text = proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return normalize_answer(text)


def evaluate(args, ckpt=None):
    from evaluate import load as metric
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
    set_seed(args.seed); out=Path(args.output_dir); ckpt=Path(ckpt or args.output_dir)
    proc = AutoProcessor.from_pretrained(ckpt)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    base = PaliGemmaForConditionalGeneration.from_pretrained(args.model_name, quantization_config=qcfg, torch_dtype=dtype, device_map="auto")
    base = PeftModel.from_pretrained(base, ckpt)
    hidden = base.get_input_embeddings().embedding_dim
    adapter = StructuralAdapter(17,16,hidden,args.struct_tokens,args.adapter_hidden).to(base.device)
    adapter.load_state_dict(torch.load(ckpt/"structural_adapter.pt", map_location=base.device))
    model = PaliGemmaStructuralWrapper(base, adapter, getattr(proc,"image_token_id",None) or proc.tokenizer.convert_tokens_to_ids("<image>")); model.eval()
    store = StructuralTensorStore(args.structural_root, args.train_structural_manifest, args.eval_structural_manifest, mode=args.structural_mode)
    _, te = data(args); preds=[]; refs=[]
    for i,s in enumerate(tqdm(te, desc="Evaluating")):
        pred = gen(model, proc, s["image"], s["question"], store.get(s), args.max_new_tokens); ref=normalize_answer(s["answer"])
        preds.append({"index":i,"img_id":str(s.get("img_id","")),"question":str(s["question"]),"answer":pred,"reference":ref}); refs.append([ref])
    txt=[p["answer"] for p in preds]
    bleu=metric("bleu").compute(predictions=txt, references=refs); rouge=metric("rouge").compute(predictions=txt, references=[r[0] for r in refs]); meteor=metric("meteor").compute(predictions=txt, references=[r[0] for r in refs])
    scores={"bleu":round(float(bleu["bleu"]),4),"rouge1":round(float(rouge["rouge1"]),4),"rouge2":round(float(rouge["rouge2"]),4),"rougeL":round(float(rouge["rougeL"]),4),"meteor":round(float(meteor["meteor"]),4)}
    print("Scores:", scores); (out/"paligemma_structural_adapter_predictions.json").write_text(json.dumps({"scores":scores,"predictions":preds}, ensure_ascii=False, indent=2), encoding="utf-8")


def args():
    p=argparse.ArgumentParser(); p.add_argument("--mode", choices=["train","eval","train_eval"], default="train_eval"); p.add_argument("--model-name", default="google/paligemma-3b-pt-224"); p.add_argument("--output-dir", default="outputs/paligemma_struct_adapter")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--gradient-accumulation-steps", type=int, default=8); p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--weight-decay", type=float, default=.01); p.add_argument("--warmup-steps", type=int, default=-1); p.add_argument("--max-grad-norm", type=float, default=1.0); p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=None); p.add_argument("--eval-samples", type=int, default=1500); p.add_argument("--max-length", type=int, default=512); p.add_argument("--max-new-tokens", type=int, default=48); p.add_argument("--use-augmentation", action="store_true")
    p.add_argument("--structural-root", default="."); p.add_argument("--train-structural-manifest", default="data/processed/structural_features/train_original_manifest.csv"); p.add_argument("--eval-structural-manifest", default="data/processed/structural_features/test_original_manifest.csv"); p.add_argument("--structural-mode", choices=["all", "tda_only"], default="all"); p.add_argument("--struct-tokens", type=int, default=8); p.add_argument("--adapter-hidden", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16); p.add_argument("--lora-alpha", type=int, default=32); p.add_argument("--lora-dropout", type=float, default=.05); p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return p.parse_args()


def main():
    a=args(); ck=None
    if a.mode in {"train","train_eval"}: ck=train(a)
    if a.mode in {"eval","train_eval"}: evaluate(a, ck)

if __name__ == "__main__": main()
