"""Generate a fresh MediaEval Medico 2026 Subtask 2 submission with CATA-Final.

This script regenerates *everything* for Task 2:
- primary answers from the current CATA checkpoint,
- targeted self-probing answers,
- heatmap PNGs,
- evidence JSON files,
- clinician-oriented textual explanations,
- confidence scores.

It intentionally does not import ``hf_submission/task1_cata/submission_task1.py``
because that file executes Task 1 inference at import time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Image as HfImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
# In the combined HF repo, Task 1 and Task 2 live in the same directory:
#   submission_task1.py, generate_task2_cata_final.py, checkpoints/, src/
HF_REPO_DIR = SCRIPT_DIR
# Local development checkout root: .../medico_vqa_2026
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name == "hf_submission" else SCRIPT_DIR
LOCAL_OUTPUT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name == "hf_submission" else SCRIPT_DIR

for path in (HF_REPO_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from src.models.structural_vqa_generative import build_structural_generative_vqa
except Exception as exc:  # pragma: no cover - environment dependent
    build_structural_generative_vqa = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_TOPO_FEATURE_DIM = 12
DEFAULT_GLOBAL_FEATURE_DIM = 8

QUESTION_FAMILIES = {
    "visible_text": ["text", "label", "caption", "writing"],
    "instrument": ["instrument", "surgical", "tool", "forceps", "tube", "catheter", "foreign"],
    "count": ["how many", "number of", "count"],
    "color": ["color", "colours", "colors", "red", "white", "pink", "black", "green"],
    "location": ["where", "which region", "location", "located", "position", "region"],
    "procedure": ["procedure", "obtained", "depicted"],
    "size": ["size", "large", "small", "millimeter", "mm"],
    "lesion": ["polyp", "lesion", "abnormal", "finding", "mucosa", "tissue"],
}

PROBES = {
    "visible_text": [
        "Is there any visible text or label in this image?",
        "Where is any visible text located in the image?",
        "Is the visible text an image overlay rather than anatomy?",
    ],
    "instrument": [
        "Are any medical instruments visible in the image?",
        "Where are visible instruments located?",
        "Are there no instruments in the endoscopic field?",
    ],
    "count": [
        "How many distinct abnormal regions are visible?",
        "Is there a single focal lesion or multiple lesions?",
        "Are there no significant abnormalities visible?",
    ],
    "color": [
        "What colors are visible in the relevant abnormal area?",
        "Is redness visible in the mucosa?",
        "Are pale, white, yellow, dark, or mixed color changes visible?",
    ],
    "location": [
        "Where is the relevant abnormality located in the image?",
        "Is the evidence central, peripheral, upper, lower, left, or right?",
        "Is the evidence focal or scattered across multiple regions?",
    ],
    "procedure": [
        "What endoscopic procedure or anatomical context is shown?",
        "Does the image show gastrointestinal mucosa?",
        "Are artifacts or overlays important for interpreting this image?",
    ],
    "size": [
        "What is the approximate relative size of the relevant lesion?",
        "Is the lesion small, moderate, or large relative to the image?",
        "Is the finding focal enough for a size estimate?",
    ],
    "lesion": [
        "Is there evidence of a polypoid lesion or abnormal mucosa?",
        "Describe the shape and focality of the lesion if present.",
        "What visible mucosal features support or weaken the finding?",
    ],
    "generic": [
        "What visual evidence is most relevant to answer the question?",
        "Are there artifacts that could affect the answer?",
        "Is the evidence focal or diffuse?",
    ],
}

NEGATIVE_TERMS = (
    "no ", "not ", "none", "without", "absent", "no evidence", "not identified", "not observed"
)
POSITIVE_TERMS = (
    "evidence", "visible", "present", "observed", "identified", "lesion", "polyp", "abnormal", "instrument", "text"
)


class TopoAdapter(torch.nn.Module):
    def __init__(self, hidden_dim: int, topo_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.topo_projector = torch.nn.Sequential(
            torch.nn.Linear(topo_dim, bottleneck_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(bottleneck_dim, bottleneck_dim),
        )
        self.down_proj = torch.nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.gating = torch.nn.Linear(bottleneck_dim, bottleneck_dim)
        self.up_proj = torch.nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        torch.nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states: torch.Tensor, topo_condition: torch.Tensor) -> torch.Tensor:
        if topo_condition.ndim == 2:
            topo_condition = topo_condition.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        topo_condition = topo_condition.to(hidden_states.device, dtype=self.topo_projector[0].weight.dtype)
        topo_condition = torch.nan_to_num(topo_condition, nan=0.0, posinf=0.0, neginf=0.0)
        hs = hidden_states.to(dtype=self.down_proj.weight.dtype)
        t_emb = self.topo_projector(topo_condition)
        gate = torch.sigmoid(self.gating(t_emb))
        delta = self.up_proj(self.down_proj(hs) * gate).to(dtype=hidden_states.dtype)
        return hidden_states + delta


class DecoderLayerWithTopoAdapter(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module, adapter: TopoAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.topo_adapter = adapter
        self.topo_condition: Optional[torch.Tensor] = None
        if hasattr(base_layer, "attention_type"):
            self.attention_type = base_layer.attention_type

    def set_topo_condition(self, condition: Optional[torch.Tensor]):
        self.topo_condition = condition

    def forward(self, *args, **kwargs):
        out = self.base_layer(*args, **kwargs)
        if self.topo_condition is None:
            return out
        if isinstance(out, tuple):
            return (self.topo_adapter(out[0], self.topo_condition),) + out[1:]
        return self.topo_adapter(out, self.topo_condition)


def _find_decoder_layers(llm) -> torch.nn.ModuleList:
    for dotted in ("model.layers", "base_model.model.model.layers", "base_model.model.layers", "model.model.layers"):
        obj = llm
        ok = True
        for part in dotted.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Cannot locate Qwen decoder layers")


def _install_topo_adapters(model, topo_dim: int, bottleneck_dim: int, last_n_layers: int, every_n_layers: int):
    layers = _find_decoder_layers(model.llm)
    hidden_dim = int(model.llm.get_input_embeddings().embedding_dim)
    selected = set(range(max(0, len(layers) - last_n_layers), len(layers))) if last_n_layers > 0 else set()
    if every_n_layers > 0:
        selected.update(range(0, len(layers), every_n_layers))
    wrappers = []
    for idx in sorted(selected):
        if isinstance(layers[idx], DecoderLayerWithTopoAdapter):
            wrappers.append(layers[idx])
            continue
        wrapper = DecoderLayerWithTopoAdapter(layers[idx], TopoAdapter(hidden_dim, topo_dim, bottleneck_dim))
        layers[idx] = wrapper
        wrappers.append(wrapper)
    print(f"Installed {len(wrappers)} TopoAdapters / {len(layers)} layers | topo_dim={topo_dim}")
    return wrappers


def _topo_condition(prior_mask: torch.Tensor, topo_features: torch.Tensor, global_features: torch.Tensor, mode: str):
    tf = topo_features.float()
    flat = tf.flatten(1, 2)
    parts = [flat.mean(1), flat.std(1), flat.amax(1)]
    if mode == "all":
        pm = prior_mask.float().flatten(1)
        parts.extend([pm.mean(1, keepdim=True), pm.std(1, keepdim=True), pm.amax(1, keepdim=True)])
        parts.append(global_features.float())
    return torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def _patch_generate_with_topo(model, wrappers, mode: str):
    original_generate = model.generate

    @torch.no_grad()
    def generate_with_topo(self, image, prior_mask, topo_features, global_features, question_text, max_new_tokens=64):
        condition = _topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        for wrapper in wrappers:
            wrapper.set_topo_condition(condition)
        try:
            return original_generate(image, prior_mask, topo_features, global_features, question_text, max_new_tokens)
        finally:
            for wrapper in wrappers:
                wrapper.set_topo_condition(None)

    import types

    model.generate = types.MethodType(generate_with_topo, model)


def safe_load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            return loaded
        return {"model_state_dict": loaded, "args": {}}


def load_cata_model(checkpoint_path: Path, device: torch.device):
    if build_structural_generative_vqa is None:
        raise ImportError(f"Could not import project model code: {_IMPORT_ERROR!r}")
    checkpoint = safe_load_checkpoint(checkpoint_path)
    train_args = checkpoint.get("args", {})
    config = {**train_args}
    config.update({
        "llm_name_or_path": train_args.get("llm_name_or_path", "Qwen/Qwen2.5-3B-Instruct"),
        "vision_pretrained": train_args.get("vision_pretrained", True),
        "vision_backend": train_args.get("vision_backend", "timm"),
        "freeze_vision_backbone": train_args.get("freeze_vision_backbone", True),
        "freeze_llm": train_args.get("freeze_llm", True),
        "use_lora": train_args.get("use_lora", True),
        "lora_r": train_args.get("lora_r", 16),
        "lora_alpha": train_args.get("lora_alpha", 32),
        "lora_dropout": train_args.get("lora_dropout", 0.05),
        "lora_target_modules": train_args.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
        "use_prior_as_ot_target": train_args.get("use_prior_as_ot_target", True),
        "use_prior_align_loss": train_args.get("use_prior_align_loss", True),
        "use_global_topo_loss": train_args.get("use_global_topo_loss", True),
        "use_patch_topo_loss": train_args.get("use_patch_topo_loss", True),
    })
    allowed = {
        "llm_name_or_path", "vision_pretrained", "vision_backend", "freeze_vision_backbone", "freeze_llm",
        "max_question_length", "max_answer_length", "use_lora", "lora_r", "lora_alpha", "lora_dropout",
        "lora_target_modules", "ot_loss_weight", "use_ot", "use_ot_fusion", "ot_fusion_mode", "ot_fusion_dropout",
        "use_prior_as_ot_target", "prior_ot_global_mass", "use_topological_loss", "use_prior_align_loss",
        "use_global_topo_loss", "use_patch_topo_loss", "prior_loss_weight", "global_topo_loss_weight",
        "patch_topo_loss_weight",
    }
    build_kwargs = {k: v for k, v in config.items() if k in allowed}
    topo_mode = train_args.get("topo_mode", "all")
    topo_dim = 36 if topo_mode == "tda_only" else 47
    bottleneck_dim = int(train_args.get("bottleneck_dim", train_args.get("adapter_bottleneck_dim", 32)))
    last_n_layers = int(train_args.get("adapter_last_n_layers", 8))
    every_n_layers = int(train_args.get("adapter_every_n_layers", 0))
    print(f"Loading CATA: {checkpoint_path}")
    print(f"Runtime: topo_mode={topo_mode} topo_dim={topo_dim} vision_pretrained={build_kwargs.get('vision_pretrained')}")
    model = build_structural_generative_vqa(**build_kwargs).to(device)
    if hasattr(model, "tokenizer"):
        model.tokenizer.padding_side = "left"
    wrappers = _install_topo_adapters(model, topo_dim, bottleneck_dim, last_n_layers, every_n_layers)
    for wrapper in wrappers:
        wrapper.to(device)
    _patch_generate_with_topo(model, wrappers, topo_mode)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if any(w in k for w in ("lora_", "visual_projector", "ot_", "topo_adapter", "global_topo_head", "patch_topo_head"))]
    if critical:
        raise RuntimeError(f"Missing critical checkpoint keys: {critical[:20]}")
    if unexpected:
        print(f"Warning unexpected keys: {unexpected[:10]}")
    model.eval()
    return model


_STRUCTURAL_EXTRACTORS = None


def get_structural_extractors():
    global _STRUCTURAL_EXTRACTORS
    if _STRUCTURAL_EXTRACTORS is not None:
        return _STRUCTURAL_EXTRACTORS
    from src.topology.lesion_prior import LesionPriorExtractor
    from src.topology.tda_morphology import TopologicalExtractor

    prior = LesionPriorExtractor(image_size=DEFAULT_IMAGE_SIZE, grid_size=DEFAULT_GRID_SIZE, use_morphology=False)
    morph = TopologicalExtractor(grid_size=DEFAULT_GRID_SIZE, image_size=DEFAULT_IMAGE_SIZE)
    _STRUCTURAL_EXTRACTORS = (prior, morph)
    return _STRUCTURAL_EXTRACTORS


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    import cv2

    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    rgb = cv2.resize(rgb, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float().unsqueeze(0)


def extract_structural(image: Image.Image) -> Dict[str, torch.Tensor]:
    import cv2

    prior_extractor, morpho_extractor = get_structural_extractors()
    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    prior_out = prior_extractor.extract_prior(bgr)
    morph_out = morpho_extractor.extract_features(bgr)
    prior_mask = prior_out["prior_mask"].astype(np.float32)
    topo_features = morph_out["topo_features"].astype(np.float32)
    global_features = morph_out["global_features"].astype(np.float32)
    return {
        "prior_mask": torch.from_numpy(prior_mask).float().unsqueeze(0),
        "topo_features": torch.from_numpy(topo_features).float().unsqueeze(0),
        "global_features": torch.from_numpy(global_features).float().unsqueeze(0),
    }


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def create_heatmap(image: Image.Image, structural: Dict[str, torch.Tensor], out_png: Path, out_json: Path) -> Dict[str, Any]:
    import cv2

    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    small = cv2.resize(rgb, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    red_like = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)).astype(np.float32) * saturation
    specular = ((value > 0.86) & (saturation < 0.35)).astype(np.float32)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0

    prior = structural["prior_mask"].squeeze(0).cpu().numpy()
    topo = structural["topo_features"].squeeze(0).cpu().numpy()
    topo_sal = normalize_map(np.linalg.norm(topo, axis=-1))
    prior_up = cv2.resize(normalize_map(prior), DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
    topo_up = cv2.resize(topo_sal, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
    color_sal = normalize_map(0.65 * red_like + 0.35 * saturation)
    edge_sal = normalize_map(edges)
    artifact_penalty = 1.0 - 0.55 * cv2.GaussianBlur(specular, (0, 0), 2.0)
    heat = normalize_map((0.42 * prior_up + 0.28 * topo_up + 0.20 * color_sal + 0.10 * edge_sal) * artifact_penalty)

    heat_uint8 = np.uint8(np.clip(heat * 255.0, 0, 255))
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.58 * small + 0.42 * color)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_png, optimize=True)

    evidence = {
        "prior_mean": round(float(prior.mean()), 6),
        "prior_max": round(float(prior.max()), 6),
        "topo_saliency_mean": round(float(topo_sal.mean()), 6),
        "topo_saliency_max": round(float(topo_sal.max()), 6),
        "heatmap_mean": round(float(heat.mean()), 6),
        "heatmap_max": round(float(heat.max()), 6),
        "heatmap_concentration": round(float((heat > 0.65).mean()), 6),
        "redness_mean": round(float(red_like.mean()), 6),
        "saturation_mean": round(float(saturation.mean()), 6),
        "specular_fraction": round(float(specular.mean()), 6),
        "global_features": [round(float(x), 6) for x in structural["global_features"].squeeze(0).cpu().numpy().tolist()],
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    return evidence


def question_family(question: str) -> str:
    q = question.lower()
    for family, terms in QUESTION_FAMILIES.items():
        if any(term in q for term in terms):
            return family
    return "generic"


def is_negative(text: str) -> bool:
    t = " " + text.lower() + " "
    return any(term in t for term in NEGATIVE_TERMS)


def has_positive_evidence(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in POSITIVE_TERMS)


def build_explanation(question: str, answer: str, family: str, probes: List[str], probe_answers: List[str], evidence: Dict[str, Any]) -> Tuple[str, float]:
    probe_text = "; ".join(a.strip() for a in probe_answers if a.strip())
    neg_answer = is_negative(answer)
    pos_probe_count = sum(1 for a in probe_answers if has_positive_evidence(a) and not is_negative(a))
    neg_probe_count = sum(1 for a in probe_answers if is_negative(a))
    conflict = (neg_answer and pos_probe_count > 0) or ((not neg_answer) and neg_probe_count >= 2)

    heat = float(evidence.get("heatmap_mean", 0.0))
    concentration = float(evidence.get("heatmap_concentration", 0.0))
    prior = float(evidence.get("prior_mean", 0.0))
    spec = float(evidence.get("specular_fraction", 0.0))
    topo = float(evidence.get("topo_saliency_mean", 0.0))

    confidence = 0.50 + 0.18 * min(1.0, prior * 3.0) + 0.14 * min(1.0, topo * 2.5) + 0.10 * min(1.0, concentration * 8.0)
    if probe_answers:
        confidence += 0.08 if not conflict else -0.12
    if neg_answer:
        confidence -= 0.03
    confidence -= 0.10 * min(1.0, spec * 5.0)
    confidence = float(np.clip(confidence, 0.05, 0.95))

    evidence_bits = []
    if prior > 0.08:
        evidence_bits.append("lesion-prior support")
    if topo > 0.12:
        evidence_bits.append("morphology/TDA structural support")
    if concentration > 0.03:
        evidence_bits.append("a relatively focal heatmap region")
    else:
        evidence_bits.append("diffuse heatmap evidence")
    if spec > 0.03:
        evidence_bits.append("possible specular artifact burden")

    if conflict:
        probe_sentence = (
            f"Targeted self-probes for this {family} question produced mixed support: {probe_text}. "
            "Because these probes do not fully agree with the primary answer, the explanation is treated cautiously."
        )
        support_phrase = "mixed"
    else:
        probe_sentence = (
            f"Targeted self-probes for this {family} question supported the primary answer: {probe_text}."
            if probe_text else "No additional self-probe answers were available."
        )
        support_phrase = "moderate" if confidence >= 0.55 else "limited"

    explanation = (
        f"The CATA-Final model predicts \"{answer}\". "
        f"The newly generated visual explanation highlights {', '.join(evidence_bits)}. "
        f"{probe_sentence} "
        f"Overall support is {support_phrase}; the reliability score combines lesion prior, morphology/TDA saliency, "
        f"heatmap concentration, artifact burden, answer specificity, and self-probe agreement. "
        f"The original question was: {question}. "
        "This explanation is intended for clinician review and should be interpreted alongside the image rather than as a standalone diagnosis."
    )
    return explanation, round(confidence, 4)


def build_val_set():
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
    return (
        ds.filter(lambda x: x["complexity"] == 1)
        .shuffle(seed=42)
        .select(range(1500))
        .add_column("val_id", list(range(1500)))
        .remove_columns(["complexity", "answer", "original", "question_class"])
        .cast_column("image", HfImage())
    )


def batched(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def generate_batch(model, device, images: List[Image.Image], questions: List[str], max_new_tokens: int):
    tensors, priors, topos, globals_ = [], [], [], []
    structs = []
    for img in images:
        st = extract_structural(img)
        structs.append(st)
        tensors.append(pil_to_tensor(img).to(device))
        priors.append(st["prior_mask"].to(device))
        topos.append(st["topo_features"].to(device))
        globals_.append(st["global_features"].to(device))
    with torch.inference_mode():
        answers = model.generate(
            image=torch.cat(tensors, 0),
            prior_mask=torch.cat(priors, 0),
            topo_features=torch.cat(topos, 0),
            global_features=torch.cat(globals_, 0),
            question_text=questions,
            max_new_tokens=max_new_tokens,
        )
    return [str(a).strip() for a in answers], structs


def main():
    parser = argparse.ArgumentParser(description="Generate Task 2 CATA-Final explanations.")
    parser.add_argument(
        "--checkpoint",
        default=str(HF_REPO_DIR / "checkpoints" / "last.pt"),
        help=(
            "Path to the CATA-Final checkpoint. In the combined HF repo this "
            "defaults to ./checkpoints/last.pt. For local experiments you can "
            "pass outputs/<run>/checkpoints/last.pt explicitly."
        ),
    )
    parser.add_argument("--output-jsonl", default="submission_task2_cata_final.jsonl")
    parser.add_argument("--visual-dir", default="visuals")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--probe-max-new-tokens", type=int, default=40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite-visuals", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-debug",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Include internal probe/debug metadata in each JSONL row. Default is false for clean final submission.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_jsonl = Path(args.output_jsonl)
    if not output_jsonl.is_absolute():
        output_jsonl = SCRIPT_DIR / output_jsonl
    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = SCRIPT_DIR / visual_dir
    visual_dir.mkdir(parents=True, exist_ok=True)

    val_set = build_val_set()
    end = len(val_set) if args.limit <= 0 else min(len(val_set), args.start_index + args.limit)
    indices = list(range(args.start_index, end))

    model = load_cata_model(checkpoint, device)
    started = time.time()

    mode = "a" if args.start_index > 0 and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as f:
        for chunk in tqdm(list(batched(indices, args.batch_size)), desc="Generating Task 2"):
            examples = [val_set[i] for i in chunk]
            images = [e["image"] for e in examples]
            questions = [str(e["question"]) for e in examples]
            answers, structs = generate_batch(model, device, images, questions, args.max_new_tokens)

            for local_idx, global_idx in enumerate(chunk):
                example = examples[local_idx]
                img = images[local_idx]
                question = questions[local_idx]
                answer = answers[local_idx]
                family = question_family(question)
                probe_questions = PROBES.get(family, PROBES["generic"])
                probe_answers, _ = generate_batch(
                    model,
                    device,
                    [img] * len(probe_questions),
                    probe_questions,
                    args.probe_max_new_tokens,
                )
                stem = f"{global_idx:04d}"
                heatmap_path = visual_dir / f"{stem}_heatmap.png"
                evidence_path = visual_dir / f"{stem}_evidence.json"
                if args.overwrite_visuals or not heatmap_path.exists() or not evidence_path.exists():
                    evidence = create_heatmap(img, structs[local_idx], heatmap_path, evidence_path)
                else:
                    with evidence_path.open("r", encoding="utf-8") as ef:
                        evidence = json.load(ef)
                explanation, confidence = build_explanation(question, answer, family, probe_questions, probe_answers, evidence)
                row = {
                    "val_id": str(int(example["val_id"])),
                    "img_id": str(example["img_id"]),
                    "question": question,
                    "answer": answer,
                    "textual_explanation": explanation,
                    "visual_explanation": [
                        {
                            "type": "heatmap",
                            "data": str(heatmap_path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
                            "description": "Fresh CATA-Final heatmap combining lesion prior, morphology/TDA saliency, color/edge evidence, and artifact suppression.",
                        },
                        {
                            "type": "evidence_json",
                            "data": str(evidence_path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
                            "description": "Structured visual evidence and confidence inputs used by the explanation generator.",
                        },
                    ],
                    "confidence_score": confidence,
                }
                if args.include_debug:
                    row["debug"] = {
                        "model_alias": "CATA-Final",
                        "question_family": family,
                        "probe_questions": probe_questions,
                        "probe_answers": probe_answers,
                    }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    elapsed = time.time() - started
    print(f"✅ Generated {len(indices)} Task 2 rows: {output_jsonl}")
    print(f"✅ Visuals/evidence: {visual_dir}")
    print(f"Time: {elapsed:.1f}s | {elapsed / max(1, len(indices)):.2f}s/item")


if __name__ == "__main__":
    main()
