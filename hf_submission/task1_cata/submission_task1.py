from transformers import AutoModelForCausalLM
from datasets import load_dataset, Image as HfImage
from transformers import AutoProcessor
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys
import os
import random
from PIL import Image
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    from itertools import batched
except ImportError:
    def batched(iterable, n):
        import itertools
        iterator = iter(iterable)
        while batch := list(itertools.islice(iterator, n)):
            yield batch

from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")


ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
ds_shuffled = ds.shuffle(seed=42) # Shuffle with fixed seed for reproducibility
val_dataset = ds_shuffled.select(range(1500)) # Select first 1500 after shuffle
val_dataset = val_dataset.cast_column("image", HfImage())
predictions = []  # List to store predictions

gpu_name = torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mem(): return torch.cuda.memory_allocated(device) / \
    (1024 ** 2) if torch.cuda.is_available() else 0


initial_mem = get_mem()

# ✏️✏️--------EDIT SECTION 1: SUBMISISON DETAILS and MODEL LOADING --------✏️✏️#

SUBMISSION_INFO = {
    "Participant_Names": "Minh Quang Nguyen",
    "Affiliations": "Independent",
    "Contact_emails": ["nmquang04072005@gmail.com"],
    "Team_Name": "Sweet&Sour",
    "Country": "Vietnam",
    "Notes_to_organizers": '''
        Task 1 submission using Qwen2.5-3B-Instruct + QLoRA r16 + Curriculum-Gated full structural Deep TopoAdapter.
        Trained for 2 epochs on 30k with warmup/gate regularization then 1 epoch full-data continuation.
        Uses exact batched required template structure.
        '''
}

# --- Deep TopoAdapter runtime injection and loading logic ---
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_DIR = Path(__file__).resolve().parent
HF_REPO_ID = "minhquang47/medico2026-task1-cata-qwen3b"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _ensure_full_repo_available() -> Path:
    if (REPO_DIR / "src").exists() and (REPO_DIR / "checkpoints").exists():
        return REPO_DIR

    try:
        from huggingface_hub import snapshot_download

        full_repo = Path(
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                local_files_only=False,
            )
        )
        if str(full_repo) not in sys.path:
            sys.path.insert(0, str(full_repo))
        return full_repo
    except Exception as exc:
        print("Warning: could not download full repo snapshot:", repr(exc))
        return REPO_DIR


RUNTIME_DIR = _ensure_full_repo_available()
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))


try:
    from src.models.structural_vqa_generative import build_structural_generative_vqa
except Exception as exc:
    print("Warning: model import failed lazily:", repr(exc))
    build_structural_generative_vqa = None


CHECKPOINT_CANDIDATES = [
    RUNTIME_DIR / "checkpoints" / "final.pt",
    RUNTIME_DIR / "checkpoints" / "epoch_3.pt",
    RUNTIME_DIR / "checkpoints" / "epoch_4.pt",
    RUNTIME_DIR / "checkpoints" / "last.pt",
    RUNTIME_DIR / "model.pt",
    REPO_DIR / "checkpoints" / "final.pt",
    REPO_DIR / "model.pt",
]

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_TOPO_FEATURE_DIM = 12
DEFAULT_GLOBAL_FEATURE_DIM = 8


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
        topo_condition = topo_condition.to(
            device=hidden_states.device,
            dtype=self.topo_projector[0].weight.dtype,
        )
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
            hs = self.topo_adapter(out[0], self.topo_condition)
            return (hs,) + out[1:]
        return self.topo_adapter(out, self.topo_condition)


def _find_decoder_layers(llm) -> torch.nn.ModuleList:
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
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Cannot locate Qwen decoder layers for TopoAdapter injection")


def _install_topo_adapters(
    model,
    topo_dim: int = 47,
    bottleneck_dim: int = 32,
    last_n_layers: int = 8,
    every_n_layers: int = 0,
) -> List[DecoderLayerWithTopoAdapter]:
    layers = _find_decoder_layers(model.llm)
    hidden_dim = int(model.llm.get_input_embeddings().embedding_dim)
    n_layers = len(layers)
    selected = set(range(max(0, n_layers - last_n_layers), n_layers)) if last_n_layers > 0 else set()
    if every_n_layers > 0:
        selected.update(range(0, n_layers, every_n_layers))
    wrappers: List[DecoderLayerWithTopoAdapter] = []
    for idx in sorted(selected):
        if isinstance(layers[idx], DecoderLayerWithTopoAdapter):
            wrappers.append(layers[idx])
            continue
        wrapper = DecoderLayerWithTopoAdapter(
            layers[idx],
            TopoAdapter(hidden_dim, topo_dim, bottleneck_dim),
        )
        layers[idx] = wrapper
        wrappers.append(wrapper)
    print(
        f"Installed {len(wrappers)} TopoAdapters / {n_layers} decoder layers "
        f"| hidden={hidden_dim} topo_dim={topo_dim}"
    )
    return wrappers


def _topo_condition(
    prior_mask: torch.Tensor,
    topo_features: torch.Tensor,
    global_features: torch.Tensor,
    mode: str = "all",
) -> torch.Tensor:
    # Perform calculations in float32 for maximum numerical safety
    tf = topo_features.float()
    flat = tf.flatten(1, 2)
    parts = [flat.mean(1), flat.std(1), flat.amax(1)]
    if mode == "all":
        pm = prior_mask.float().flatten(1)
        parts.extend([
            pm.mean(1, keepdim=True),
            pm.std(1, keepdim=True),
            pm.amax(1, keepdim=True),
        ])
        parts.append(global_features.float())
    cond = torch.cat(parts, dim=-1)
    return torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)


def _set_topo_condition(wrappers: List[DecoderLayerWithTopoAdapter], condition: Optional[torch.Tensor]):
    for wrapper in wrappers:
        wrapper.set_topo_condition(condition)


def _patch_model_generate_with_topo(model, wrappers: List[DecoderLayerWithTopoAdapter], mode: str = "all"):
    original_generate = model.generate

    @torch.no_grad()
    def generate_with_topo(
        self,
        image,
        prior_mask,
        topo_features,
        global_features,
        question_text,
        max_new_tokens=64,
    ):
        condition = _topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        _set_topo_condition(wrappers, condition)
        try:
            return original_generate(
                image,
                prior_mask,
                topo_features,
                global_features,
                question_text,
                max_new_tokens,
            )
        finally:
            _set_topo_condition(wrappers, None)

    import types

    model.generate = types.MethodType(generate_with_topo, model)


def _find_checkpoint() -> Path:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found. Expected one of: "
        + ", ".join(str(path.relative_to(REPO_DIR)) for path in CHECKPOINT_CANDIDATES)
    )


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            return loaded
        return {
            "model_state_dict": loaded,
            "args": {
                "llm_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                "vision_backend": "timm",
                "freeze_llm": True,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_modules": "q_proj,v_proj",
                "use_ot": True,
                "use_ot_fusion": True,
                "ot_fusion_mode": "prefix",
                "ot_fusion_dropout": 0.10,
                "use_prior_as_ot_target": True,
                "use_topological_loss": True,
                "use_prior_align_loss": True,
                "use_global_topo_loss": True,
                "use_patch_topo_loss": True,
                "topo_mode": "all",
                "vision_pretrained": True,
                "freeze_vision_backbone": True,
            },
        }


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    if build_structural_generative_vqa is None:
        raise ImportError(
            "Could not import src.models.structural_vqa_generative. "
            "Copy the project src/ directory into the HF repo root."
        )

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = _safe_load_checkpoint(checkpoint_path)
    train_args = checkpoint.get("args", {})
    config = {**train_args}
    config.update(
        {
            "llm_name_or_path": train_args.get("llm_name_or_path", "Qwen/Qwen2.5-3B-Instruct"),
            "vision_pretrained": train_args.get("vision_pretrained", True),
            "vision_backend": train_args.get("vision_backend", "timm"),
            "freeze_vision_backbone": train_args.get("freeze_vision_backbone", True),
            "freeze_llm": train_args.get("freeze_llm", True),
            "use_lora": train_args.get("use_lora", True),
            "lora_r": train_args.get("lora_r", 16),
            "lora_alpha": train_args.get("lora_alpha", 32),
            "lora_dropout": train_args.get("lora_dropout", 0.05),
            "lora_target_modules": train_args.get(
                "lora_target_modules",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            ),
            "use_prior_as_ot_target": train_args.get("use_prior_as_ot_target", True),
            "use_prior_align_loss": train_args.get("use_prior_align_loss", True),
            "use_global_topo_loss": train_args.get("use_global_topo_loss", True),
            "use_patch_topo_loss": train_args.get("use_patch_topo_loss", True),
        }
    )
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
    }
    build_kwargs = {key: value for key, value in config.items() if key in allowed}
    topo_mode = train_args.get("topo_mode", "all")
    topo_dim = 36 if topo_mode == "tda_only" else 47
    bottleneck_dim = int(train_args.get("bottleneck_dim", train_args.get("adapter_bottleneck_dim", 32)))
    last_n_layers = int(train_args.get("adapter_last_n_layers", 8))
    every_n_layers = int(train_args.get("adapter_every_n_layers", 0))
    print(
        "Runtime config: "
        f"topo_mode={topo_mode} topo_dim={topo_dim} "
        f"vision_pretrained={build_kwargs.get('vision_pretrained')} "
        f"use_patch_topo_loss={build_kwargs.get('use_patch_topo_loss')}"
    )
    model = build_structural_generative_vqa(**build_kwargs).to(device)
    model.tokenizer.padding_side = "left"  # Set padding side to left for batch generation safety
    wrappers = _install_topo_adapters(
        model,
        topo_dim=topo_dim,
        bottleneck_dim=bottleneck_dim,
        last_n_layers=last_n_layers,
        every_n_layers=every_n_layers,
    )
    for wrapper in wrappers:
        wrapper.to(device)
    _patch_model_generate_with_topo(model, wrappers, mode=topo_mode)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_keywords = (
        "lora_",
        "visual_projector",
        "ot_visual_projector",
        "ot_text_projector",
        "topo_adapter",
        "global_topo_head",
        "patch_topo_head",
    )
    critical_missing = [key for key in missing if any(word in key for word in critical_keywords)]
    unexpected_topo = [key for key in unexpected if "topo_adapter" in key]
    if any("topo_adapter" in key for key in critical_missing) and unexpected_topo:
        remapped_state = dict(state)
        prefix_pairs = [
            ("llm.base_model.model.model.layers.", "llm.model.layers."),
            ("llm.model.layers.", "llm.base_model.model.model.layers."),
        ]
        for old_prefix, new_prefix in prefix_pairs:
            for key, value in state.items():
                if key.startswith(old_prefix):
                    remapped_state[new_prefix + key[len(old_prefix):]] = value
        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        critical_missing = [key for key in missing if any(word in key for word in critical_keywords)]

    # Detailed trainable validation
    trainable_missing = [
        name for name, param in model.named_parameters()
        if param.requires_grad and name in missing
    ]
    
    print("\n" + "="*50)
    print("MODEL LOAD DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Total missing keys: {len(missing)}")
    print(f"Total unexpected keys: {len(unexpected)}")
    print(f"Trainable missing keys: {len(trainable_missing)}")
    if trainable_missing:
        print(f"Warning: Missing trainable keys: {trainable_missing[:15]}")
    if critical_missing:
        print(f"Error: Critical missing fine-tuned keys: {critical_missing[:15]}")
    print("="*50 + "\n")

    if trainable_missing:
        raise RuntimeError(
            f"Checkpoint is missing trainable weights required for fine-tuning/inference! "
            f"Missing {len(trainable_missing)} keys, including: {trainable_missing[:15]}"
        )
    if critical_missing:
        raise RuntimeError(f"Missing critical fine-tuned checkpoint keys: {critical_missing[:20]}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {unexpected[:20]}")
    print(f"Loaded checkpoint successfully. Status: OK")
    model.eval()
    model.llm.eval()
    model.vision_encoder.eval()
    return model


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    import cv2

    image_rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    image_rgb = cv2.resize(image_rgb, DEFAULT_IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    arr = image_rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float().unsqueeze(0)


_STRUCTURAL_EXTRACTORS = None


def _get_structural_extractors():
    global _STRUCTURAL_EXTRACTORS
    if _STRUCTURAL_EXTRACTORS is not None:
        return _STRUCTURAL_EXTRACTORS

    from src.topology.lesion_prior import LesionPriorExtractor
    from src.topology.tda_morphology import TopologicalExtractor as MorphologyTopologicalExtractor

    prior_extractor = LesionPriorExtractor(
        image_size=DEFAULT_IMAGE_SIZE,
        grid_size=DEFAULT_GRID_SIZE,
        use_morphology=False,
    )
    morpho_extractor = MorphologyTopologicalExtractor(
        grid_size=DEFAULT_GRID_SIZE,
        image_size=DEFAULT_IMAGE_SIZE,
    )
    _STRUCTURAL_EXTRACTORS = (prior_extractor, morpho_extractor)
    return _STRUCTURAL_EXTRACTORS


def _extract_online_structural_tensors(image: Image.Image) -> Dict[str, torch.Tensor]:
    import cv2

    prior_extractor, morpho_extractor = _get_structural_extractors()

    image_rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    prior_output = prior_extractor.extract_prior(image_bgr)
    morpho_output = morpho_extractor.extract_features(image_bgr)

    prior_mask = prior_output["prior_mask"].astype(np.float32)
    topo_features = morpho_output["topo_features"].astype(np.float32)
    global_features = morpho_output["global_features"].astype(np.float32)

    if prior_mask.shape != DEFAULT_GRID_SIZE:
        raise ValueError(f"prior_mask shape {prior_mask.shape} != {DEFAULT_GRID_SIZE}")
    expected_topo_shape = (*DEFAULT_GRID_SIZE, DEFAULT_TOPO_FEATURE_DIM)
    if topo_features.shape != expected_topo_shape:
        raise ValueError(f"topo_features shape {topo_features.shape} != {expected_topo_shape}")
    if global_features.shape != (DEFAULT_GLOBAL_FEATURE_DIM,):
        raise ValueError(
            f"global_features shape {global_features.shape} != {(DEFAULT_GLOBAL_FEATURE_DIM,)}"
        )

    return {
        "prior_mask": torch.from_numpy(prior_mask).float().unsqueeze(0),
        "topo_features": torch.from_numpy(topo_features).float().unsqueeze(0),
        "global_features": torch.from_numpy(global_features).float().unsqueeze(0),
    }


def predict_one(image: Image.Image, question: str, max_new_tokens: int = 48) -> str:
    model = load_model()
    device = _DEVICE or _select_device()
    image_tensor = _pil_to_tensor(image).to(device)
    structural = {key: value.to(device) for key, value in _extract_online_structural_tensors(image).items()}

    with torch.inference_mode():
        prediction = model.generate(
            image=image_tensor,
            prior_mask=structural["prior_mask"],
            topo_features=structural["topo_features"],
            global_features=structural["global_features"],
            question_text=[question],
            max_new_tokens=max_new_tokens,
        )[0]
    return prediction.strip()


def predict(batch: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs = []
    for record in batch:
        image = record.get("image")
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise TypeError("record['image'] must be a PIL.Image.Image or image path")
        question = str(record.get("question", ""))
        answer = predict_one(image, question)
        out = {key: record[key] for key in ("id", "img_id", "question") if key in record}
        out["answer"] = answer
        outputs.append(out)
    return outputs


class MedicoTask1Model:
    def __call__(self, image: Image.Image, question: str) -> str:
        return predict_one(image, question)

    def predict(self, image: Image.Image, question: str) -> str:
        return predict_one(image, question)


def load():
    load_model()
    return MedicoTask1Model()


model_hf = _load_model_from_checkpoint(_find_checkpoint(), device)
processor = None
BATCH_SIZE = 32

# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

with tqdm(total=len(val_dataset), desc="Validating", unit="samples") as pbar:
  for batch in batched(enumerate(val_dataset), BATCH_SIZE):
    pbar.update(len(batch))
    idxs, exs = zip(*batch)
    
    # ✏️✏️___________EDIT SECTION 2: ANSWER GENERATION___________✏️✏️#

    batch_images = []
    batch_prior_masks = []
    batch_topo_features = []
    batch_global_features = []
    batch_questions = []

    for e in exs:
        img = e["image"]
        q = e["question"]

        img_tensor = _pil_to_tensor(img).to(device)
        batch_images.append(img_tensor)

        struct = _extract_online_structural_tensors(img)
        batch_prior_masks.append(struct["prior_mask"].to(device))
        batch_topo_features.append(struct["topo_features"].to(device))
        batch_global_features.append(struct["global_features"].to(device))

        batch_questions.append(q)

    batch_img_tensor = torch.cat(batch_images, dim=0)
    batch_pm_tensor = torch.cat(batch_prior_masks, dim=0)
    batch_tf_tensor = torch.cat(batch_topo_features, dim=0)
    batch_gf_tensor = torch.cat(batch_global_features, dim=0)

    with torch.inference_mode():
        answers = model_hf.generate(
            image=batch_img_tensor,
            prior_mask=batch_pm_tensor,
            topo_features=batch_tf_tensor,
            global_features=batch_gf_tensor,
            question_text=batch_questions,
            max_new_tokens=48,
        )

    answers = [a.strip() for a in answers]

    #   ___________END SECTION 2: ANSWER GENERATION___________    #

# ⛔ DO NOT EDIT any lines below from here, can edit only upto decoding step above as required. ⛔

    assert all(isinstance(a, str) for a in answers), next(f"Non-string answer at index {i}" 
         for i, a in zip(idxs, answers)
         if not isinstance(a, str)) # safe test

    predictions.extend(
        {"index": i, "img_id": e["img_id"], "question": e["question"], "answer": a.strip()}
        for i, e, a in zip(idxs, exs, answers)
    )

# Ensure all predictions match dataset length
assert len(predictions) == len(
    val_dataset), "Mismatch between predictions and dataset length"

total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

# caulcualtes metrics
references = [[e] for e in val_dataset['answer']]
preds = [pred['answer'] for pred in predictions]

bleu_result = bleu.compute(predictions=preds, references=references)
rouge_result = rouge.compute(predictions=preds, references=references)
meteor_result = meteor.compute(predictions=preds, references=references)
bleu_score = round(bleu_result['bleu'], 4)
rouge1_score = round(float(rouge_result['rouge1']), 4)
rouge2_score = round(float(rouge_result['rouge2']), 4)
rougeL_score = round(float(rouge_result['rougeL']), 4)
meteor_score = round(float(meteor_result['meteor']), 4)

public_scores = {
    'bleu': bleu_score,
    'rouge1': rouge1_score,
    'rouge2': rouge2_score,
    'rougeL': rougeL_score,
    'meteor': meteor_score
}
print("✨Public scores: ", public_scores)

# Saves predictions to a JSON file

output_data = {"submission_info": SUBMISSION_INFO, "public_scores": public_scores,
               "predictions": predictions, "total_time": total_time, "time_per_item": total_time / len(val_dataset),
               "memory_used_mb": final_mem, "model_memory_mb": model_mem_used, "gpu_name": gpu_name,
               "debug": {
                   "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                   "system": {
                       "python": platform.python_version(),
                       "os": platform.system(),
                       "platform": platform.platform(),
                       "arch": platform.machine()
                   }}}


with open("predictions_1.json", "w") as f:
    json.dump(output_data, f, indent=4)
print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Scripts Looks Good! Generation process completed successfully. Results saved to 'predictions_1.json'.")
print("Next Step:\n 1) Upload this submission_task1.py script file to HuggingFace model repository.")
print('''\n 2) Make a submission to the competition:\n Run:: medvqa validate_and_submit --competition=medico-2026 --task=1 --repo_id=...''')
