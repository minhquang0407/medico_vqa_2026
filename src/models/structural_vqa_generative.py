from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # PEFT is optional unless LoRA is enabled.
    LoraConfig = None
    get_peft_model = None

from src.alignment.sinkhorn_ot import SinkhornOTConfig, SinkhornOTRouter
from src.models.structural_vqa import StructuralVQAPrototype
from src.models.vision_encoder import StructuralVisionEncoder, build_structural_vision_encoder


@dataclass
class StructuralGenerativeVQAConfig:
    llm_name_or_path: str = "distilgpt2"
    vision_d_model: int = 768
    freeze_llm: bool = True
    ot_loss_weight: float = 0.05
    use_ot: bool = True
    use_ot_fusion: bool = True
    ot_fusion_mode: str = "prefix"
    ot_fusion_dropout: float = 0.10
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    use_prior_as_ot_target: bool = True
    prior_ot_global_mass: float = 0.05
    use_topological_loss: bool = True
    use_prior_align_loss: bool = True
    use_global_topo_loss: bool = True
    use_patch_topo_loss: bool = False
    prior_loss_weight: float = 0.05
    global_topo_loss_weight: float = 0.01
    patch_topo_loss_weight: float = 0.005
    topo_feature_dim: int = 12
    global_feature_dim: int = 8
    max_question_length: int = 128
    max_answer_length: int = 128
    dropout: float = 0.10

    def __post_init__(self):
        if not 0.0 <= self.prior_ot_global_mass < 1.0:
            raise ValueError("prior_ot_global_mass phải nằm trong [0, 1)")
        if self.ot_fusion_mode not in {"none", "prefix"}:
            raise ValueError("ot_fusion_mode hiện chỉ hỗ trợ 'none' hoặc 'prefix'")
        if self.use_lora and (LoraConfig is None or get_peft_model is None):
            raise ImportError("use_lora=True nhưng chưa cài package 'peft'. Hãy chạy: pip install peft")
        if self.lora_r <= 0:
            raise ValueError("lora_r phải > 0")


class VisualToLLMProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_context: torch.Tensor) -> torch.Tensor:
        return self.net(visual_context)


class StructuralGenerativeVQA(nn.Module):
    """
    Generative VQA model:

        StructuralVisionEncoder -> visual prefix embeddings -> causal LM.

    Nếu `use_ot_fusion=True`, Sinkhorn OT aligned visual features cũng được
    project thành prefix tokens và đưa trực tiếp vào LLM input.
    """

    def __init__(
        self,
        config: StructuralGenerativeVQAConfig,
        vision_encoder: StructuralVisionEncoder,
    ):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name_or_path)
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        llm_dim = self.llm.get_input_embeddings().embedding_dim

        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        if config.use_lora:
            target_modules = [m.strip() for m in config.lora_target_modules.split(",") if m.strip()]
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        self.visual_projector = VisualToLLMProjector(
            vision_dim=config.vision_d_model,
            llm_dim=llm_dim,
            dropout=config.dropout,
        )
        self.ot_visual_projector = VisualToLLMProjector(
            vision_dim=config.vision_d_model,
            llm_dim=llm_dim,
            dropout=config.ot_fusion_dropout,
        )
        self.ot_text_projector = nn.Linear(llm_dim, config.vision_d_model)
        self.ot_router = SinkhornOTRouter(SinkhornOTConfig())

        hidden = config.vision_d_model * 2
        self.global_topo_head = nn.Sequential(
            nn.LayerNorm(config.vision_d_model),
            nn.Linear(config.vision_d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.global_feature_dim),
        )
        self.patch_topo_head = nn.Sequential(
            nn.LayerNorm(config.vision_d_model),
            nn.Linear(config.vision_d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.topo_feature_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        prior_mask: torch.Tensor,
        topo_features: torch.Tensor,
        global_features: torch.Tensor,
        question_text: Sequence[str],
        answer_text: Optional[Sequence[str]] = None,
        return_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = image.device
        prior_mask = prior_mask.to(device=device, dtype=image.dtype)
        topo_features = topo_features.to(device=device, dtype=image.dtype)
        global_features = global_features.to(device=device, dtype=image.dtype)

        vision_out = self.vision_encoder(
            image=image,
            prior_mask=prior_mask,
            topo_features=topo_features,
            global_features=global_features,
            return_diagnostics=True,
        )
        visual_context = vision_out["visual_context"]

        prompt_text = [self._format_prompt(q, use_chat_template=True) for q in question_text]
        prompt = self.tokenizer(
            prompt_text,
            padding=True,
            truncation=True,
            max_length=self.config.max_question_length,
            return_tensors="pt",
        ).to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt.input_ids)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        ot_out = self._compute_ot(
            visual_context=visual_context,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt.attention_mask,
            prior_mask=prior_mask,
        )
        ot_cost = ot_out["ot_cost"] if ot_out is not None else visual_context.new_tensor(0.0)

        visual_prefix = self.visual_projector(visual_context).to(dtype=llm_dtype)
        prefix_parts = [visual_prefix]
        prefix_masks = [torch.ones(visual_prefix.shape[:2], device=device, dtype=prompt.attention_mask.dtype)]
        ot_aligned_prefix = None

        if self._use_ot_prefix(ot_out):
            ot_aligned_prefix = self.ot_visual_projector(ot_out["aligned_visual"]).to(dtype=llm_dtype)
            prefix_parts.append(ot_aligned_prefix)
            prefix_masks.append(prompt.attention_mask)

        prefix_embeds = torch.cat(prefix_parts, dim=1)
        prefix_attention = torch.cat(prefix_masks, dim=1)
        full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        full_attention = torch.cat([prefix_attention, prompt.attention_mask], dim=1)

        labels = None
        if answer_text is not None:
            answer = self.tokenizer(
                [str(a) + self.tokenizer.eos_token for a in answer_text],
                padding=True,
                truncation=True,
                max_length=self.config.max_answer_length,
                return_tensors="pt",
            ).to(device)
            answer_embeds = self.llm.get_input_embeddings()(answer.input_ids)
            full_embeds = torch.cat([full_embeds, answer_embeds], dim=1)
            full_attention = torch.cat([full_attention, answer.attention_mask], dim=1)

            ignore_prefix = torch.full(
                (image.shape[0], prefix_embeds.shape[1] + prompt.input_ids.shape[1]),
                -100,
                device=device,
                dtype=torch.long,
            )
            answer_labels = answer.input_ids.masked_fill(answer.attention_mask == 0, -100)
            labels = torch.cat([ignore_prefix, answer_labels], dim=1)

        llm_out = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attention,
            labels=labels,
            return_dict=True,
        )
        lm_loss = llm_out.loss if labels is not None else visual_context.new_tensor(0.0)

        topo_out = self.compute_topological_losses(
            vision_out=vision_out,
            ot_out=ot_out,
            prior_mask=prior_mask,
            topo_features=topo_features,
            global_features=global_features,
            attention_mask=prompt.attention_mask,
        )
        loss = lm_loss + self.config.ot_loss_weight * ot_cost + topo_out["topological_loss"] if labels is not None else None

        if not return_diagnostics:
            return {"loss": loss, "logits": llm_out.logits}
        out = {
            "loss": loss,
            "lm_loss": lm_loss,
            "ot_cost": ot_cost,
            "logits": llm_out.logits,
            "vision_out": vision_out,
            "visual_prefix_shape": torch.tensor(visual_prefix.shape, device=device),
            "ot_fusion_enabled": torch.tensor(self._use_ot_prefix(ot_out), device=device),
            **topo_out,
        }
        if ot_aligned_prefix is not None:
            out["ot_aligned_prefix_shape"] = torch.tensor(ot_aligned_prefix.shape, device=device)
        if ot_out is not None:
            out["ot_out"] = ot_out
        return out

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prior_mask: torch.Tensor,
        topo_features: torch.Tensor,
        global_features: torch.Tensor,
        question_text: Sequence[str],
        max_new_tokens: int = 64,
    ) -> Sequence[str]:
        self.eval()
        device = image.device
        prior_mask = prior_mask.to(device=device, dtype=image.dtype)
        topo_features = topo_features.to(device=device, dtype=image.dtype)
        global_features = global_features.to(device=device, dtype=image.dtype)

        vision_out = self.vision_encoder(
            image=image,
            prior_mask=prior_mask,
            topo_features=topo_features,
            global_features=global_features,
            return_diagnostics=True,
        )
        visual_context = vision_out["visual_context"]
        prompt = self.tokenizer(
            [self._format_prompt(q, use_chat_template=True) for q in question_text],
            padding=True,
            truncation=True,
            max_length=self.config.max_question_length,
            return_tensors="pt",
        ).to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt.input_ids)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        ot_out = self._compute_ot(
            visual_context=visual_context,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt.attention_mask,
            prior_mask=prior_mask,
        )
        visual_prefix = self.visual_projector(visual_context).to(dtype=llm_dtype)
        prefix_parts = [visual_prefix]
        prefix_masks = [torch.ones(visual_prefix.shape[:2], device=device, dtype=prompt.attention_mask.dtype)]
        if self._use_ot_prefix(ot_out):
            prefix_parts.append(self.ot_visual_projector(ot_out["aligned_visual"]).to(dtype=llm_dtype))
            prefix_masks.append(prompt.attention_mask)

        prefix_embeds = torch.cat(prefix_parts, dim=1)
        prefix_attention = torch.cat(prefix_masks, dim=1)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([prefix_attention, prompt.attention_mask], dim=1)

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._generation_eos_token_ids(),
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    def _compute_ot(
        self,
        visual_context: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prior_mask: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.use_ot:
            return None
        prompt_token_embeds = prompt_embeds.detach() if self.config.freeze_llm else prompt_embeds
        prompt_token_embeds = prompt_token_embeds.to(dtype=self.ot_text_projector.weight.dtype)
        text_tokens = self.ot_text_projector(prompt_token_embeds)
        visual_target_mass = None
        if self.config.use_prior_as_ot_target:
            visual_target_mass = StructuralVQAPrototype.build_prior_visual_target_mass(
                prior_mask=prior_mask,
                num_visual_tokens=visual_context.shape[1],
                global_mass=self.config.prior_ot_global_mass,
            )
        return self.ot_router(
            visual_tokens=visual_context,
            text_tokens=text_tokens,
            text_mask=prompt_attention_mask,
            visual_target_mass=visual_target_mass,
            return_diagnostics=True,
        )

    def _use_ot_prefix(self, ot_out: Optional[Dict[str, torch.Tensor]]) -> bool:
        return (
            self.config.use_ot
            and self.config.use_ot_fusion
            and self.config.ot_fusion_mode == "prefix"
            and ot_out is not None
        )

    def compute_topological_losses(
        self,
        vision_out: Dict[str, torch.Tensor],
        ot_out: Optional[Dict[str, torch.Tensor]],
        prior_mask: torch.Tensor,
        topo_features: torch.Tensor,
        global_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = prior_mask.new_tensor(0.0)
        prior_align_loss = zero
        global_topo_loss = zero
        patch_topo_loss = zero

        if self.config.use_topological_loss:
            if self.config.use_prior_align_loss and ot_out is not None:
                prior_align_loss = StructuralVQAPrototype.compute_prior_alignment_loss(
                    transport_plan=ot_out["transport_plan"],
                    prior_mask=prior_mask,
                    attention_mask=attention_mask,
                )
            if self.config.use_global_topo_loss and vision_out.get("global_token") is not None:
                global_pred = self.global_topo_head(vision_out["global_token"].squeeze(1))
                global_topo_loss = F.mse_loss(global_pred, global_features.float())
            if self.config.use_patch_topo_loss:
                patch_pred = self.patch_topo_head(vision_out["fused_tokens"])
                patch_target = topo_features.flatten(start_dim=1, end_dim=2).float()
                patch_topo_loss = F.mse_loss(patch_pred, patch_target)

        effective_prior_loss_weight = 0.0 if self.config.use_prior_as_ot_target else self.config.prior_loss_weight
        topological_loss = (
            effective_prior_loss_weight * prior_align_loss
            + self.config.global_topo_loss_weight * global_topo_loss
            + self.config.patch_topo_loss_weight * patch_topo_loss
        )
        return {
            "prior_align_loss": prior_align_loss,
            "global_topo_loss": global_topo_loss,
            "patch_topo_loss": patch_topo_loss,
            "topological_loss": topological_loss,
            "effective_prior_loss_weight": prior_mask.new_tensor(effective_prior_loss_weight),
        }

    def _generation_eos_token_ids(self):
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        return eos_ids or self.tokenizer.eos_token_id

    def _format_prompt(self, question: str, use_chat_template: bool = True) -> str:
        question = str(question or "").replace("<image>", "").strip()
        instruction = (
            "You are a medical visual question answering assistant. "
            "Answer using only the image evidence. "
            "Give one concise sentence and do not repeat the question."
        )
        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {instruction}\nUser: {question}\nAssistant:"


def build_structural_generative_vqa(
    llm_name_or_path: str = "distilgpt2",
    vision_pretrained: bool = False,
    vision_backend: str = "timm",
    freeze_vision_backbone: bool = True,
    freeze_llm: bool = True,
    **kwargs,
) -> StructuralGenerativeVQA:
    config = StructuralGenerativeVQAConfig(
        llm_name_or_path=llm_name_or_path,
        freeze_llm=freeze_llm,
        **kwargs,
    )
    vision_encoder = build_structural_vision_encoder(
        d_model=config.vision_d_model,
        pretrained=vision_pretrained,
        backend=vision_backend,
        freeze_backbone=freeze_vision_backbone,
    )
    return StructuralGenerativeVQA(config=config, vision_encoder=vision_encoder)
