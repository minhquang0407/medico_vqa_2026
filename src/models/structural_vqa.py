from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.sinkhorn_ot import SinkhornOTConfig, SinkhornOTRouter
from src.models.text_encoder import BasicVQATokenizer, SimpleTextEncoder, SimpleTextEncoderConfig
from src.models.vision_encoder import StructuralVisionEncoder, build_structural_vision_encoder


class AnswerVocabulary:
    """
    Map answer text <-> class id cho prototype VQA classification.

    Đây là baseline classification, không phải generative decoder.
    """

    def __init__(self, answers: Optional[Sequence[str]] = None, min_freq: int = 1, max_answers: Optional[int] = None):
        self.answer_to_id: Dict[str, int] = {}
        self.id_to_answer: Dict[int, str] = {}
        if answers is not None:
            self.build(answers, min_freq=min_freq, max_answers=max_answers)

    def __len__(self) -> int:
        return len(self.answer_to_id)

    def build(self, answers: Sequence[str], min_freq: int = 1, max_answers: Optional[int] = None) -> None:
        from collections import Counter

        counter = Counter(self.normalize(answer) for answer in answers)
        self.answer_to_id.clear()
        self.id_to_answer.clear()

        for answer, freq in counter.most_common(max_answers):
            if freq < min_freq:
                continue
            idx = len(self.answer_to_id)
            self.answer_to_id[answer] = idx
            self.id_to_answer[idx] = answer

    @staticmethod
    def normalize(answer: str) -> str:
        return " ".join(str(answer or "").strip().lower().split())

    def encode(self, answer: str) -> int:
        key = self.normalize(answer)
        if key not in self.answer_to_id:
            raise KeyError(f"Answer ngoài vocabulary: {answer}")
        return self.answer_to_id[key]

    def batch_encode(self, answers: Sequence[str]) -> torch.Tensor:
        return torch.tensor([self.encode(answer) for answer in answers], dtype=torch.long)

    def decode(self, answer_id: int) -> str:
        return self.id_to_answer[int(answer_id)]


@dataclass
class StructuralVQAConfig:
    d_model: int = 768
    num_answers: int = 1000
    text_max_length: int = 64
    dropout: float = 0.10
    ot_loss_weight: float = 0.01
    use_ot: bool = True
    use_topological_loss: bool = True
    use_prior_align_loss: bool = True
    use_global_topo_loss: bool = True
    use_patch_topo_loss: bool = False
    prior_loss_weight: float = 0.05
    global_topo_loss_weight: float = 0.01
    patch_topo_loss_weight: float = 0.005
    topo_feature_dim: int = 12
    global_feature_dim: int = 8
    use_prior_as_ot_target: bool = True
    prior_ot_global_mass: float = 0.05
    classifier_hidden_dim: Optional[int] = None

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError("d_model phải > 0")
        if self.num_answers <= 0:
            raise ValueError("num_answers phải > 0")
        if self.topo_feature_dim <= 0:
            raise ValueError("topo_feature_dim phải > 0")
        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim phải > 0")
        if not 0.0 <= self.prior_ot_global_mass < 1.0:
            raise ValueError("prior_ot_global_mass phải nằm trong [0, 1)")
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model * 2


class StructuralVQAPrototype(nn.Module):
    """
    Prototype VQA classification model:

        image + structural features -> StructuralVisionEncoder -> visual_context
        question_text/input_ids -> SimpleTextEncoder -> text_tokens
        text_tokens + visual_context -> SinkhornOTRouter -> aligned_visual
        pooled multimodal representation -> answer logits

    Topological losses optional:
        - prior alignment: OT transport should align with lesion prior.
        - global topology reconstruction: global token should reconstruct global features.
        - patch topology reconstruction: fused tokens should reconstruct patch topo features.
    """

    def __init__(
        self,
        config: StructuralVQAConfig,
        vision_encoder: StructuralVisionEncoder,
        text_encoder: SimpleTextEncoder,
        tokenizer: Optional[BasicVQATokenizer] = None,
        ot_router: Optional[SinkhornOTRouter] = None,
    ):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.ot_router = ot_router or SinkhornOTRouter(SinkhornOTConfig())

        fusion_dim = config.d_model * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_answers),
        )

        self.global_topo_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.global_feature_dim),
        )
        self.patch_topo_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.topo_feature_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        prior_mask: torch.Tensor,
        topo_features: torch.Tensor,
        global_features: torch.Tensor,
        question_text: Optional[Sequence[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ):
        device = image.device

        if input_ids is None:
            if question_text is None:
                raise ValueError("Cần truyền question_text hoặc input_ids")
            if self.tokenizer is None:
                raise ValueError("question_text được truyền nhưng model.tokenizer=None")
            tokenized = self.tokenizer.batch_encode(
                question_text,
                max_length=self.config.text_max_length,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

        input_ids = input_ids.to(device=device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
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

        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_tokens = text_out["text_tokens"]
        pooled_text = text_out["pooled_text"]
        attention_mask = text_out["attention_mask"]

        if self.config.use_ot:
            visual_target_mass = None
            if self.config.use_prior_as_ot_target:
                visual_target_mass = self.build_prior_visual_target_mass(
                    prior_mask=prior_mask,
                    num_visual_tokens=visual_context.shape[1],
                    global_mass=self.config.prior_ot_global_mass,
                )

            ot_out = self.ot_router(
                visual_tokens=visual_context,
                text_tokens=text_tokens,
                text_mask=attention_mask,
                visual_target_mass=visual_target_mass,
                return_diagnostics=True,
            )
            aligned_visual_tokens = ot_out["aligned_visual"]
            pooled_aligned_visual = SimpleTextEncoder.masked_mean_pool(
                aligned_visual_tokens,
                attention_mask,
            )
            ot_cost = ot_out["ot_cost"]
        else:
            ot_out = None
            pooled_aligned_visual = visual_context.mean(dim=1)
            ot_cost = visual_context.new_tensor(0.0)

        pooled_visual = visual_context.mean(dim=1)
        multimodal = torch.cat(
            [
                pooled_text,
                pooled_visual,
                pooled_aligned_visual,
                pooled_text * pooled_aligned_visual,
            ],
            dim=-1,
        )
        logits = self.classifier(multimodal)

        topo_loss_out = self.compute_topological_losses(
            vision_out=vision_out,
            ot_out=ot_out,
            prior_mask=prior_mask,
            topo_features=topo_features,
            global_features=global_features,
            attention_mask=attention_mask,
        )

        loss = None
        ce_loss = None
        if answer_ids is not None:
            answer_ids = answer_ids.to(device=device)
            ce_loss = F.cross_entropy(logits, answer_ids)
            loss = ce_loss + self.config.ot_loss_weight * ot_cost + topo_loss_out["topological_loss"]

        if not return_diagnostics:
            return logits

        output = {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss,
            "ot_cost": ot_cost,
            "visual_context": visual_context,
            "text_tokens": text_tokens,
            "pooled_text": pooled_text,
            "pooled_visual": pooled_visual,
            "pooled_aligned_visual": pooled_aligned_visual,
            "vision_out": vision_out,
            **topo_loss_out,
        }
        if ot_out is not None:
            output["ot_out"] = ot_out
        return output

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
        global_topo_pred = None
        patch_topo_pred = None

        if self.config.use_topological_loss:
            if self.config.use_prior_align_loss and ot_out is not None:
                prior_align_loss = self.compute_prior_alignment_loss(
                    transport_plan=ot_out["transport_plan"],
                    prior_mask=prior_mask,
                    attention_mask=attention_mask,
                )

            if self.config.use_global_topo_loss:
                global_token = vision_out.get("global_token")
                if global_token is not None:
                    global_topo_pred = self.global_topo_head(global_token.squeeze(1))
                    global_topo_loss = F.mse_loss(global_topo_pred, global_features.float())

            if self.config.use_patch_topo_loss:
                fused_tokens = vision_out["fused_tokens"]
                patch_topo_pred = self.patch_topo_head(fused_tokens)
                topo_target = topo_features.flatten(start_dim=1, end_dim=2).float()
                patch_topo_loss = F.mse_loss(patch_topo_pred, topo_target)

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
            "global_topo_pred": global_topo_pred,
            "patch_topo_pred": patch_topo_pred,
        }

    @staticmethod
    def compute_prior_alignment_loss(
        transport_plan: torch.Tensor,
        prior_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if transport_plan.shape[-1] < 2:
            return transport_plan.new_tensor(0.0)

        patch_transport = transport_plan[:, :, 1:]
        text_mask = attention_mask.to(device=transport_plan.device, dtype=transport_plan.dtype).unsqueeze(-1)
        patch_attention = (patch_transport * text_mask).sum(dim=1)

        prior = prior_mask.flatten(start_dim=1).to(device=transport_plan.device, dtype=transport_plan.dtype)
        if prior.shape[1] != patch_attention.shape[1]:
            raise ValueError(
                f"prior patches={prior.shape[1]} không khớp transport patches={patch_attention.shape[1]}"
            )

        patch_attention = patch_attention / patch_attention.sum(dim=1, keepdim=True).clamp_min(eps)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(eps)
        cosine = F.cosine_similarity(patch_attention, prior, dim=1, eps=eps)
        return (1.0 - cosine).mean()

    @staticmethod
    def build_prior_visual_target_mass(
        prior_mask: torch.Tensor,
        num_visual_tokens: int,
        global_mass: float = 0.05,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Tạo target marginal cho visual side của Sinkhorn từ lesion prior.

        visual_context có dạng:
            token 0      = global structural token
            token 1..196 = patch tokens

        Nếu Sinkhorn dùng target marginal uniform, tổng mass trên patch tokens
        bị ép gần đều nên prior_align_loss gần như không thể giảm. Hàm này
        đổi target marginal thành:
            global token mass = global_mass
            patch token mass  = (1 - global_mass) * normalized prior_mask
        """
        if num_visual_tokens < 2:
            raise ValueError("Cần ít nhất 1 global token + patch tokens để dùng prior target mass")

        prior = prior_mask.flatten(start_dim=1)
        expected_patches = num_visual_tokens - 1
        if prior.shape[1] != expected_patches:
            raise ValueError(
                f"prior patches={prior.shape[1]} không khớp visual patch tokens={expected_patches}"
            )

        prior = prior.clamp_min(0.0)
        fallback = torch.ones_like(prior) / max(expected_patches, 1)
        prior_sum = prior.sum(dim=1, keepdim=True)
        prior_dist = torch.where(prior_sum > eps, prior / prior_sum.clamp_min(eps), fallback)

        global_column = prior.new_full((prior.shape[0], 1), global_mass)
        patch_mass = (1.0 - global_mass) * prior_dist
        return torch.cat([global_column, patch_mass], dim=1)

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> torch.Tensor:
        self.eval()
        out = self.forward(*args, **kwargs)
        logits = out["logits"] if isinstance(out, dict) else out
        return logits.argmax(dim=-1)


def build_structural_vqa_prototype(
    questions: Sequence[str],
    answers: Sequence[str],
    d_model: int = 768,
    num_answers: Optional[int] = None,
    text_max_length: int = 64,
    vision_pretrained: bool = False,
    vision_backend: str = "timm",
    freeze_vision_backbone: bool = True,
    max_text_vocab_size: int = 30000,
    max_answer_vocab_size: Optional[int] = None,
    ot_loss_weight: float = 0.01,
    use_topological_loss: bool = True,
    use_prior_align_loss: bool = True,
    use_global_topo_loss: bool = True,
    use_patch_topo_loss: bool = False,
    prior_loss_weight: float = 0.05,
    global_topo_loss_weight: float = 0.01,
    patch_topo_loss_weight: float = 0.005,
    use_prior_as_ot_target: bool = True,
    prior_ot_global_mass: float = 0.05,
) -> Tuple[StructuralVQAPrototype, BasicVQATokenizer, AnswerVocabulary]:
    tokenizer = BasicVQATokenizer.build_from_texts(
        questions,
        max_vocab_size=max_text_vocab_size,
    )
    answer_vocab = AnswerVocabulary(
        answers,
        max_answers=max_answer_vocab_size,
    )

    if num_answers is None:
        num_answers = len(answer_vocab)
    if num_answers <= 0:
        raise ValueError("Không build được answer vocabulary; num_answers=0")

    vision_encoder = build_structural_vision_encoder(
        d_model=d_model,
        pretrained=vision_pretrained,
        backend=vision_backend,
        freeze_backbone=freeze_vision_backbone,
    )

    text_config = SimpleTextEncoderConfig(
        vocab_size=len(tokenizer),
        d_model=d_model,
        max_length=text_max_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    text_encoder = SimpleTextEncoder(text_config)

    config = StructuralVQAConfig(
        d_model=d_model,
        num_answers=num_answers,
        text_max_length=text_max_length,
        ot_loss_weight=ot_loss_weight,
        use_topological_loss=use_topological_loss,
        use_prior_align_loss=use_prior_align_loss,
        use_global_topo_loss=use_global_topo_loss,
        use_patch_topo_loss=use_patch_topo_loss,
        prior_loss_weight=prior_loss_weight,
        global_topo_loss_weight=global_topo_loss_weight,
        patch_topo_loss_weight=patch_topo_loss_weight,
        use_prior_as_ot_target=use_prior_as_ot_target,
        prior_ot_global_mass=prior_ot_global_mass,
    )

    model = StructuralVQAPrototype(
        config=config,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    return model, tokenizer, answer_vocab
