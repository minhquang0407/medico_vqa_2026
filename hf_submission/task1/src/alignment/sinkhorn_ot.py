from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SinkhornOTConfig:
    """
    Cấu hình Sinkhorn Optimal Transport Router.

    Args:
        epsilon: Entropic regularization. Nhỏ hơn -> transport sắc hơn nhưng dễ unstable.
        num_iters: Số vòng Sinkhorn normalization.
        cost_type: "cosine" hoặc "euclidean".
        normalize_embeddings: L2 normalize embeddings trước khi tính cosine cost.
        reduction: "mean", "sum", hoặc "none" cho ot_cost.
        eps: Số nhỏ tránh chia log(0).
    """

    epsilon: float = 0.07
    num_iters: int = 50
    cost_type: str = "cosine"
    normalize_embeddings: bool = True
    reduction: str = "mean"
    eps: float = 1e-8

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("epsilon phải > 0")
        if self.num_iters <= 0:
            raise ValueError("num_iters phải > 0")
        if self.cost_type not in {"cosine", "euclidean"}:
            raise ValueError("cost_type phải là 'cosine' hoặc 'euclidean'")
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction phải là 'mean', 'sum', hoặc 'none'")


class SinkhornOTRouter(nn.Module):
    """
    Cross-modal alignment bằng Sinkhorn Optimal Transport.

    Input:
        visual_tokens: [B, Nv, D]
        text_tokens:   [B, Nt, D]
        visual_mask:   [B, Nv] optional, 1=valid, 0=pad
        text_mask:     [B, Nt] optional, 1=valid, 0=pad

    Output:
        transport_plan: [B, Nt, Nv]
        cost_matrix:    [B, Nt, Nv]
        aligned_visual: [B, Nt, D]
        ot_cost:        scalar hoặc [B]

    Ý nghĩa:
        Mỗi text token học cách "vận tải attention" tới các visual tokens phù hợp.
    """

    def __init__(self, config: Optional[SinkhornOTConfig] = None):
        super().__init__()
        self.config = config or SinkhornOTConfig()

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        visual_target_mass: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ):
        self._validate_inputs(visual_tokens, text_tokens)
        batch_size, num_visual, d_model = visual_tokens.shape
        _, num_text, _ = text_tokens.shape

        visual_mask = self._prepare_mask(
            visual_mask,
            batch_size=batch_size,
            length=num_visual,
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        text_mask = self._prepare_mask(
            text_mask,
            batch_size=batch_size,
            length=num_text,
            device=text_tokens.device,
            dtype=text_tokens.dtype,
        )

        visual_target_mass = self._prepare_mask(
            visual_target_mass,
            batch_size=batch_size,
            length=num_visual,
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )

        cost_matrix = self.compute_cost_matrix(text_tokens, visual_tokens)
        transport_plan = self.sinkhorn(
            cost_matrix,
            text_mask=text_mask,
            visual_mask=visual_mask,
            visual_target_mass=visual_target_mass,
        )

        aligned_visual = torch.bmm(transport_plan, visual_tokens)
        text_mass = transport_plan.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        aligned_visual = aligned_visual / text_mass

        per_sample_cost = (transport_plan * cost_matrix).sum(dim=(1, 2))
        ot_cost = self._reduce(per_sample_cost)

        if not return_diagnostics:
            return aligned_visual

        return {
            "transport_plan": transport_plan,
            "cost_matrix": cost_matrix,
            "aligned_visual": aligned_visual,
            "ot_cost": ot_cost,
            "per_sample_ot_cost": per_sample_cost,
            "transport_entropy": self.transport_entropy(transport_plan),
            "text_mass": transport_plan.sum(dim=-1),
            "visual_mass": transport_plan.sum(dim=1),
            "visual_target_mass": visual_target_mass,
        }

    def compute_cost_matrix(self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        if self.config.cost_type == "cosine":
            text = text_tokens
            visual = visual_tokens
            if self.config.normalize_embeddings:
                text = F.normalize(text, p=2, dim=-1)
                visual = F.normalize(visual, p=2, dim=-1)
            similarity = torch.bmm(text, visual.transpose(1, 2))
            return 1.0 - similarity

        # Euclidean squared cost: ||t - v||^2
        text_sq = (text_tokens ** 2).sum(dim=-1, keepdim=True)
        visual_sq = (visual_tokens ** 2).sum(dim=-1).unsqueeze(1)
        cross = torch.bmm(text_tokens, visual_tokens.transpose(1, 2))
        return (text_sq + visual_sq - 2.0 * cross).clamp_min(0.0)

    def sinkhorn(
        self,
        cost_matrix: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        visual_target_mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_text, num_visual = cost_matrix.shape
        dtype = cost_matrix.dtype
        device = cost_matrix.device

        if text_mask is None:
            text_mask = torch.ones(batch_size, num_text, device=device, dtype=dtype)
        if visual_mask is None:
            visual_mask = torch.ones(batch_size, num_visual, device=device, dtype=dtype)

        text_mask = text_mask.to(device=device, dtype=dtype)
        visual_mask = visual_mask.to(device=device, dtype=dtype)

        source_mass = text_mask / text_mask.sum(dim=1, keepdim=True).clamp_min(self.config.eps)
        if visual_target_mass is None:
            target_mass = visual_mask / visual_mask.sum(dim=1, keepdim=True).clamp_min(self.config.eps)
        else:
            visual_target_mass = visual_target_mass.to(device=device, dtype=dtype) * visual_mask
            target_mass = visual_target_mass / visual_target_mass.sum(dim=1, keepdim=True).clamp_min(self.config.eps)

        kernel = torch.exp(-cost_matrix / self.config.epsilon)
        valid_matrix = text_mask.unsqueeze(-1) * visual_mask.unsqueeze(1)
        kernel = kernel * valid_matrix + self.config.eps * valid_matrix

        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)

        for _ in range(self.config.num_iters):
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(self.config.eps)
            u = source_mass / kv
            u = u * text_mask

            ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(self.config.eps)
            v = target_mass / ktu
            v = v * visual_mask

        transport_plan = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
        transport_plan = transport_plan * valid_matrix
        return transport_plan

    @staticmethod
    def transport_entropy(transport_plan: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        entropy = -(transport_plan * torch.log(transport_plan.clamp_min(eps))).sum(dim=(1, 2))
        return entropy

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if self.config.reduction == "mean":
            return values.mean()
        if self.config.reduction == "sum":
            return values.sum()
        return values

    @staticmethod
    def _validate_inputs(visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> None:
        if not torch.is_tensor(visual_tokens):
            raise TypeError("visual_tokens phải là torch.Tensor")
        if not torch.is_tensor(text_tokens):
            raise TypeError("text_tokens phải là torch.Tensor")
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens phải có shape [B, Nv, D], nhận {tuple(visual_tokens.shape)}")
        if text_tokens.ndim != 3:
            raise ValueError(f"text_tokens phải có shape [B, Nt, D], nhận {tuple(text_tokens.shape)}")
        if visual_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("Batch size visual_tokens và text_tokens không khớp")
        if visual_tokens.shape[-1] != text_tokens.shape[-1]:
            raise ValueError("Hidden dim visual_tokens và text_tokens không khớp")

    @staticmethod
    def _prepare_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device, dtype=dtype)
        if mask.ndim != 2:
            raise ValueError(f"mask phải có shape [B, L], nhận {tuple(mask.shape)}")
        if mask.shape != (batch_size, length):
            raise ValueError(f"mask shape={tuple(mask.shape)} không khớp expected={(batch_size, length)}")
        return mask
