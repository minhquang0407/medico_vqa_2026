from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StructuralFusionConfig:
    """
    Cấu hình cho StructuralVisualFusion.

    Args:
        d_model: Hidden dimension của visual tokens từ ViT/VLM.
        topo_feature_dim: Số chiều feature topology/morphology mỗi patch.
        hidden_dim: Hidden dimension của MLP chiếu topo_features -> d_model.
        prior_strength: Hệ số spatial modulation ban đầu cho lesion prior.
        topo_strength: Hệ số enrichment ban đầu cho topo embedding.
        dropout: Dropout trong topo MLP.
        use_layer_norm: Chuẩn hóa visual/topo branch để ổn định training.
        learnable_gates: Nếu True, alpha/beta là tham số học được.
        normalize_prior: Nếu True, normalize prior_mask về [0, 1] theo từng sample.
        residual_scale: Scale residual cuối để tránh làm lệch mạnh pretrained tokens.
    """

    d_model: int
    topo_feature_dim: int
    hidden_dim: Optional[int] = None
    prior_strength: float = 0.30
    topo_strength: float = 0.10
    dropout: float = 0.10
    use_layer_norm: bool = True
    learnable_gates: bool = True
    normalize_prior: bool = True
    residual_scale: float = 1.0

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError("d_model phải > 0")
        if self.topo_feature_dim <= 0:
            raise ValueError("topo_feature_dim phải > 0")
        if self.hidden_dim is None:
            self.hidden_dim = max(self.d_model // 2, self.topo_feature_dim * 4)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim phải > 0")


class StructuralVisualFusion(nn.Module):
    """
    Fusion module nối ViT visual tokens với lesion prior và topology features.

    Forward logic:
        V_fused = V * (1 + alpha * P) + beta * MLP(F_topo)

    Trong đó:
        V: visual_tokens, shape [B, N, D]
        P: prior_mask, shape [B, N] hoặc [B, H, W]
        F_topo: topo_features, shape [B, N, F] hoặc [B, H, W, F]

    Output chính:
        fused_tokens, shape [B, N, D]
    """

    def __init__(self, config: StructuralFusionConfig):
        super().__init__()
        self.config = config

        self.visual_norm = nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        self.topo_norm = nn.LayerNorm(config.topo_feature_dim) if config.use_layer_norm else nn.Identity()

        self.topo_projector = nn.Sequential(
            nn.Linear(config.topo_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

        if config.learnable_gates:
            self.prior_gate = nn.Parameter(torch.tensor(float(config.prior_strength)))
            self.topo_gate = nn.Parameter(torch.tensor(float(config.topo_strength)))
        else:
            self.register_buffer("prior_gate", torch.tensor(float(config.prior_strength)))
            self.register_buffer("topo_gate", torch.tensor(float(config.topo_strength)))

        self.output_norm = nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()

    def forward(
        self,
        visual_tokens: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
        topo_features: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ):
        """
        Args:
            visual_tokens: Tensor [B, N, D].
            prior_mask: Optional tensor [B, N] hoặc [B, H, W].
            topo_features: Optional tensor [B, N, F] hoặc [B, H, W, F].
            return_diagnostics: Nếu True trả về dict debug; nếu False chỉ trả fused_tokens.

        Returns:
            dict hoặc Tensor:
                {
                    "fused_tokens": [B, N, D],
                    "prior_weights": [B, N, 1] | None,
                    "topo_embedding": [B, N, D] | None,
                    "prior_gate": scalar,
                    "topo_gate": scalar,
                }
        """
        self._validate_visual_tokens(visual_tokens)
        batch_size, num_tokens, d_model = visual_tokens.shape

        fused_tokens = self.visual_norm(visual_tokens)
        prior_weights = None
        topo_embedding = None

        if prior_mask is not None:
            prior_weights = self._prepare_prior_mask(
                prior_mask,
                batch_size=batch_size,
                num_tokens=num_tokens,
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            fused_tokens = fused_tokens * (1.0 + self.prior_gate * prior_weights)

        if topo_features is not None:
            topo_features = self._prepare_topo_features(
                topo_features,
                batch_size=batch_size,
                num_tokens=num_tokens,
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            topo_features = self.topo_norm(topo_features)
            topo_embedding = self.topo_projector(topo_features)
            fused_tokens = fused_tokens + self.topo_gate * topo_embedding

        if self.config.residual_scale != 1.0:
            fused_tokens = visual_tokens + self.config.residual_scale * (fused_tokens - visual_tokens)

        fused_tokens = self.output_norm(fused_tokens)

        if not return_diagnostics:
            return fused_tokens

        return {
            "fused_tokens": fused_tokens,
            "prior_weights": prior_weights,
            "topo_embedding": topo_embedding,
            "prior_gate": self.prior_gate.detach().clone(),
            "topo_gate": self.topo_gate.detach().clone(),
        }

    def _validate_visual_tokens(self, visual_tokens: torch.Tensor) -> None:
        if not torch.is_tensor(visual_tokens):
            raise TypeError("visual_tokens phải là torch.Tensor")
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"visual_tokens phải có shape [B, N, D], nhận {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.config.d_model:
            raise ValueError(
                f"visual_tokens D={visual_tokens.shape[-1]} không khớp d_model={self.config.d_model}"
            )

    def _prepare_prior_mask(
        self,
        prior_mask: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(prior_mask):
            prior_mask = torch.as_tensor(prior_mask)

        prior_mask = prior_mask.to(device=device, dtype=dtype)

        if prior_mask.ndim == 3:
            prior_mask = prior_mask.flatten(start_dim=1)
        elif prior_mask.ndim == 2:
            pass
        else:
            raise ValueError(
                f"prior_mask phải có shape [B, N] hoặc [B, H, W], nhận {tuple(prior_mask.shape)}"
            )

        if prior_mask.shape[0] != batch_size:
            raise ValueError(
                f"prior_mask batch={prior_mask.shape[0]} không khớp visual batch={batch_size}"
            )
        if prior_mask.shape[1] != num_tokens:
            raise ValueError(
                f"prior_mask N={prior_mask.shape[1]} không khớp visual tokens N={num_tokens}"
            )

        if self.config.normalize_prior:
            prior_mask = self._minmax_normalize_per_sample(prior_mask)

        return prior_mask.unsqueeze(-1)

    def _prepare_topo_features(
        self,
        topo_features: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(topo_features):
            topo_features = torch.as_tensor(topo_features)

        topo_features = topo_features.to(device=device, dtype=dtype)

        if topo_features.ndim == 4:
            topo_features = topo_features.flatten(start_dim=1, end_dim=2)
        elif topo_features.ndim == 3:
            pass
        else:
            raise ValueError(
                "topo_features phải có shape [B, N, F] hoặc [B, H, W, F], "
                f"nhận {tuple(topo_features.shape)}"
            )

        if topo_features.shape[0] != batch_size:
            raise ValueError(
                f"topo_features batch={topo_features.shape[0]} không khớp visual batch={batch_size}"
            )
        if topo_features.shape[1] != num_tokens:
            raise ValueError(
                f"topo_features N={topo_features.shape[1]} không khớp visual tokens N={num_tokens}"
            )
        if topo_features.shape[2] != self.config.topo_feature_dim:
            raise ValueError(
                f"topo_features F={topo_features.shape[2]} không khớp "
                f"topo_feature_dim={self.config.topo_feature_dim}"
            )

        return topo_features

    @staticmethod
    def _minmax_normalize_per_sample(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        min_values = values.amin(dim=1, keepdim=True)
        max_values = values.amax(dim=1, keepdim=True)
        return (values - min_values) / (max_values - min_values + eps)


def build_structural_visual_fusion(
    d_model: int,
    topo_feature_dim: int,
    **kwargs,
) -> StructuralVisualFusion:
    """Factory tiện dụng cho training scripts."""
    config = StructuralFusionConfig(
        d_model=d_model,
        topo_feature_dim=topo_feature_dim,
        **kwargs,
    )
    return StructuralVisualFusion(config)


__all__ = [
    "StructuralFusionConfig",
    "StructuralVisualFusion",
    "build_structural_visual_fusion",
]
