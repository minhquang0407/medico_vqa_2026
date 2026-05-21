from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion import StructuralFusionConfig, StructuralVisualFusion


@dataclass
class VisionEncoderConfig:
    """
    Cấu hình cho vision encoder wrapper.

    Args:
        backbone_name: Tên backbone. Ưu tiên timm nếu có, ví dụ
            "vit_base_patch16_224".
        pretrained: Dùng pretrained weights nếu backend hỗ trợ.
        image_size: Kích thước ảnh input HxW.
        patch_size: Patch size của ViT. Với 224/16 -> 14x14 -> 196 tokens.
        d_model: Hidden dimension output patch tokens.
        freeze_backbone: Nếu True, freeze ViT backbone để train fusion trước.
        backend: "auto", "timm", hoặc "torchvision".
        drop_cls_token: Nếu True, bỏ CLS token để chỉ giữ 196 patch tokens.
    """

    backbone_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    d_model: int = 768
    freeze_backbone: bool = True
    backend: str = "auto"
    drop_cls_token: bool = True

    @property
    def num_patches(self) -> int:
        return (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)


class PatchVisionEncoder(nn.Module):
    """
    Wrapper lấy patch tokens từ pretrained ViT.

    Input:
        image: [B, 3, H, W]

    Output:
        patch_tokens: [B, N, D]

    Backend:
        - timm: ưu tiên vì hỗ trợ forward_features tiện.
        - torchvision: fallback cho vit_b_16 nếu không có timm.
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.backbone, self.backend, inferred_dim = self._build_backbone(config)

        self.output_projection = nn.Identity()
        if inferred_dim != config.d_model:
            self.output_projection = nn.Linear(inferred_dim, config.d_model)

        if config.freeze_backbone:
            self.freeze_backbone()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)

        if self.backend == "timm":
            tokens = self._forward_timm(image)
        elif self.backend == "torchvision":
            tokens = self._forward_torchvision(image)
        else:
            raise RuntimeError(f"Backend không hỗ trợ: {self.backend}")

        tokens = self._ensure_patch_tokens(tokens)
        tokens = self.output_projection(tokens)
        return tokens

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze vài block cuối nếu backend expose blocks/encoder.layers."""
        self.freeze_backbone()
        if num_blocks <= 0:
            return

        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
            for block in blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            layers = self.backbone.encoder.layers
            for layer in list(layers)[-num_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError("image phải là torch.Tensor")
        if image.ndim != 4:
            raise ValueError(f"image phải có shape [B, 3, H, W], nhận {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"image channel phải là 3 RGB, nhận C={image.shape[1]}")

    def _build_backbone(self, config: VisionEncoderConfig):
        backend = config.backend.lower()
        if backend not in {"auto", "timm", "torchvision"}:
            raise ValueError("backend phải là 'auto', 'timm', hoặc 'torchvision'")

        if backend in {"auto", "timm"}:
            try:
                import timm

                backbone = timm.create_model(
                    config.backbone_name,
                    pretrained=config.pretrained,
                    num_classes=0,
                )
                inferred_dim = getattr(backbone, "num_features", config.d_model)
                return backbone, "timm", int(inferred_dim)
            except ImportError:
                if backend == "timm":
                    raise ImportError(
                        "Bạn chọn backend='timm' nhưng chưa cài timm. Cài bằng: pip install timm"
                    )
            except Exception:
                if backend == "timm":
                    raise

        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16

            weights = ViT_B_16_Weights.DEFAULT if config.pretrained else None
            backbone = vit_b_16(weights=weights)
            inferred_dim = backbone.hidden_dim
            return backbone, "torchvision", int(inferred_dim)
        except Exception as exc:
            raise RuntimeError(
                "Không build được ViT backend. Hãy cài timm bằng `pip install timm` "
                "hoặc kiểm tra torchvision."
            ) from exc

    def _forward_timm(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(image)

        if isinstance(features, dict):
            for key in ("x", "tokens", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise RuntimeError(f"Không nhận diện được output dict từ timm: {features.keys()}")

        return features

    def _forward_torchvision(self, image: torch.Tensor) -> torch.Tensor:
        # Logic gần giống torchvision VisionTransformer._process_input + encoder.
        x = self.backbone._process_input(image)
        batch_size = x.shape[0]
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.backbone.encoder(x)
        return x

    def _ensure_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"ViT output phải có shape [B, N, D], nhận {tuple(tokens.shape)}")

        expected_patches = self.config.num_patches
        num_tokens = tokens.shape[1]

        if self.config.drop_cls_token and num_tokens == expected_patches + 1:
            tokens = tokens[:, 1:, :]
        elif num_tokens == expected_patches:
            pass
        elif num_tokens > expected_patches:
            tokens = tokens[:, -expected_patches:, :]
        else:
            raise ValueError(
                f"Số token ViT={num_tokens} nhỏ hơn expected patches={expected_patches}. "
                "Kiểm tra image_size/patch_size/backbone."
            )

        return tokens


class GlobalFeatureProjector(nn.Module):
    """
    Chiếu global structural features [B, G] thành global token [B, 1, D].
    """

    def __init__(
        self,
        global_feature_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.10,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(d_model // 2, global_feature_dim * 4)

        self.input_norm = nn.LayerNorm(global_feature_dim) if use_layer_norm else nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(global_features):
            global_features = torch.as_tensor(global_features)
        if global_features.ndim != 2:
            raise ValueError(
                f"global_features phải có shape [B, G], nhận {tuple(global_features.shape)}"
            )

        token = self.projector(self.input_norm(global_features.float()))
        token = self.output_norm(token)
        return token.unsqueeze(1)


class StructuralVisionEncoder(nn.Module):
    """
    End-to-end vision side:

        image -> ViT -> patch_tokens
        patch_tokens + prior_mask + topo_features -> StructuralVisualFusion
        global_features -> global structural token
        concat -> visual_context

    Output:
        visual_context: [B, 197, D] nếu dùng global token
        fused_tokens: [B, 196, D]
        patch_tokens: [B, 196, D]
    """

    def __init__(
        self,
        vision_config: VisionEncoderConfig,
        topo_feature_dim: int = 12,
        global_feature_dim: int = 8,
        use_global_token: bool = True,
        fusion_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        fusion_kwargs = fusion_kwargs or {}

        self.vision_config = vision_config
        self.use_global_token = use_global_token
        self.patch_encoder = PatchVisionEncoder(vision_config)

        fusion_config = StructuralFusionConfig(
            d_model=vision_config.d_model,
            topo_feature_dim=topo_feature_dim,
            **fusion_kwargs,
        )
        self.fusion = StructuralVisualFusion(fusion_config)

        self.global_projector = None
        if use_global_token:
            self.global_projector = GlobalFeatureProjector(
                global_feature_dim=global_feature_dim,
                d_model=vision_config.d_model,
                dropout=fusion_config.dropout,
                use_layer_norm=fusion_config.use_layer_norm,
            )

    def forward(
        self,
        image: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
        topo_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ):
        patch_tokens = self.patch_encoder(image)

        fusion_out = self.fusion(
            visual_tokens=patch_tokens,
            prior_mask=prior_mask,
            topo_features=topo_features,
            return_diagnostics=True,
        )
        fused_tokens = fusion_out["fused_tokens"]

        global_token = None
        visual_context = fused_tokens
        if self.use_global_token:
            if global_features is None:
                raise ValueError("use_global_token=True nhưng global_features=None")
            global_features = global_features.to(device=fused_tokens.device, dtype=fused_tokens.dtype)
            global_token = self.global_projector(global_features)
            visual_context = torch.cat([global_token, fused_tokens], dim=1)

        if not return_diagnostics:
            return visual_context

        return {
            "visual_context": visual_context,
            "patch_tokens": patch_tokens,
            "fused_tokens": fused_tokens,
            "global_token": global_token,
            **fusion_out,
        }


def build_structural_vision_encoder(
    d_model: int = 768,
    topo_feature_dim: int = 12,
    global_feature_dim: int = 8,
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    backend: str = "auto",
    use_global_token: bool = True,
    **fusion_kwargs,
) -> StructuralVisionEncoder:
    vision_config = VisionEncoderConfig(
        backbone_name=backbone_name,
        pretrained=pretrained,
        d_model=d_model,
        freeze_backbone=freeze_backbone,
        backend=backend,
    )
    return StructuralVisionEncoder(
        vision_config=vision_config,
        topo_feature_dim=topo_feature_dim,
        global_feature_dim=global_feature_dim,
        use_global_token=use_global_token,
        fusion_kwargs=fusion_kwargs,
    )
