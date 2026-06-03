"""Standalone TopoAdapter module for CATA/Qwen decoder layers.

This file contains only the topological adapter components used by CATA:

- ``TopoAdapter``: a gated bottleneck residual adapter conditioned on topology.
- ``DecoderLayerWithTopoAdapter``: wrapper that injects the adapter after a
  decoder layer.
- utility functions to locate decoder layers, install adapters, build topology
  condition vectors, set conditions, collect gate diagnostics, and patch a model's
  ``forward``/``generate`` methods.

The adapter is initialized as an exact no-op because ``up_proj`` starts at zero.
This makes it safe to insert into pretrained LLM decoder layers before training.
"""
from __future__ import annotations

import types
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class TopoAdapter(nn.Module):
    """Gated bottleneck adapter conditioned on topological/morphological cues.

    Args:
        hidden_dim: LLM hidden-state dimension.
        topo_dim: Dimension of the topology condition vector.
        bottleneck_dim: Adapter bottleneck size.

    Input shapes:
        hidden_states: ``[batch, seq_len, hidden_dim]``.
        topo_condition: either ``[batch, topo_dim]`` or
            ``[batch, seq_len, topo_dim]``.

    The adapter computes::

        t = projector(topo_condition)
        gate = sigmoid(gating(t))
        delta = up_proj(down_proj(hidden_states) * gate)
        output = hidden_states + delta

    ``up_proj`` is zero-initialized, so the adapter is an exact no-op at
    insertion time and learns residual corrections during training.
    """

    def __init__(self, hidden_dim: int, topo_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.topo_dim = int(topo_dim)
        self.bottleneck_dim = int(bottleneck_dim)

        self.topo_projector = nn.Sequential(
            nn.Linear(self.topo_dim, self.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
        )
        self.down_proj = nn.Linear(self.hidden_dim, self.bottleneck_dim, bias=False)
        self.gating = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, self.hidden_dim, bias=False)

        # Exact no-op at initialization: delta == 0 until the adapter learns.
        nn.init.zeros_(self.up_proj.weight)

        self.last_gate_loss: Optional[torch.Tensor] = None
        self.last_gate_mean: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, topo_condition: torch.Tensor) -> torch.Tensor:
        if topo_condition.ndim == 2:
            topo_condition = topo_condition.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        if topo_condition.ndim != 3:
            raise ValueError(
                "topo_condition must have shape [B, D] or [B, S, D], "
                f"got {tuple(topo_condition.shape)}"
            )
        if topo_condition.shape[-1] != self.topo_dim:
            raise ValueError(f"Expected topo_dim={self.topo_dim}, got {topo_condition.shape[-1]}")

        topo_condition = topo_condition.to(
            device=hidden_states.device,
            dtype=self.topo_projector[0].weight.dtype,
        )
        hs = hidden_states.to(dtype=self.down_proj.weight.dtype)

        topo_emb = self.topo_projector(topo_condition)
        gate = torch.sigmoid(self.gating(topo_emb))
        self.last_gate_loss = (gate * (1.0 - gate)).mean()
        self.last_gate_mean = gate.detach().mean()

        delta = self.up_proj(self.down_proj(hs) * gate).to(dtype=hidden_states.dtype)
        return hidden_states + delta


class DecoderLayerWithTopoAdapter(nn.Module):
    """Wrap an LLM decoder layer and apply ``TopoAdapter`` to its hidden states."""

    def __init__(self, base_layer: nn.Module, adapter: TopoAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.topo_adapter = adapter
        self.topo_condition: Optional[torch.Tensor] = None

    def set_topo_condition(self, condition: Optional[torch.Tensor]) -> None:
        self.topo_condition = condition

    def forward(self, *args, **kwargs):
        out = self.base_layer(*args, **kwargs)
        if self.topo_condition is None:
            return out
        if isinstance(out, tuple):
            hidden_states = self.topo_adapter(out[0], self.topo_condition)
            return (hidden_states,) + out[1:]
        return self.topo_adapter(out, self.topo_condition)


def _resolve_attr(root: object, dotted_path: str) -> object | None:
    obj = root
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def find_decoder_layers(llm: nn.Module, candidates: Optional[Sequence[str]] = None) -> nn.ModuleList:
    """Find the decoder ``ModuleList`` for Qwen/PEFT-style model wrappers."""
    candidates = candidates or (
        "model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
        "model.model.layers",
    )
    for path in candidates:
        obj = _resolve_attr(llm, path)
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("Cannot locate decoder layers for TopoAdapter injection")


def get_hidden_dim(llm: nn.Module) -> int:
    """Return the LLM embedding hidden dimension."""
    if hasattr(llm, "get_input_embeddings"):
        emb = llm.get_input_embeddings()
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    config = getattr(llm, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    raise RuntimeError("Cannot infer LLM hidden dimension")


def install_topo_adapters(
    model: nn.Module,
    topo_dim: int,
    bottleneck_dim: int = 32,
    last_n_layers: int = 8,
    every_n_layers: int = 0,
    llm_attr: str = "llm",
    verbose: bool = True,
) -> List[DecoderLayerWithTopoAdapter]:
    """Install TopoAdapters into selected decoder layers.

    Args:
        model: CATA/Structural VQA model or an object containing an LLM.
        topo_dim: Dimension of the topology condition vector.
        bottleneck_dim: Adapter bottleneck size.
        last_n_layers: Install into the final N decoder layers.
        every_n_layers: Additionally install every N layers when > 0.
        llm_attr: Attribute name used to access the LLM from ``model``.
        verbose: Print installation summary.

    Returns:
        List of installed/wrapped decoder layers.
    """
    llm = getattr(model, llm_attr, model)
    layers = find_decoder_layers(llm)
    hidden_dim = get_hidden_dim(llm)
    n_layers = len(layers)

    selected = set(range(max(0, n_layers - int(last_n_layers)), n_layers)) if last_n_layers > 0 else set()
    if every_n_layers > 0:
        selected.update(range(0, n_layers, int(every_n_layers)))

    wrappers: List[DecoderLayerWithTopoAdapter] = []
    for idx in sorted(selected):
        if isinstance(layers[idx], DecoderLayerWithTopoAdapter):
            wrappers.append(layers[idx])
            continue
        wrapper = DecoderLayerWithTopoAdapter(
            layers[idx],
            TopoAdapter(hidden_dim=hidden_dim, topo_dim=topo_dim, bottleneck_dim=bottleneck_dim),
        )
        layers[idx] = wrapper
        wrappers.append(wrapper)

    if verbose:
        print(
            f"Installed {len(wrappers)} TopoAdapters / {n_layers} decoder layers | "
            f"hidden={hidden_dim} topo_dim={topo_dim} bottleneck={bottleneck_dim}"
        )
    return wrappers


def build_topo_condition(
    prior_mask: torch.Tensor,
    topo_features: torch.Tensor,
    global_features: Optional[torch.Tensor] = None,
    mode: str = "all",
) -> torch.Tensor:
    """Build the topology condition vector used by CATA.

    Args:
        prior_mask: Prior mask tensor, usually ``[B, 14, 14]``.
        topo_features: Patch topology features, usually ``[B, 14, 14, C]``.
        global_features: Optional global morphology vector, usually ``[B, G]``.
        mode: ``"patch"`` uses patch topology statistics only. ``"all"`` adds
            prior-mask summary statistics and global features.

    Returns:
        Tensor of shape ``[B, 3*C]`` for ``mode='patch'`` or
        ``[B, 3*C + 3 + G]`` for ``mode='all'``.
    """
    if topo_features.ndim != 4:
        raise ValueError(f"topo_features must have shape [B, H, W, C], got {tuple(topo_features.shape)}")

    tf = topo_features.float()
    flat = tf.flatten(1, 2)
    parts = [flat.mean(1), flat.std(1), flat.amax(1)]

    if mode == "all":
        if prior_mask is None:
            raise ValueError("prior_mask is required when mode='all'")
        pm = prior_mask.float().flatten(1)
        parts.extend([
            pm.mean(1, keepdim=True),
            pm.std(1, keepdim=True),
            pm.amax(1, keepdim=True),
        ])
        if global_features is not None:
            parts.append(global_features.float())
    elif mode not in {"patch", "topo", "topology"}:
        raise ValueError(f"Unsupported topo condition mode: {mode}")

    return torch.cat(parts, dim=-1)


def set_topo_condition(wrappers: Sequence[DecoderLayerWithTopoAdapter], condition: Optional[torch.Tensor]) -> None:
    for wrapper in wrappers:
        wrapper.set_topo_condition(condition)


def collect_gate_stats(
    wrappers: Sequence[DecoderLayerWithTopoAdapter],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return average gate decisiveness loss and average gate value."""
    if device is None:
        device = next(wrappers[0].parameters()).device if wrappers else torch.device("cpu")
    losses = [w.topo_adapter.last_gate_loss for w in wrappers if w.topo_adapter.last_gate_loss is not None]
    means = [w.topo_adapter.last_gate_mean for w in wrappers if w.topo_adapter.last_gate_mean is not None]
    gate_loss = torch.stack([x.to(device) for x in losses]).mean() if losses else torch.tensor(0.0, device=device)
    gate_mean = torch.stack([x.to(device) for x in means]).mean() if means else torch.tensor(0.0, device=device)
    return gate_loss, gate_mean


def patch_model_with_topo_adapters(
    model: nn.Module,
    wrappers: Sequence[DecoderLayerWithTopoAdapter],
    mode: str = "all",
) -> None:
    """Patch a CATA-style model so forward/generate set adapter conditions.

    The target model is expected to expose methods with this signature::

        forward(image, prior_mask, topo_features, global_features,
                question_text, answer_text=None, return_diagnostics=True)
        generate(image, prior_mask, topo_features, global_features,
                 question_text, max_new_tokens=64)
    """
    original_forward = model.forward
    original_generate = model.generate

    def forward_with_topo(
        self,
        image,
        prior_mask,
        topo_features,
        global_features,
        question_text,
        answer_text=None,
        return_diagnostics=True,
    ):
        condition = build_topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        set_topo_condition(wrappers, condition)
        try:
            return original_forward(
                image,
                prior_mask,
                topo_features,
                global_features,
                question_text,
                answer_text,
                return_diagnostics,
            )
        finally:
            set_topo_condition(wrappers, None)

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
        condition = build_topo_condition(prior_mask, topo_features, global_features, mode).to(image.device)
        set_topo_condition(wrappers, condition)
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
            set_topo_condition(wrappers, None)

    model.forward = types.MethodType(forward_with_topo, model)
    model.generate = types.MethodType(generate_with_topo, model)


__all__ = [
    "TopoAdapter",
    "DecoderLayerWithTopoAdapter",
    "find_decoder_layers",
    "get_hidden_dim",
    "install_topo_adapters",
    "build_topo_condition",
    "set_topo_condition",
    "collect_gate_stats",
    "patch_model_with_topo_adapters",
]
