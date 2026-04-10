from __future__ import annotations

from typing import Iterable, List, Literal, Sequence

import torch
from torch import nn

from .erp_geometry import TokenLayout, build_features_per_image


class ERPSphericalPosAdapter(nn.Module):
    """A low-intrusion ERP token adapter.

    It starts close to identity by zero-initializing the final projection layer and
    using a small learnable gate. This makes it friendly to pretrained Qwen weights.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 512,
        feature_dim: int = 10,
        mode: Literal["paper", "extended"] = "extended",
        gate_init: float = 0.01,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.gate = nn.Parameter(torch.full((embed_dim,), float(gate_init)))

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def _adapt_split(
        self,
        token_splits: Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for tokens, spherical_feat in zip(
            token_splits,
            build_features_per_image(
                grid_thw,
                spatial_merge_size,
                mode=self.mode,
                token_layout=token_layout,
                device=token_splits[0].device,
                dtype=token_splits[0].dtype,
            ),
        ):
            delta = self.norm(self.proj(spherical_feat))
            outputs.append(tokens + delta * self.gate.tanh())
        return outputs

    def _adapt_concat(
        self,
        tokens: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> torch.Tensor:
        features = build_features_per_image(
            grid_thw,
            spatial_merge_size,
            mode=self.mode,
            token_layout=token_layout,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        token_splits = list(tokens.split(
            [feat.shape[0] for feat in features],
            dim=0,
        ))
        adapted = self._adapt_split(
            token_splits,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
        return torch.cat(adapted, dim=0)

    def forward(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout = "merged",
    ) -> torch.Tensor | List[torch.Tensor]:
        if isinstance(tokens, torch.Tensor):
            return self._adapt_concat(
                tokens,
                grid_thw,
                spatial_merge_size,
                token_layout=token_layout,
            )
        return self._adapt_split(
            tokens,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
