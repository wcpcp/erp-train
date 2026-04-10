from __future__ import annotations

import math
from typing import Iterable, List, Literal, Sequence, Tuple

import torch


TokenLayout = Literal["premerge", "merged"]


def merged_grid_shapes(grid_thw: torch.Tensor, spatial_merge_size: int) -> List[Tuple[int, int, int]]:
    """Convert raw grid_thw into merged-token grid sizes."""
    shapes: List[Tuple[int, int, int]] = []
    for t, h, w in grid_thw.tolist():
        merged_h = max(int(h) // spatial_merge_size, 1)
        merged_w = max(int(w) // spatial_merge_size, 1)
        shapes.append((int(t), merged_h, merged_w))
    return shapes


def split_sizes_from_grid(grid_thw: torch.Tensor, spatial_merge_size: int) -> List[int]:
    return [t * h * w for t, h, w in merged_grid_shapes(grid_thw, spatial_merge_size)]


def token_grid_shapes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    token_layout: TokenLayout,
) -> List[Tuple[int, int, int]]:
    if token_layout == "premerge":
        return [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
    if token_layout == "merged":
        return merged_grid_shapes(grid_thw, spatial_merge_size)
    raise ValueError(f"Unsupported token layout: {token_layout}")


def build_erp_sincos_features(
    num_frames: int,
    grid_h: int,
    grid_w: int,
    *,
    mode: Literal["paper", "extended"] = "extended",
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create ERP-aware angular features for each merged visual token.

    The features extend the PanoVGGT-style sin/cos encoding with:
    - second-order harmonics
    - latitude area weighting
    """
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / float(grid_w)

    pitch = (0.5 - ys) * math.pi
    yaw = (xs - 0.5) * (2.0 * math.pi)

    pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw, indexing="ij")
    if mode == "paper":
        # Closest to the PanoVGGT paper:
        # pvec = [sin(theta), cos(theta), sin(phi), cos(phi)]
        base = torch.stack(
            [
                torch.sin(yaw_grid),
                torch.cos(yaw_grid),
                torch.sin(pitch_grid),
                torch.cos(pitch_grid),
            ],
            dim=-1,
        ).reshape(grid_h * grid_w, -1)
    elif mode == "extended":
        area_weight = torch.cos(pitch_grid).clamp_min(1e-4)
        base = torch.stack(
            [
                torch.sin(yaw_grid),
                torch.cos(yaw_grid),
                torch.sin(pitch_grid),
                torch.cos(pitch_grid),
                torch.sin(2.0 * yaw_grid),
                torch.cos(2.0 * yaw_grid),
                torch.sin(2.0 * pitch_grid),
                torch.cos(2.0 * pitch_grid),
                area_weight,
                pitch_grid / (0.5 * math.pi),
            ],
            dim=-1,
        ).reshape(grid_h * grid_w, -1)
    else:
        raise ValueError(f"Unsupported ERP position mode: {mode}")

    if num_frames > 1:
        base = base.repeat(num_frames, 1)

    return base.to(device=device, dtype=dtype)


def build_features_per_image(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    mode: Literal["paper", "extended"] = "extended",
    token_layout: TokenLayout = "merged",
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    return [
        build_erp_sincos_features(t, h, w, mode=mode, device=device, dtype=dtype)
        for t, h, w in token_grid_shapes(grid_thw, spatial_merge_size, token_layout=token_layout)
    ]
