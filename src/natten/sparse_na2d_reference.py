#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def sparse_na2d_query_to_grid(
    coords: Tensor,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    """Maps normalized `[y, x]` coordinates to clamped key/value grid indices."""
    y = ((coords[..., 0] + 1.0) * 0.5 * height - 0.5).round().long().clamp(0, height - 1)
    x = ((coords[..., 1] + 1.0) * 0.5 * width - 0.5).round().long().clamp(0, width - 1)
    return y, x


def sparse_na2d_neighborhood_indices(
    coords: Tensor,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
) -> Tensor:
    """Returns flattened clamped neighborhood indices with shape `[B, N, kh * kw]`."""
    kh, kw = kernel_size
    cy, cx = sparse_na2d_query_to_grid(coords, height, width)
    off_y = torch.arange(-(kh // 2), kh // 2 + 1, device=coords.device)
    off_x = torch.arange(-(kw // 2), kw // 2 + 1, device=coords.device)
    oy, ox = torch.meshgrid(off_y, off_x, indexing="ij")
    yy = (cy[..., None] + oy.reshape(1, 1, -1)).clamp(0, height - 1)
    xx = (cx[..., None] + ox.reshape(1, 1, -1)).clamp(0, width - 1)
    return yy * width + xx


def sparse_na2d_indexed_positions(
    coords: Tensor,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
) -> Tensor:
    """Returns indexed neighborhood positions `[B, N, kh * kw, 2]` in `[y, x]` order."""
    kh, kw = kernel_size
    cy, cx = sparse_na2d_query_to_grid(coords, height, width)
    off_y = torch.arange(-(kh // 2), kh // 2 + 1, device=coords.device)
    off_x = torch.arange(-(kw // 2), kw // 2 + 1, device=coords.device)
    oy, ox = torch.meshgrid(off_y, off_x, indexing="ij")
    yy = (cy[..., None] + oy.reshape(1, 1, -1)).clamp(0, height - 1)
    xx = (cx[..., None] + ox.reshape(1, 1, -1)).clamp(0, width - 1)
    denom_y = max(height - 1, 1)
    denom_x = max(width - 1, 1)
    return torch.stack([yy.to(coords.dtype) / denom_y, xx.to(coords.dtype) / denom_x], dim=-1)


def sparse_na2d_neighborhood_grid(
    coords: Tensor,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
) -> Tensor:
    """Returns grid-sample coordinates `[B, N, kh * kw, 2]` in `[x, y]` order."""
    kh, kw = kernel_size
    cy = ((coords[..., 0] + 1.0) * 0.5 * height - 0.5).clamp(0, height - 1)
    cx = ((coords[..., 1] + 1.0) * 0.5 * width - 0.5).clamp(0, width - 1)
    off_y = torch.arange(-(kh // 2), kh // 2 + 1, device=coords.device, dtype=coords.dtype)
    off_x = torch.arange(-(kw // 2), kw // 2 + 1, device=coords.device, dtype=coords.dtype)
    oy, ox = torch.meshgrid(off_y, off_x, indexing="ij")
    yy = (cy[..., None] + oy.reshape(1, 1, -1)).clamp(0, height - 1)
    xx = (cx[..., None] + ox.reshape(1, 1, -1)).clamp(0, width - 1)
    sample_x = (xx + 0.5) * (2.0 / width) - 1.0
    sample_y = (yy + 0.5) * (2.0 / height) - 1.0
    return torch.stack([sample_x, sample_y], dim=-1)


def sparse_na2d_bilinear_positions(
    coords: Tensor,
    height: int,
    width: int,
    kernel_size: Tuple[int, int],
) -> Tensor:
    """Returns bilinear sample positions `[B, N, kh * kw, 2]` in normalized `[y, x]` order."""
    kh, kw = kernel_size
    cy = ((coords[..., 0] + 1.0) * 0.5 * height - 0.5).clamp(0, height - 1)
    cx = ((coords[..., 1] + 1.0) * 0.5 * width - 0.5).clamp(0, width - 1)
    off_y = torch.arange(-(kh // 2), kh // 2 + 1, device=coords.device, dtype=coords.dtype)
    off_x = torch.arange(-(kw // 2), kw // 2 + 1, device=coords.device, dtype=coords.dtype)
    oy, ox = torch.meshgrid(off_y, off_x, indexing="ij")
    yy = (cy[..., None] + oy.reshape(1, 1, -1)).clamp(0, height - 1)
    xx = (cx[..., None] + ox.reshape(1, 1, -1)).clamp(0, width - 1)
    denom_y = max(height - 1, 1)
    denom_x = max(width - 1, 1)
    return torch.stack([yy / denom_y, xx / denom_x], dim=-1)


def gather_sparse_na2d_neighborhood(x: Tensor, flat_idx: Tensor) -> Tensor:
    """Gathers `[B, Hk, Wk, heads, D]` into `[B, N, K, heads, D]`."""
    if x.dim() != 5:
        raise ValueError(f"x must be [B, Hk, Wk, heads, D], got {x.shape}.")
    batch, height, width, heads, dim = x.shape
    if flat_idx.dim() != 3 or flat_idx.shape[0] != batch:
        raise ValueError(f"flat_idx must be [B, N, K] with matching batch, got {flat_idx.shape}.")

    flat = x.reshape(batch, height * width, heads, dim)
    gather_idx = flat_idx.reshape(batch, -1)[:, :, None, None].expand(-1, -1, heads, dim)
    gathered = torch.gather(flat, 1, gather_idx)
    return gathered.view(batch, flat_idx.shape[1], flat_idx.shape[2], heads, dim)


def sample_sparse_na2d_neighborhood(x: Tensor, sample_grid: Tensor) -> Tensor:
    """Bilinearly samples `[B, Hk, Wk, heads, D]` into `[B, N, K, heads, D]`."""
    if x.dim() != 5:
        raise ValueError(f"x must be [B, Hk, Wk, heads, D], got {x.shape}.")
    batch, height, width, heads, dim = x.shape
    if sample_grid.dim() != 4 or sample_grid.shape[0] != batch or sample_grid.shape[-1] != 2:
        raise ValueError(f"sample_grid must be [B, N, K, 2] with matching batch, got {sample_grid.shape}.")

    x_nchw = x.permute(0, 3, 4, 1, 2).reshape(batch, heads * dim, height, width)
    sample_grid = sample_grid.to(dtype=x_nchw.dtype)
    sampled = torch.nn.functional.grid_sample(
        x_nchw,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1).reshape(batch, sample_grid.shape[1], sample_grid.shape[2], heads, dim)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_anyup_key_rope(k_local: Tensor, key_pos: Tensor, theta: float = 100.0) -> Tensor:
    """Applies AnyUp-style 2-D RoPE over the full `heads * dim` key vector."""
    if k_local.dim() != 5:
        raise ValueError(f"k_local must be [B, N, K, heads, D], got {k_local.shape}.")
    batch, num_queries, num_tokens, heads, dim = k_local.shape
    total_dim = heads * dim
    if total_dim % 4 != 0:
        raise ValueError(f"heads * dim must be divisible by 4 for RoPE, got {total_dim}.")

    freqs_1d = theta ** torch.linspace(
        0,
        -1,
        total_dim // 4,
        device=k_local.device,
        dtype=k_local.dtype,
    )
    freqs_1d = torch.cat([freqs_1d, freqs_1d])
    freqs = torch.zeros(2, total_dim, device=k_local.device, dtype=k_local.dtype)
    freqs[0, : total_dim // 2] = freqs_1d
    freqs[1, -total_dim // 2 :] = freqs_1d
    freqs = freqs * (2 * torch.pi)

    key_pos = key_pos.to(dtype=k_local.dtype)
    angle = key_pos @ freqs
    k_full = k_local.reshape(batch, num_queries, num_tokens, total_dim)
    k_full = k_full * angle.cos() + rotate_half(k_full) * angle.sin()
    return k_full.reshape(batch, num_queries, num_tokens, heads, dim)


def apply_anyup_query_rope(query: Tensor, coords: Tensor, theta: float = 100.0) -> Tensor:
    """Applies AnyUp-style 2-D RoPE over the full `heads * dim` query vector."""
    if query.dim() != 4:
        raise ValueError(f"query must be [B, N, heads, D], got {query.shape}.")
    batch, num_queries, heads, dim = query.shape
    total_dim = heads * dim
    if total_dim % 4 != 0:
        raise ValueError(f"heads * dim must be divisible by 4 for RoPE, got {total_dim}.")

    q_pos = ((coords + 1.0) * 0.5).to(dtype=query.dtype)
    freqs_1d = theta ** torch.linspace(
        0,
        -1,
        total_dim // 4,
        device=query.device,
        dtype=query.dtype,
    )
    freqs_1d = torch.cat([freqs_1d, freqs_1d])
    freqs = torch.zeros(2, total_dim, device=query.device, dtype=query.dtype)
    freqs[0, : total_dim // 2] = freqs_1d
    freqs[1, -total_dim // 2 :] = freqs_1d
    freqs = freqs * (2 * torch.pi)

    angle = q_pos @ freqs
    q_full = query.reshape(batch, num_queries, total_dim)
    q_full = q_full * angle.cos() + rotate_half(q_full) * angle.sin()
    return q_full.reshape(batch, num_queries, heads, dim)


def rms_norm_full_channel(
    x: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Applies RMSNorm over the last full-channel dimension."""
    if weight is not None and weight.numel() != x.shape[-1]:
        raise ValueError(f"RMSNorm weight must have {x.shape[-1]} values, got {weight.numel()}.")
    inv_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(dtype=x.dtype)
    out = x * inv_rms
    if weight is not None:
        out = out * weight.to(device=x.device, dtype=x.dtype)
    return out


def apply_query_rms_norm(
    query: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Applies Q RMSNorm over `heads * D`, preserving `[B, N, heads, D]` layout."""
    batch, num_queries, heads, dim = query.shape
    full = query.reshape(batch, num_queries, heads * dim)
    full = rms_norm_full_channel(full, weight=weight, eps=eps)
    return full.reshape(batch, num_queries, heads, dim)


def apply_key_rms_norm(
    k_local: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Applies K RMSNorm over `heads * D`, preserving `[B, N, K, heads, D]` layout."""
    batch, num_queries, num_tokens, heads, dim = k_local.shape
    full = k_local.reshape(batch, num_queries, num_tokens, heads * dim)
    full = rms_norm_full_channel(full, weight=weight, eps=eps)
    return full.reshape(batch, num_queries, num_tokens, heads, dim)


def sparse_na2d_pytorch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    coords: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    scale: Optional[float] = None,
    sample_mode: str = "indexed",
    apply_query_rope: bool = False,
    apply_key_rope: bool = False,
    rope_theta: float = 100.0,
    apply_qk_norm: bool = False,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    qk_norm_eps: float = 1e-6,
    qk_norm_before_rope: bool = True,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Naive materialized PyTorch implementation of sparse NA2D.

    This follows the same tensor contract and coordinate semantics as
    `natten.sparse_na2d`, but explicitly materializes local neighborhoods.
    It is intended for correctness checks and benchmarking, not production use.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        kernel_size = tuple(kernel_size)

    if query.dim() != 4:
        raise ValueError(f"query must be [B, N, heads, D], got {query.shape}.")
    if key.dim() != 5 or value.dim() != 5:
        raise ValueError(f"key/value must be [B, Hk, Wk, heads, D], got {key.shape=} and {value.shape=}.")
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be [B, N, 2], got {coords.shape}.")

    _, height, width, _, _ = key.shape
    scale = scale or query.shape[-1] ** -0.5
    if sample_mode == "indexed":
        flat_idx = sparse_na2d_neighborhood_indices(coords, height, width, kernel_size)
        k_local = gather_sparse_na2d_neighborhood(key, flat_idx)
        v_local = gather_sparse_na2d_neighborhood(value, flat_idx)
        key_pos = sparse_na2d_indexed_positions(coords, height, width, kernel_size)
    elif sample_mode == "bilinear":
        sample_grid = sparse_na2d_neighborhood_grid(coords, height, width, kernel_size)
        k_local = sample_sparse_na2d_neighborhood(key, sample_grid)
        v_local = sample_sparse_na2d_neighborhood(value, sample_grid)
        key_pos = sparse_na2d_bilinear_positions(coords, height, width, kernel_size)
    else:
        raise ValueError(f"sample_mode must be 'indexed' or 'bilinear', got {sample_mode}.")

    if apply_qk_norm and qk_norm_before_rope:
        query = apply_query_rms_norm(query, weight=q_norm_weight, eps=qk_norm_eps)

    if apply_qk_norm and qk_norm_before_rope:
        k_local = apply_key_rms_norm(k_local, weight=k_norm_weight, eps=qk_norm_eps)

    if apply_query_rope:
        query = apply_anyup_query_rope(query, coords, theta=rope_theta)

    if apply_key_rope:
        k_local = apply_anyup_key_rope(k_local, key_pos, theta=rope_theta)

    if apply_qk_norm and not qk_norm_before_rope:
        query = apply_query_rms_norm(query, weight=q_norm_weight, eps=qk_norm_eps)
        k_local = apply_key_rms_norm(k_local, weight=k_norm_weight, eps=qk_norm_eps)

    logits = torch.einsum("bnhd,bnkhd->bnhk", query, k_local) * scale
    attn = logits.softmax(dim=-1)
    output = torch.einsum("bnhk,bnkhd->bnhd", attn, v_local)
    if return_lse:
        return output, logits.logsumexp(dim=-1)
    return output
