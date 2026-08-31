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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

import argparse
import gc
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

import natten
from natten.functional import neighborhood_attention_generic


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def make_grid_coords(batch: int, height: int, width: int, num_queries: int) -> torch.Tensor:
    y = (torch.arange(height, device="cuda", dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device="cuda", dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2)
    return coords[:, :num_queries].expand(batch, -1, -1).contiguous()


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def make_inputs(args: argparse.Namespace, kind: str):
    if kind == "dense":
        query = torch.randn(
            args.batch,
            args.height,
            args.width,
            args.heads,
            args.dim,
            device="cuda",
            dtype=args.dtype,
            requires_grad=True,
        )
        grad = torch.randn(
            args.batch,
            args.height,
            args.width,
            args.heads,
            args.dim_value,
            device="cuda",
            dtype=args.dtype,
        )
    else:
        query = torch.randn(
            args.batch,
            args.num_queries,
            args.heads,
            args.dim,
            device="cuda",
            dtype=args.dtype,
            requires_grad=True,
        )
        grad = torch.randn(
            args.batch,
            args.num_queries,
            args.heads,
            args.dim_value,
            device="cuda",
            dtype=args.dtype,
        )

    key = torch.randn(
        args.batch,
        args.height,
        args.width,
        args.heads,
        args.dim,
        device="cuda",
        dtype=args.dtype,
        requires_grad=True,
    )
    value = torch.randn(
        args.batch,
        args.height,
        args.width,
        args.heads,
        args.dim_value,
        device="cuda",
        dtype=args.dtype,
        requires_grad=True,
    )
    coords = make_grid_coords(args.batch, args.height, args.width, args.num_queries)
    return query, key, value, coords, grad


def measure(
    label: str,
    kind: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    use_checkpoint: bool,
    args: argparse.Namespace,
) -> None:
    for _ in range(args.warmup):
        clear_cuda()
        query, key, value, coords, grad = make_inputs(args, kind)
        if use_checkpoint:
            out = checkpoint(fn, query, key, value, coords, use_reentrant=False, preserve_rng_state=False)
        else:
            out = fn(query, key, value, coords)
        out.backward(grad)
        torch.cuda.synchronize()
        del query, key, value, coords, grad, out

    clear_cuda()
    query, key, value, coords, grad = make_inputs(args, kind)
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    if use_checkpoint:
        out = checkpoint(fn, query, key, value, coords, use_reentrant=False, preserve_rng_state=False)
    else:
        out = fn(query, key, value, coords)
    out.backward(grad)
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    print(
        f"{label:42s} "
        f"baseline={baseline_alloc / 2**20:9.2f} MiB  "
        f"peak={peak_alloc / 2**20:9.2f} MiB  "
        f"extra={max(0, peak_alloc - baseline_alloc) / 2**20:9.2f} MiB  "
        f"reserved_base={baseline_reserved / 2**20:9.2f} MiB  "
        f"reserved_peak={peak_reserved / 2**20:9.2f} MiB"
    )
    del query, key, value, coords, grad, out
    clear_cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure peak CUDA memory for NA real-case forward+backward.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=1024)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=148)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dim-value", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, nargs=2, default=(9, 9))
    parser.add_argument("--scale", type=float, default=0.1767766952966369)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    args.dtype = parse_dtype(args.dtype)
    args.kernel_size = tuple(args.kernel_size)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not natten.HAS_LIBNATTEN:
        raise RuntimeError("libnatten is not available; build with `python setup.py build_ext --inplace`.")

    torch.manual_seed(42)

    def dense_fn(query, key, value, coords):
        del coords
        return neighborhood_attention_generic(
            query,
            key,
            value,
            kernel_size=args.kernel_size,
            scale=args.scale,
        )

    def sparse_fn(query, key, value, coords):
        return natten.sparse_na2d(
            query,
            key,
            value,
            coords,
            kernel_size=args.kernel_size,
            scale=args.scale,
            sample_mode="indexed",
        )

    def simple_fn(query, key, value, coords):
        return natten.sparse_na2d_simple(
            query,
            key,
            value,
            coords,
            kernel_size=args.kernel_size,
            scale=args.scale,
        )

    print(
        f"real case memory dtype={str(args.dtype).replace('torch.', '')} "
        f"B={args.batch} N={args.num_queries} HxW={args.height}x{args.width} "
        f"heads={args.heads} D={args.dim} Dv={args.dim_value} "
        f"K={args.kernel_size[0]}x{args.kernel_size[1]} scale={args.scale:.6g}"
    )
    print(
        "module                                     baseline       peak      extra  "
        "reserved_base  reserved_peak"
    )
    print("-" * 112)

    for label, kind, fn in [
        ("neighborhood_attention_generic", "dense", dense_fn),
        ("natten_sparse_na2d", "sparse", sparse_fn),
        ("natten_sparse_na2d_simple", "sparse", simple_fn),
    ]:
        measure(label, kind, fn, False, args)
        measure(f"{label} ckpt", kind, fn, True, args)


if __name__ == "__main__":
    main()
