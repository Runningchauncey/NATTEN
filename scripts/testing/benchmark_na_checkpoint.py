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
import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

import natten
from natten import sparse_na2d as natten_sparse_na2d
from natten import sparse_na2d_bilinear as natten_sparse_na2d_bilinear
from natten import sparse_na2d_simple as natten_sparse_na2d_simple
from natten.functional import neighborhood_attention_generic


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    num_queries: Optional[int]
    height: int
    width: int
    heads: int
    dim: int
    dim_value: int
    kernel_size: Tuple[int, int]
    scale: Optional[float] = None


CASES = [
    Case("tiny", 1, None, 32, 32, 4, 32, 64, (5, 5)),
    Case("small", 1, None, 37, 37, 4, 32, 128, (5, 5)),
    Case("medium", 1, None, 64, 64, 4, 32, 128, (5, 5)),
    Case("real", 8, 1024, 112, 148, 8, 32, 32, (9, 9), 0.1767766952966369),
]


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def make_grid_coords(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    num_queries: Optional[int],
) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2)
    if num_queries is not None:
        coords = coords[:, :num_queries]
    return coords.expand(batch, -1, -1).contiguous()


def make_inputs(case: Case, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    query = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    key = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    value = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    num_sparse_queries = case.num_queries or case.height * case.width
    sparse_query = torch.randn(
        case.batch,
        num_sparse_queries,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    coords = make_grid_coords(case.batch, case.height, case.width, query.device, case.num_queries)
    dense_grad = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
    )
    sparse_grad = torch.randn(
        case.batch,
        num_sparse_queries,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
    )
    return query, sparse_query, key, value, coords, dense_grad, sparse_grad


def mean_std(samples: list[float]) -> Tuple[float, float]:
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return mean_std(samples)


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def run_checkpointed(
    fn: Callable[..., torch.Tensor],
    *args: torch.Tensor,
    use_reentrant: bool,
) -> torch.Tensor:
    return checkpoint(fn, *args, use_reentrant=use_reentrant, preserve_rng_state=False)


def benchmark_forward(
    fn: Callable[[], torch.Tensor],
    checkpointed_fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def run_plain() -> None:
        fn()

    def run_checkpoint() -> None:
        checkpointed_fn()

    return time_cuda(run_plain, warmup, iters), time_cuda(run_checkpoint, warmup, iters)


def benchmark_backward(
    inputs: Tuple[torch.Tensor, ...],
    grad: torch.Tensor,
    fn: Callable[[], torch.Tensor],
    checkpointed_fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def time_backward(make_output: Callable[[], torch.Tensor]) -> Tuple[float, float]:
        samples = []
        for i in range(warmup + iters):
            clear_grads(*inputs)
            out = make_output()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if i >= warmup:
                start.record()
            out.backward(grad)
            if i >= warmup:
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                torch.cuda.synchronize()
        return mean_std(samples)

    return time_backward(fn), time_backward(checkpointed_fn)


def benchmark_case(
    case: Case,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    backend: Optional[str],
    sample_mode: str,
    use_reentrant: bool,
) -> None:
    query, sparse_query, key, value, coords, dense_grad, sparse_grad = make_inputs(case, dtype)

    def dense_na(query_arg: torch.Tensor, key_arg: torch.Tensor, value_arg: torch.Tensor) -> torch.Tensor:
        return neighborhood_attention_generic(
            query_arg,
            key_arg,
            value_arg,
            kernel_size=case.kernel_size,
            scale=case.scale,
            backend=backend,
        )

    def sparse_na(
        query_arg: torch.Tensor,
        key_arg: torch.Tensor,
        value_arg: torch.Tensor,
        coords_arg: torch.Tensor,
    ) -> torch.Tensor:
        return natten_sparse_na2d(
            query_arg,
            key_arg,
            value_arg,
            coords_arg,
            kernel_size=case.kernel_size,
            scale=case.scale,
            sample_mode=sample_mode,
        )

    def sparse_na_simple(
        query_arg: torch.Tensor,
        key_arg: torch.Tensor,
        value_arg: torch.Tensor,
        coords_arg: torch.Tensor,
    ) -> torch.Tensor:
        return natten_sparse_na2d_simple(
            query_arg,
            key_arg,
            value_arg,
            coords_arg,
            kernel_size=case.kernel_size,
            scale=case.scale,
        )

    def sparse_na_bilinear(
        query_arg: torch.Tensor,
        key_arg: torch.Tensor,
        value_arg: torch.Tensor,
        coords_arg: torch.Tensor,
    ) -> torch.Tensor:
        return natten_sparse_na2d_bilinear(
            query_arg,
            key_arg,
            value_arg,
            coords_arg,
            kernel_size=case.kernel_size,
            scale=case.scale,
        )

    dense_fn = lambda: dense_na(query, key, value)
    dense_checkpointed_fn = lambda: run_checkpointed(dense_na, query, key, value, use_reentrant=use_reentrant)
    sparse_fn = lambda: sparse_na(sparse_query, key, value, coords)
    sparse_checkpointed_fn = lambda: run_checkpointed(
        sparse_na,
        sparse_query,
        key,
        value,
        coords,
        use_reentrant=use_reentrant,
    )
    sparse_simple_fn = lambda: sparse_na_simple(sparse_query, key, value, coords)
    sparse_simple_checkpointed_fn = lambda: run_checkpointed(
        sparse_na_simple,
        sparse_query,
        key,
        value,
        coords,
        use_reentrant=use_reentrant,
    )
    sparse_bilinear_fn = lambda: sparse_na_bilinear(sparse_query, key, value, coords)
    sparse_bilinear_checkpointed_fn = lambda: run_checkpointed(
        sparse_na_bilinear,
        sparse_query,
        key,
        value,
        coords,
        use_reentrant=use_reentrant,
    )

    dense_fwd, dense_fwd_ckpt = benchmark_forward(dense_fn, dense_checkpointed_fn, warmup, iters)
    dense_bwd, dense_bwd_ckpt = benchmark_backward(
        (query, key, value),
        dense_grad,
        dense_fn,
        dense_checkpointed_fn,
        warmup,
        iters,
    )
    sparse_fwd, sparse_fwd_ckpt = benchmark_forward(sparse_fn, sparse_checkpointed_fn, warmup, iters)
    sparse_bwd, sparse_bwd_ckpt = benchmark_backward(
        (sparse_query, key, value),
        sparse_grad,
        sparse_fn,
        sparse_checkpointed_fn,
        warmup,
        iters,
    )
    sparse_simple_fwd, sparse_simple_fwd_ckpt = benchmark_forward(
        sparse_simple_fn,
        sparse_simple_checkpointed_fn,
        warmup,
        iters,
    )
    sparse_simple_bwd, sparse_simple_bwd_ckpt = benchmark_backward(
        (sparse_query, key, value),
        sparse_grad,
        sparse_simple_fn,
        sparse_simple_checkpointed_fn,
        warmup,
        iters,
    )
    sparse_bilinear_fwd, sparse_bilinear_fwd_ckpt = benchmark_forward(
        sparse_bilinear_fn,
        sparse_bilinear_checkpointed_fn,
        warmup,
        iters,
    )
    sparse_bilinear_bwd, sparse_bilinear_bwd_ckpt = benchmark_backward(
        (sparse_query, key, value),
        sparse_grad,
        sparse_bilinear_fn,
        sparse_bilinear_checkpointed_fn,
        warmup,
        iters,
    )

    kh, kw = case.kernel_size
    print(
        f"\ncase={case.name} dtype={str(dtype).replace('torch.', '')} "
        f"B={case.batch} HxW={case.height}x{case.width} N={case.num_queries or case.height * case.width} "
        f"heads={case.heads} D={case.dim} Dv={case.dim_value} K={kh}x{kw} "
        f"scale={case.scale or case.dim ** -0.5:.6g} backend={backend or 'auto'} sparse_sample={sample_mode}"
    )
    print("module                              fwd ms          bwd ms")
    print("----------------------------------------------------------")
    print(f"neighborhood_attention_generic       {dense_fwd[0]:8.3f} +/- {dense_fwd[1]:6.3f}  {dense_bwd[0]:8.3f} +/- {dense_bwd[1]:6.3f}")
    print(f"neighborhood_attention_generic ckpt  {dense_fwd_ckpt[0]:8.3f} +/- {dense_fwd_ckpt[1]:6.3f}  {dense_bwd_ckpt[0]:8.3f} +/- {dense_bwd_ckpt[1]:6.3f}")
    print(f"natten_sparse_na2d                   {sparse_fwd[0]:8.3f} +/- {sparse_fwd[1]:6.3f}  {sparse_bwd[0]:8.3f} +/- {sparse_bwd[1]:6.3f}")
    print(f"natten_sparse_na2d ckpt              {sparse_fwd_ckpt[0]:8.3f} +/- {sparse_fwd_ckpt[1]:6.3f}  {sparse_bwd_ckpt[0]:8.3f} +/- {sparse_bwd_ckpt[1]:6.3f}")
    print(f"natten_sparse_na2d_simple            {sparse_simple_fwd[0]:8.3f} +/- {sparse_simple_fwd[1]:6.3f}  {sparse_simple_bwd[0]:8.3f} +/- {sparse_simple_bwd[1]:6.3f}")
    print(f"natten_sparse_na2d_simple ckpt       {sparse_simple_fwd_ckpt[0]:8.3f} +/- {sparse_simple_fwd_ckpt[1]:6.3f}  {sparse_simple_bwd_ckpt[0]:8.3f} +/- {sparse_simple_bwd_ckpt[1]:6.3f}")
    print(f"natten_sparse_na2d_bilinear          {sparse_bilinear_fwd[0]:8.3f} +/- {sparse_bilinear_fwd[1]:6.3f}  {sparse_bilinear_bwd[0]:8.3f} +/- {sparse_bilinear_bwd[1]:6.3f}")
    print(f"natten_sparse_na2d_bilinear ckpt     {sparse_bilinear_fwd_ckpt[0]:8.3f} +/- {sparse_bilinear_fwd_ckpt[1]:6.3f}  {sparse_bilinear_bwd_ckpt[0]:8.3f} +/- {sparse_bilinear_bwd_ckpt[1]:6.3f}")

    del query, sparse_query, key, value, coords, dense_grad, sparse_grad
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense 2D neighborhood_attention_generic and natten.sparse_na2d with/without checkpoint."
    )
    parser.add_argument("--case", choices=[case.name for case in CASES] + ["all"], default="all")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--backend", default=None, help="Optional dense NA backend, e.g. cutlass-fna, hopper-fna, flex-fna.")
    parser.add_argument("--sample-mode", choices=["indexed", "bilinear"], default="indexed")
    parser.add_argument("--use-reentrant", action="store_true", help="Use reentrant torch checkpointing.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not natten.HAS_LIBNATTEN:
        raise RuntimeError("libnatten is not available; build with `python setup.py build_ext --inplace`.")

    torch.manual_seed(42)
    dtype = parse_dtype(args.dtype)
    cases = CASES if args.case == "all" else [case for case in CASES if case.name == args.case]
    for case in cases:
        benchmark_case(
            case=case,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            backend=args.backend,
            sample_mode=args.sample_mode,
            use_reentrant=args.use_reentrant,
        )


if __name__ == "__main__":
    main()
