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

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

import natten
from natten.sparse_na2d_reference import sparse_na2d_pytorch


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    num_queries: int
    height: int
    width: int
    heads: int
    dim: int
    dim_value: int
    kernel_size: Tuple[int, int]


CASES = [
    Case("tiny", 1, 1024, 37, 37, 4, 32, 768, (5, 5)),
    Case("small", 1, 2048, 37, 37, 4, 32, 768, (5, 5)),
    Case("medium", 1, 8192, 37, 37, 4, 32, 768, (5, 5)),
]


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def make_inputs(case: Case, dtype: torch.dtype, requires_grad: bool):
    query = torch.randn(
        case.batch,
        case.num_queries,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    coords = torch.rand(case.batch, case.num_queries, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    return query, key, value, coords


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
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def check_correctness(case: Case, dtype: torch.dtype, check_backward: bool, sample_mode: str, apply_key_rope: bool) -> None:
    query, key, value, coords = make_inputs(case, dtype, requires_grad=check_backward)
    actual = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        case.kernel_size,
        sample_mode=sample_mode,
        apply_key_rope=apply_key_rope,
    )
    expected = sparse_na2d_pytorch(
        query,
        key,
        value,
        coords,
        case.kernel_size,
        sample_mode=sample_mode,
        apply_key_rope=apply_key_rope,
    )

    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    elif check_backward and sample_mode == "bilinear":
        # Bilinear half-precision backward combines interpolation weights with
        # atomic accumulation, so rare single-element outliers are a bit larger
        # than the indexed path while still matching fp32 checks tightly.
        rtol, atol = 2e-1, 2e-1
    else:
        rtol, atol = 6e-2, 6e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    if not check_backward:
        return

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())
    query.grad = None
    key.grad = None
    value.grad = None
    expected.backward(grad)
    for actual_grad, expected_grad in zip(actual_grads, (query.grad, key.grad, value.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)


def benchmark_case(
    case: Case,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    backward: bool,
    sample_mode: str,
    apply_key_rope: bool,
) -> None:
    query, key, value, coords = make_inputs(case, dtype, requires_grad=backward)
    grad = torch.randn(
        case.batch,
        case.num_queries,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
    )

    def run_kernel_forward():
        natten.sparse_na2d(
            query,
            key,
            value,
            coords,
            case.kernel_size,
            sample_mode=sample_mode,
            apply_key_rope=apply_key_rope,
        )

    def run_naive_forward():
        sparse_na2d_pytorch(
            query,
            key,
            value,
            coords,
            case.kernel_size,
            sample_mode=sample_mode,
            apply_key_rope=apply_key_rope,
        )

    def run_kernel_backward():
        for tensor in (query, key, value):
            tensor.grad = None
        out = natten.sparse_na2d(
            query,
            key,
            value,
            coords,
            case.kernel_size,
            sample_mode=sample_mode,
            apply_key_rope=apply_key_rope,
        )
        out.backward(grad)

    def run_naive_backward():
        for tensor in (query, key, value):
            tensor.grad = None
        out = sparse_na2d_pytorch(
            query,
            key,
            value,
            coords,
            case.kernel_size,
            sample_mode=sample_mode,
            apply_key_rope=apply_key_rope,
        )
        out.backward(grad)

    kernel_ms, kernel_std = time_cuda(run_kernel_backward if backward else run_kernel_forward, warmup, iters)
    naive_ms, naive_std = time_cuda(run_naive_backward if backward else run_naive_forward, warmup, iters)
    speedup = naive_ms / kernel_ms
    mode = "fwd+bwd" if backward else "fwd"
    kh, kw = case.kernel_size
    print(
        f"{case.name:>8} {mode:>7} sample={sample_mode:<8} key_rope={str(apply_key_rope):<5} "
        f"dtype={str(dtype).replace('torch.', ''):<8} "
        f"B={case.batch} N={case.num_queries} HxW={case.height}x{case.width} "
        f"heads={case.heads} D={case.dim} Dv={case.dim_value} K={kh}x{kw}: "
        f"kernel={kernel_ms:.3f}+/-{kernel_std:.3f} ms "
        f"naive={naive_ms:.3f}+/-{naive_std:.3f} ms "
        f"speedup={speedup:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse_na2d CUDA against naive PyTorch.")
    parser.add_argument("--case", choices=[case.name for case in CASES] + ["all"], default="all")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sample-mode", choices=["indexed", "bilinear"], default="indexed")
    parser.add_argument("--apply-key-rope", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not natten.HAS_LIBNATTEN:
        raise RuntimeError("libnatten is not available; build with `python setup.py build_ext --inplace`.")

    dtype = parse_dtype(args.dtype)
    cases = CASES if args.case == "all" else [case for case in CASES if case.name == args.case]
    for case in cases:
        if args.check:
            check_correctness(
                case,
                dtype,
                check_backward=args.backward,
                sample_mode=args.sample_mode,
                apply_key_rope=args.apply_key_rope,
            )
        benchmark_case(case, dtype, args.warmup, args.iters, args.backward, args.sample_mode, args.apply_key_rope)


if __name__ == "__main__":
    main()
