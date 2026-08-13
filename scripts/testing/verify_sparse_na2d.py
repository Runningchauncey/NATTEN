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
from dataclasses import dataclass
from typing import Tuple

import torch

import natten
from natten.sparse_na2d_reference import sparse_na2d_pytorch


@dataclass(frozen=True)
class VerificationCase:
    batch: int
    num_queries: int
    height: int
    width: int
    heads: int
    dim: int
    dim_value: int
    kernel_size: Tuple[int, int]


CASES = {
    "micro": VerificationCase(1, 11, 5, 6, 4, 8, 13, (3, 3)),
    "tiny": VerificationCase(1, 128, 13, 11, 4, 16, 64, (5, 5)),
    "realish": VerificationCase(1, 1024, 37, 37, 4, 32, 768, (5, 5)),
}


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}.")


def tolerances(dtype: torch.dtype, backward: bool, sample_mode: str) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if backward and sample_mode == "bilinear":
        return 2e-1, 2e-1
    return 6e-2, 6e-2


def make_inputs(case: VerificationCase, dtype: torch.dtype, backward: bool):
    query = torch.randn(
        case.batch,
        case.num_queries,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=backward,
    )
    key = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim,
        device="cuda",
        dtype=dtype,
        requires_grad=backward,
    )
    value = torch.randn(
        case.batch,
        case.height,
        case.width,
        case.heads,
        case.dim_value,
        device="cuda",
        dtype=dtype,
        requires_grad=backward,
    )

    coords = torch.rand(case.batch, case.num_queries, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    if case.num_queries >= 4:
        boundary = torch.tensor(
            [[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [0.93, -0.87]]],
            device="cuda",
            dtype=torch.float32,
        )
        coords[:, :4] = boundary
    return query, key, value, coords


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    diff = (actual - expected).abs()
    abs_err = diff.max().item()
    rel_err = (diff / expected.abs().clamp_min(1e-12)).max().item()
    return abs_err, rel_err


def verify(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to verify natten.sparse_na2d.")
    if not natten.HAS_LIBNATTEN:
        raise RuntimeError("libnatten is not available. Build with `python setup.py build_ext --inplace`.")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    case = CASES[args.case]

    query, key, value, coords = make_inputs(case, dtype, backward=args.backward)
    actual = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        case.kernel_size,
        sample_mode=args.sample_mode,
        apply_query_rope=args.apply_query_rope,
        apply_key_rope=args.apply_key_rope,
        apply_qk_norm=args.apply_qk_norm,
        qk_norm_before_rope=not args.rope_before_qk_norm,
    )
    expected = sparse_na2d_pytorch(
        query,
        key,
        value,
        coords,
        case.kernel_size,
        sample_mode=args.sample_mode,
        apply_query_rope=args.apply_query_rope,
        apply_key_rope=args.apply_key_rope,
        apply_qk_norm=args.apply_qk_norm,
        qk_norm_before_rope=not args.rope_before_qk_norm,
    )

    rtol, atol = tolerances(dtype, args.backward, args.sample_mode)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    out_abs, out_rel = max_error(actual.float(), expected.float())
    print(f"forward ok: max_abs={out_abs:.6g} max_rel={out_rel:.6g} rtol={rtol} atol={atol}")

    if not args.backward:
        return

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())
    query.grad = None
    key.grad = None
    value.grad = None

    expected.backward(grad)
    grad_names = ("query", "key", "value")
    for name, actual_grad, expected_grad in zip(grad_names, actual_grads, (query.grad, key.grad, value.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)
        grad_abs, grad_rel = max_error(actual_grad.float(), expected_grad.float())
        print(f"{name:>5} grad ok: max_abs={grad_abs:.6g} max_rel={grad_rel:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify sparse_na2d CUDA numerics against naive PyTorch.")
    parser.add_argument("--case", choices=CASES.keys(), default="micro")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--sample-mode", choices=["indexed", "bilinear"], default="bilinear")
    parser.add_argument("--apply-query-rope", action="store_true")
    parser.add_argument("--apply-key-rope", action="store_true")
    parser.add_argument("--apply-qk-norm", action="store_true")
    parser.add_argument(
        "--rope-before-qk-norm",
        action="store_true",
        help="Use sample -> RoPE -> QK-Norm -> dot instead of the default sample -> QK-Norm -> RoPE -> dot.",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()
