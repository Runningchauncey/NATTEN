#!/usr/bin/env python

import argparse
import gc
import json
import statistics
import subprocess
import sys
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

import natten
from natten.sparse_na2d_reference import sparse_na2d_bilinear_query_neighbor_pytorch


IMPLEMENTATIONS = ("query_neighbor", "sparse_na2d", "sparse_na2d_bilinear", "materialized")


def parse_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_rope_freqs(channels: int, device: torch.device) -> torch.Tensor:
    base = 100.0 ** torch.linspace(0, -1, channels // 4, device=device)
    base = torch.cat((base, base)) * (2 * torch.pi)
    freqs = torch.zeros(2, channels, device=device)
    freqs[0, : channels // 2] = base
    freqs[1, channels // 2 :] = base
    return freqs


def make_inputs(args: argparse.Namespace):
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    query = torch.randn(args.batch, args.num_queries, args.heads, args.dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(args.batch, args.height, args.width, args.heads, args.dim, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(
        args.batch, args.height, args.width, args.heads, args.dim_value,
        device=device, dtype=dtype, requires_grad=True,
    )
    coords = torch.rand(args.batch, args.num_queries, 2, device=device, dtype=torch.float32) * 2 - 1
    q_weight = torch.ones(args.heads * args.dim, device=device, dtype=torch.float32, requires_grad=True)
    k_weight = torch.ones(args.heads * args.dim, device=device, dtype=torch.float32, requires_grad=True)
    rope_freqs = make_rope_freqs(args.heads * args.dim, device).requires_grad_(True)
    grad = torch.randn(
        args.batch, args.num_queries, args.heads, args.dim_value,
        device=device, dtype=dtype,
    )
    return (query, key, value, coords, q_weight, k_weight, rope_freqs), grad


def make_function(args: argparse.Namespace) -> Callable[..., torch.Tensor]:
    kernel_size = (args.kernel_height, args.kernel_width)
    query_resolution = (args.query_height, args.query_width)
    scale = args.dim**-0.5

    if args.implementation == "query_neighbor":
        return lambda q, k, v, c, qw, kw, rf: natten.sparse_na2d_bilinear_query_neighbor(
            q, k, v, c, kernel_size, qw, kw, rf,
            query_resolution=query_resolution, scale=scale,
        )
    if args.implementation == "materialized":
        return lambda q, k, v, c, qw, kw, rf: sparse_na2d_bilinear_query_neighbor_pytorch(
            q, k, v, c, kernel_size, qw, kw, rf,
            query_resolution=query_resolution, scale=scale,
        )
    if args.implementation == "sparse_na2d":
        return lambda q, k, v, c, qw, kw, rf: natten.sparse_na2d(
            q, k, v, c, kernel_size, scale=scale, sample_mode="bilinear",
            apply_query_rope=True, apply_key_rope=True, apply_qk_norm=True,
            q_norm_weight=qw, k_norm_weight=kw,
        )
    if args.implementation == "sparse_na2d_bilinear":
        return lambda q, k, v, c, qw, kw, rf: natten.sparse_na2d_bilinear(
            q, k, v, c, kernel_size, scale=scale,
        )
    raise ValueError(args.implementation)


def clear_grads(inputs) -> None:
    for tensor in inputs:
        if tensor.requires_grad:
            tensor.grad = None


def timed_samples(operation: Callable[[], None], warmup: int, iterations: int):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def timed_backward(
    make_output: Callable[[], torch.Tensor],
    inputs,
    grad: torch.Tensor,
    warmup: int,
    iterations: int,
):
    samples = []
    for index in range(warmup + iterations):
        clear_grads(inputs)
        output = make_output()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if index >= warmup:
            start.record()
        output.backward(grad)
        if index >= warmup:
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def memory_sample(operation: Callable[[], None], baseline: int):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    return {
        "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "allocated_extra_mib": max(0, torch.cuda.max_memory_allocated() - baseline) / 2**20,
        "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def run_child(args: argparse.Namespace) -> dict:
    inputs, grad = make_inputs(args)
    fn = make_function(args)

    def plain_forward():
        clear_grads(inputs)
        output = fn(*inputs)
        del output

    def plain_backward():
        clear_grads(inputs)
        output = fn(*inputs)
        torch.cuda.synchronize()
        output.backward(grad)

    def checkpoint_forward():
        clear_grads(inputs)
        output = checkpoint(fn, *inputs, use_reentrant=False, preserve_rng_state=False)
        del output

    def checkpoint_backward():
        clear_grads(inputs)
        output = checkpoint(fn, *inputs, use_reentrant=False, preserve_rng_state=False)
        torch.cuda.synchronize()
        output.backward(grad)

    forward = timed_samples(plain_forward, args.warmup, args.iterations)
    backward = timed_backward(lambda: fn(*inputs), inputs, grad, args.warmup, args.iterations)
    checkpointed_forward = timed_samples(checkpoint_forward, args.warmup, args.iterations)
    checkpointed_backward = timed_backward(
        lambda: checkpoint(fn, *inputs, use_reentrant=False, preserve_rng_state=False),
        inputs,
        grad,
        args.warmup,
        args.iterations,
    )

    clear_grads(inputs)
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    def inference_memory():
        with torch.no_grad():
            output = fn(*inputs)
            del output

    def training_forward_memory():
        output = fn(*inputs)
        torch.cuda.synchronize()
        del output

    def backward_memory():
        clear_grads(inputs)
        output = fn(*inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        output.backward(grad)

    memory = {
        "inference_forward": memory_sample(inference_memory, baseline),
        "training_forward": memory_sample(training_forward_memory, baseline),
        "forward_backward": memory_sample(plain_backward, baseline),
        "checkpoint_forward_backward": memory_sample(checkpoint_backward, baseline),
    }
    backward_peak = backward_memory
    memory["backward"] = memory_sample(backward_peak, baseline)

    return {
        "implementation": args.implementation,
        "dtype": args.dtype,
        "forward_ms": forward,
        "backward_ms": backward,
        "total_ms": [forward[0] + backward[0], (forward[1] ** 2 + backward[1] ** 2) ** 0.5],
        "checkpoint_forward_ms": checkpointed_forward,
        "checkpoint_backward_ms": checkpointed_backward,
        "checkpoint_total_ms": [
            checkpointed_forward[0] + checkpointed_backward[0],
            (checkpointed_forward[1] ** 2 + checkpointed_backward[1] ** 2) ** 0.5,
        ],
        "memory": memory,
    }


def print_tables(results: list[dict]) -> None:
    print("\nRuntime (ms, mean +/- std)")
    print(f"{'implementation':30s} {'fwd':>17s} {'bwd':>17s} {'total':>17s} {'ckpt fwd':>17s} {'ckpt bwd':>17s} {'ckpt total':>17s}")
    for result in results:
        values = [result[name] for name in (
            "forward_ms", "backward_ms", "total_ms", "checkpoint_forward_ms",
            "checkpoint_backward_ms", "checkpoint_total_ms",
        )]
        formatted = [f"{mean:7.2f} +/- {std:6.2f}" for mean, std in values]
        print(f"{result['implementation']:30s} " + " ".join(f"{value:>17s}" for value in formatted))

    print("\nPeak allocated VRAM (MiB; extra above inputs in parentheses)")
    phases = ("inference_forward", "training_forward", "backward", "forward_backward", "checkpoint_forward_backward")
    print(f"{'implementation':30s} " + " ".join(f"{phase:>27s}" for phase in phases))
    for result in results:
        cells = []
        for phase in phases:
            sample = result["memory"][phase]
            cells.append(f"{sample['allocated_mib']:9.1f} ({sample['allocated_extra_mib']:8.1f})")
        print(f"{result['implementation']:30s} " + " ".join(f"{cell:>27s}" for cell in cells))

    print("\nPeak reserved VRAM (MiB)")
    print(f"{'implementation':30s} " + " ".join(f"{phase:>27s}" for phase in phases))
    for result in results:
        cells = [f"{result['memory'][phase]['reserved_mib']:9.1f}" for phase in phases]
        print(f"{result['implementation']:30s} " + " ".join(f"{cell:>27s}" for cell in cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--num-queries", type=int, default=4096)
    parser.add_argument("--height", type=int, default=37)
    parser.add_argument("--width", type=int, default=37)
    parser.add_argument("--query-height", type=int, default=259)
    parser.add_argument("--query-width", type=int, default=259)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dim-value", type=int, default=192)
    parser.add_argument("--kernel-height", type=int, default=5)
    parser.add_argument("--kernel-width", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.implementation:
        print("RESULT_JSON=" + json.dumps(run_child(args)), flush=True)
        return

    results = []
    for implementation in IMPLEMENTATIONS:
        command = [
            sys.executable,
            __file__,
            "--implementation",
            implementation,
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--batch",
            str(args.batch),
            "--num-queries",
            str(args.num_queries),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--query-height",
            str(args.query_height),
            "--query-width",
            str(args.query_width),
            "--heads",
            str(args.heads),
            "--dim",
            str(args.dim),
            "--dim-value",
            str(args.dim_value),
            "--kernel-height",
            str(args.kernel_height),
            "--kernel-width",
            str(args.kernel_width),
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON="))
        results.append(json.loads(line.removeprefix("RESULT_JSON=")))
    print_tables(results)


if __name__ == "__main__":
    main()
