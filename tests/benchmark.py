import sys
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


# try:
#     import torch
# except ImportError:
torch = None
#
import zigtorch as zt


def generate_matrices(m, n, k, dtype=np.float32, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, k), dtype=dtype)
    B = rng.standard_normal((k, n), dtype=dtype)
    return A, B


def benchmark_mm(A_np, B_np, iterations=10, warmup=3):
    # Warmup for Zig
    for _ in range(warmup):
        _ = zt.matrix_multiply(A_np, B_np)

    # Benchmark Zig
    start_time = time.perf_counter()
    for _ in range(iterations):
        C_zig = zt.matrix_multiply(A_np, B_np)
    zig_time = (time.perf_counter() - start_time) / iterations

    # Benchmark NumPy
    for _ in range(warmup):
        _ = np.matmul(A_np, B_np)
    start_time = time.perf_counter()
    for _ in range(iterations):
        C_numpy = np.matmul(A_np, B_np)
    numpy_time = (time.perf_counter() - start_time) / iterations

    # Check correctness against NumPy
    numpy_diff = float(np.max(np.abs(C_numpy - C_zig)))

    # Benchmark PyTorch if available
    if torch is not None:
        A_t = torch.from_numpy(A_np)
        B_t = torch.from_numpy(B_np)
        for _ in range(warmup):
            _ = torch.mm(A_t, B_t)
        start_time = time.perf_counter()
        for _ in range(iterations):
            C_torch = torch.mm(A_t, B_t)
        torch_time = (time.perf_counter() - start_time) / iterations
        torch_diff = float(torch.max(torch.abs(C_torch - torch.from_numpy(C_zig))).item())
    else:
        torch_time = None
        torch_diff = None

    return {
        "torch_time": torch_time,
        "numpy_time": numpy_time,
        "zig_time": zig_time,
        "torch_diff": torch_diff,
        "numpy_diff": numpy_diff,
        "correct": numpy_diff < 1e-4,
    }


def run_benchmark_suite():
    sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        # (2048, 2048, 2048),
        (1024, 512, 256),
        # (2048, 512, 1024),
    ]

    header = f"{'Size M×K × K×N':<22} {'Torch (ms)':<12} {'NumPy (ms)':<12} {'Zig (ms)':<12} {'Zig vs Torch':<14} {'Zig vs NumPy':<14} {'Correct'}"
    print(header)
    print("-" * len(header))
    for m, k, n in sizes:
        A, B = generate_matrices(m, n, k)
        res = benchmark_mm(A, B)
        torch_ms = res["torch_time"] * 1000 if res["torch_time"] is not None else None
        numpy_ms = res["numpy_time"] * 1000
        zig_ms = res["zig_time"] * 1000

        # Calculate speedups
        zig_vs_torch = torch_ms / zig_ms if torch_ms else None
        zig_vs_numpy = numpy_ms / zig_ms

        print(
            f"{m}×{k} × {k}×{n:<6} "
            f"{(f'{torch_ms:.3f}' if torch_ms else 'n/a'):>12} "
            f"{numpy_ms:>12.3f} "
            f"{zig_ms:>12.3f} "
            f"{(f'{zig_vs_torch:.2f}x' if zig_vs_torch else 'n/a'):>14} "
            f"{f'{zig_vs_numpy:.2f}x':>14} "
            f"{res['correct']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ZigTorch matrix multiplication")
    parser.add_argument("--custom", action="store_true", help="Run a custom size benchmark")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args = parser.parse_args()

    if args.custom:
        A, B = generate_matrices(args.m, args.n, args.k)
        res = benchmark_mm(A, B, iterations=5)
        print(f"{args.m}×{args.k} × {args.k}×{args.n}")
        print(f"ZigTorch: {res['zig_time']*1000:.3f} ms")
        print(f"NumPy:    {res['numpy_time']*1000:.3f} ms")
        if res["torch_time"] is not None:
            print(f"Torch:    {res['torch_time']*1000:.3f} ms")
        print(f"Zig vs NumPy: {res['numpy_time']/res['zig_time']:.2f}x")
        if res["torch_time"] is not None:
            print(f"Zig vs Torch: {res['torch_time']/res['zig_time']:.2f}x")
        print(f"Correct: {res['correct']}")
    else:
        run_benchmark_suite()
