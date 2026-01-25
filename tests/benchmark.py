import torch
import numpy as np
import zigtorch
import time
import argparse

def generate_matrices(m, n, k, dtype=torch.float32, seed=42):
    """Generate random matrices for testing."""
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=dtype)
    B = torch.randn(k, n, dtype=dtype)
    return A, B

def benchmark_mm(A, B, iterations=10, warmup=3):
    """Benchmark matrix multiplication with both PyTorch and ZigTorch."""
    # Warmup
    for _ in range(warmup):
        _ = torch.mm(A, B)
        _ = zigtorch.mm(A, B)
    
    # PyTorch benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(iterations):
        C_torch = torch.mm(A, B)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch_time = (time.time() - start_time) / iterations
    
    # ZigTorch benchmark
    start_time = time.time()
    for _ in range(iterations):
        C_zig = zigtorch.mm(A, B)
    zig_time = (time.time() - start_time) / iterations
    
    # Verify results
    max_diff = torch.max(torch.abs(C_torch - C_zig)).item()
    
    return {
        "pytorch_time": torch_time,
        "zigtorch_time": zig_time,
        "speedup": torch_time / zig_time if zig_time > 0 else float('inf'),
        "max_diff": max_diff,
        "correct": max_diff < 1e-4  # Allow small floating-point differences
    }

def run_benchmark_suite():
    """Run a comprehensive benchmark suite."""
    print("=" * 80)
    print("ZigTorch Matrix Multiplication Benchmark")
    print("=" * 80)
    
    matrix_sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        # Non-square matrices
        (1024, 512, 256),
        (2048, 512, 1024),
    ]
    
    print(f"{'Size M×K × K×N':<20} {'PyTorch (ms)':<15} {'ZigTorch (ms)':<15} {'Speedup':<10} {'Max Diff':<10} {'Correct'}")
    print("-" * 80)
    
    for m, k, n in matrix_sizes:
        try:
            A, B = generate_matrices(m, k, n)
            result = benchmark_mm(A, B)
            
            # Print results
            size_str = f"{m}×{k} × {k}×{n}"
            torch_ms = result["pytorch_time"] * 1000
            zig_ms = result["zigtorch_time"] * 1000
            
            print(f"{size_str:<20} {torch_ms:<15.3f} {zig_ms:<15.3f} {result['speedup']:<10.3f} {result['max_diff']:<10.3e} {result['correct']}")
        except RuntimeError as e:
            print(f"{m}×{k} × {k}×{n}: Error - {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ZigTorch matrix multiplication")
    parser.add_argument("--custom", action="store_true", help="Run a custom size benchmark")
    parser.add_argument("--m", type=int, default=1024, help="M dimension")
    parser.add_argument("--n", type=int, default=1024, help="N dimension")
    parser.add_argument("--k", type=int, default=1024, help="K dimension")
    args = parser.parse_args()
    
    if args.custom:
        A, B = generate_matrices(args.m, args.n, args.k)
        result = benchmark_mm(A, B, iterations=5)
        
        print(f"Custom benchmark for {args.m}×{args.k} × {args.k}×{args.n}:")
        print(f"PyTorch: {result['pytorch_time']*1000:.3f} ms")
        print(f"ZigTorch: {result['zigtorch_time']*1000:.3f} ms")
        print(f"Speedup: {result['speedup']:.3f}x")
        print(f"Max difference: {result['max_diff']:.3e}")
        print(f"Results match: {result['correct']}")
    else:
        run_benchmark_suite()
