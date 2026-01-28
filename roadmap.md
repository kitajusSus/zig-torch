# 🗺️ ZigTorch Development Roadmap

**Project:** testing-ZIG-torch
**Repository:** https://github.com/kitajusSus/testing-ZIG-torch
**Author:** kitajusSus
**Last Updated:** 2026-01-26
**Zig Version:** 0.13.0+

---

## 📑 Table of Contents

1. [Project Vision](#project-vision)
2. [Current State Analysis](#current-state-analysis)
3. [Phase 1: Foundation & Correctness](#phase-1-foundation--correctness-weeks-1-4)
4. [Phase 2: Performance Engineering](#phase-2-performance-engineering-weeks-5-8)
5. [Phase 3: API & Usability](#phase-3-api--usability-weeks-9-12)
6. [Phase 4: Advanced Features](#phase-4-advanced-features-weeks-13-20)
7. [Phase 5: Production Ready](#phase-5-production-ready-weeks-21-24)
8. [Long-term Vision](#long-term-vision)
9. [Learning Resources](#learning-resources)
10. [Success Metrics](#success-metrics)

---

## Project Vision

### Goal
Build a high-performance tensor computation library in Zig that can:
- Serve as a learning project for systems programming and numerical computing
- Compete with NumPy/OpenBLAS for CPU-bound workloads
- Provide Python bindings with minimal overhead
- Demonstrate Zig's capabilities for scientific computing

### Non-Goals (at least initially)
- Replace PyTorch/TensorFlow for production ML
- Support every deep learning operation
- Beat MKL/cuBLAS in all scenarios

### Inspiration Projects
- **BLIS** - https://github.com/flame/blis
  *Reference for blocked matrix multiplication algorithms*
- **Eigen** - https://gitlab.com/libeigen/eigen
  *Template-based linear algebra, excellent cache optimization*
- **xtensor** - https://github.com/xtensor-stack/xtensor
  *NumPy-style API in C++, good API design patterns*
- **NumPy** - https://github.com/numpy/numpy
  *Gold standard for scientific Python API*
- **PyTorch** - https://github.com/pytorch/pytorch
  *Modern tensor library design, autograd architecture*
- **tinygrad** - https://github.com/tinygrad/tinygrad
  *Minimalist ML framework, shows how little you need for real ML*
- **Halide** - https://github.com/halide/Halide
  *Auto-tuning and schedule optimization techniques*
- **OpenBLAS** - https://github.com/xianyi/OpenBLAS
  *Production-grade BLAS implementation*

---

## Current State Analysis

### ✅ Strengths
1. **Core Algorithm**: Blocked matrix multiplication with SIMD vectorization
2. **Hardware Adaptation**: AVX-512, AVX2 detection and fallback paths
3. **Multi-threading**: Worker pool with barrier synchronization
4. **Prefetching**: Manual cache management for L1/L2
5. **Python Bindings**: Working ctypes interface

### ⚠️ Weaknesses
1. **Boundary Conditions**: Edge cases not handled (line 203 in mm.zig)
2. **Memory Layout**: Inconsistent stride usage (lda/ldb vs hardcoded)
3. **Testing**: No comprehensive test suite for correctness
4. **Error Handling**: Silent failures, no validation
5. **Documentation**: Minimal inline comments, no API docs
6. **Single Operation**: Only matrix multiply implemented
7. **Type Support**: Only f32, no f64/f16/int8

### 🐛 Known Bugs
| Issue | Location | Severity | Fix Effort |
|-------|----------|----------|------------|
| Partial block overflow | `mm.zig:203` | High | Medium |
| Integer overflow in size calculation | `mm.zig:340` | High | Low |
| Race condition in global initialization | `mm.zig:7,41` | Medium | Medium |
| Missing alignment guarantees | `mm.zig:329` | Medium | Low |
| Thread spawn error handling | `mm.zig:391` | Low | Low |

---

## Phase 1: Foundation & Correctness (Weeks 1-4)

**Goal:** Fix bugs, establish testing infrastructure, ensure algorithm correctness

### Week 1: Bug Fixes & Core Stability

#### Task 1.1: Fix Boundary Conditions
**Priority:** 🔴 Critical
**Effort:** 6 hours

**Current Problem:**
```zig
// mm.zig:203 - Calls microKernel16x16 even when block_k < 16
while (kk < block_k) : (kk += MICRO_M) {
    microKernel16x16(A, B, C, i + ii, j + jj, k + kk, lda, ldb, ldc);
}
```

**Solution Approach:**
```zig
while (kk < block_k) : (kk += MICRO_M) {
    const actual_k = @min(MICRO_M, block_k - kk);

    if (actual_k == MICRO_M and block_m == MICRO_M and block_n == MICRO_N) {
        // Fast path: full 16x16x16 block
        microKernel16x16(A, B, C, i + ii, j + jj, k + kk, lda, ldb, ldc);
    } else {
        // Slow path: partial block
        microKernelPartial(A, B, C, i + ii, j + jj, k + kk,
                          @min(MICRO_M, block_m - ii),
                          @min(MICRO_N, block_n - jj),
                          actual_k, lda, ldb, ldc);
    }
}
```

**Reference Implementation:**
- BLIS micro-kernel handling: https://github.com/flame/blis/blob/master/kernels/zen/3/bli_gemm_zen_asm_d6x8.c#L200-L250
- Eigen block size logic: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/products/GeneralBlockPanelKernel.h#L1850

**Testing:**
```zig
test "matrix multiply boundary - odd sizes" {
    const sizes = [_]usize{ 1, 7, 15, 17, 31, 33, 63, 65 };
    for (sizes) |m| {
        for (sizes) |n| {
            for (sizes) |k| {
                // Test m×k @ k×n
            }
        }
    }
}
```

---

#### Task 1.2: Integer Overflow Protection
**Priority:** 🔴 Critical
**Effort:** 2 hours

**Current Problem:**
```zig
// mm.zig:340 - Can overflow for large matrices (e.g., 10000³ > u32::MAX)
const total_elements = M * N * K;
```

**Solution:**
```zig
pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    // Validate dimensions
    if (M == 0 or N == 0 or K == 0) return;

    // Check for overflow using u64
    const m64: u64 = @intCast(M);
    const n64: u64 = @intCast(N);
    const k64: u64 = @intCast(K);

    var total_ops: u64 = undefined;
    if (@mulWithOverflow(&total_ops, m64 *% n64, k64)) {
        // Overflow occurred - dimensions too large
        std.debug.print("Error: Matrix dimensions too large (overflow)\n", .{});
        return;
    }

    // Rest of implementation...
}
```

**Reference:**
- Rust overflow handling: https://doc.rust-lang.org/std/primitive.usize.html#method.checked_mul
- BLIS dimension validation: https://github.com/flame/blis/blob/master/frame/3/bli_l3_check.c

---

#### Task 1.3: Thread-Safe Global Initialization
**Priority:** 🟡 High
**Effort:** 4 hours

**Current Problem:**
```zig
// mm.zig:7 - Mutable globals modified during initialization
var BLOCK_M: usize = 256;
// Later in determineOptimalBlockSize():
BLOCK_M = 384; // Race condition if multiple threads call zig_mm
```

**Solution:**
```zig
// Define as compile-time or thread-local
const Config = struct {
    block_m: usize,
    block_n: usize,
    block_k: usize,
    thread_count: usize,
};

threadlocal var cached_config: ?Config = null;

fn getConfig() Config {
    if (cached_config) |config| return config;

    // Initialize once per thread
    const cpu_count = Thread.getCpuCount() catch 4;
    const thread_count = @min(cpu_count, 32);

    var block_m: usize = 256;
    var block_n: usize = 256;
    var block_k: usize = 256;

    // CPU-specific tuning
    if ((builtin.cpu.arch == .x86_64) and
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
        block_m = 384;
        block_n = 384;
        block_k = 384;
    }

    cached_config = Config{
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .thread_count = thread_count,
    };

    return cached_config.?;
}
```

**Reference:**
- Zig thread-local storage: https://ziglang.org/documentation/master/#threadlocal
- NumPy thread-safe config: https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/multiarraymodule.c#L4500

---

### Week 2: Test Infrastructure

#### Task 2.1: Unit Test Framework
**Priority:** 🔴 Critical
**Effort:** 12 hours

**Goal:** Comprehensive correctness tests for all matrix sizes and edge cases

**Test Categories:**

1. **Dimension Tests**
```zig
// tests/test_mm_dimensions.zig
const std = @import("std");
const mm = @import("../src/ops/mm.zig");
const testing = std.testing;

test "square matrices - powers of 2" {
    const sizes = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    for (sizes) |size| {
        const A = try allocator.alloc(f32, size * size);
        const B = try allocator.alloc(f32, size * size);
        const C = try allocator.alloc(f32, size * size);
        const expected = try allocator.alloc(f32, size * size);

        // Fill with known pattern
        fillSequential(A, size * size);
        fillSequential(B, size * size);

        // Reference implementation (naive triple loop)
        naiveMatMul(A, B, expected, size, size, size);

        // Test implementation
        mm.zig_mm(A.ptr, B.ptr, C.ptr, size, size, size);

        // Compare with tolerance
        try expectMatricesEqual(expected, C, 1e-4);
    }
}

test "square matrices - odd sizes" {
    const sizes = [_]usize{ 1, 3, 7, 15, 17, 31, 33, 63, 65, 127, 129 };
    // Same as above...
}

test "rectangular matrices - various shapes" {
    const cases = [_][3]usize{
        .{ 1, 1, 1 },       // Minimal
        .{ 1, 100, 1 },     // Row vector × matrix
        .{ 100, 1, 100 },   // Matrix × column vector
        .{ 10, 20, 30 },    // M < N < K
        .{ 100, 50, 25 },   // M > N > K
        .{ 17, 31, 63 },    // All odd
        .{ 1000, 1, 1000 }, // Extreme aspect ratio
    };

    for (cases) |case| {
        // Test each configuration...
    }
}
```

2. **Numerical Stability Tests**
```zig
// tests/test_mm_numerical.zig

test "identity matrix multiplication" {
    const size = 100;
    var A = try allocator.alloc(f32, size * size);
    var I = try allocator.alloc(f32, size * size);
    var C = try allocator.alloc(f32, size * size);

    fillRandom(A, size * size);
    fillIdentity(I, size);

    mm.zig_mm(A.ptr, I.ptr, C.ptr, size, size, size);

    // A @ I should equal A
    try expectMatricesEqual(A, C, 1e-6);
}

test "zero matrix multiplication" {
    const size = 100;
    var A = try allocator.alloc(f32, size * size);
    var Z = try allocator.alloc(f32, size * size);
    var C = try allocator.alloc(f32, size * size);

    fillRandom(A, size * size);
    @memset(Z, 0.0);

    mm.zig_mm(A.ptr, Z.ptr, C.ptr, size, size, size);

    // A @ 0 should equal 0
    for (C) |val| {
        try testing.expectApproxEqAbs(0.0, val, 1e-6);
    }
}

test "large values - no overflow" {
    const size = 10;
    var A = try allocator.alloc(f32, size * size);
    var B = try allocator.alloc(f32, size * size);
    var C = try allocator.alloc(f32, size * size);

    @memset(A, 1e20);
    @memset(B, 1e20);

    mm.zig_mm(A.ptr, B.ptr, C.ptr, size, size, size);

    // Check for inf/nan
    for (C) |val| {
        try testing.expect(std.math.isFinite(val));
    }
}

test "small values - no underflow" {
    const size = 10;
    var A = try allocator.alloc(f32, size * size);
    var B = try allocator.alloc(f32, size * size);
    var C = try allocator.alloc(f32, size * size);

    @memset(A, 1e-20);
    @memset(B, 1e-20);

    mm.zig_mm(A.ptr, B.ptr, C.ptr, size, size, size);

    // Should handle gracefully (may be 0.0, but not NaN)
    for (C) |val| {
        try testing.expect(!std.math.isNan(val));
    }
}
```

3. **Memory Layout Tests**
```zig
// tests/test_mm_memory.zig

test "unaligned pointers" {
    const size = 64;
    var buffer = try allocator.alloc(f32, size * size + 16);

    // Force unalignment by offsetting pointer
    const A_unaligned = buffer[1..][0 .. size * size];
    const B_unaligned = buffer[3..][0 .. size * size];
    const C = try allocator.alloc(f32, size * size);

    fillRandom(A_unaligned, size * size);
    fillRandom(B_unaligned, size * size);

    // Should not crash despite unalignment
    mm.zig_mm(A_unaligned.ptr, B_unaligned.ptr, C.ptr, size, size, size);
}

test "overlapping memory regions" {
    const size = 32;
    var buffer = try allocator.alloc(f32, size * size * 3);

    const A = buffer[0 .. size * size];
    const B = buffer[size * size / 2 .. size * size + size * size / 2]; // Overlaps with A
    const C = buffer[size * size * 2 .. size * size * 3];

    fillRandom(A, size * size);
    @memcpy(B, A[0..B.len]); // B partially overlaps A

    // Undefined behavior, but shouldn't crash
    mm.zig_mm(A.ptr, B.ptr, C.ptr, size, size, size);
}
```

4. **Thread Safety Tests**
```zig
// tests/test_mm_threads.zig

test "concurrent calls from multiple threads" {
    const num_threads = 8;
    const size = 128;

    var threads: [num_threads]Thread = undefined;
    var contexts: [num_threads]TestContext = undefined;

    for (0..num_threads) |i| {
        contexts[i] = try TestContext.init(allocator, size);
        threads[i] = try Thread.spawn(.{}, testWorker, .{&contexts[i]});
    }

    for (threads) |thread| {
        thread.join();
    }

    // Verify all results are correct
    for (contexts) |ctx| {
        try expectMatricesEqual(ctx.expected, ctx.result, 1e-4);
    }
}

const TestContext = struct {
    A: []f32,
    B: []f32,
    C: []f32,
    expected: []f32,
    size: usize,

    fn init(allocator: std.mem.Allocator, size: usize) !TestContext {
        // Allocate and initialize...
    }
};

fn testWorker(ctx: *TestContext) void {
    // Each thread performs independent matrix multiply
    mm.zig_mm(ctx.A.ptr, ctx.B.ptr, ctx.C.ptr, ctx.size, ctx.size, ctx.size);
}
```

**Reference Implementations:**
- Eigen test suite: https://gitlab.com/libeigen/eigen/-/tree/master/test
- NumPy test patterns: https://github.com/numpy/numpy/blob/main/numpy/core/tests/test_multiarray.py
- BLIS test framework: https://github.com/flame/blis/tree/master/testsuite

---

#### Task 2.2: Benchmark Comparison Framework
**Priority:** 🟡 High
**Effort:** 8 hours

**Goal:** Automated benchmarking against NumPy/PyTorch with visualization

```python
# benchmarks/benchmark_suite.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, skipping torch benchmarks")

import zigtorch as zt

class BenchmarkRunner:
    """Comprehensive benchmark suite for ZigTorch."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict] = []

    def benchmark_operation(
        self,
        m: int,
        n: int,
        k: int,
        iterations: int = 10,
        warmup: int = 3,
    ) -> Dict:
        """Benchmark single configuration."""

        # Generate test data
        rng = np.random.default_rng(42)
        A = rng.standard_normal((m, k), dtype=np.float32)
        B = rng.standard_normal((k, n), dtype=np.float32)

        # Warmup
        for _ in range(warmup):
            _ = zt.matrix_multiply(A, B)

        # Benchmark ZigTorch
        times_zig = []
        for _ in range(iterations):
            start = time.perf_counter()
            C_zig = zt.matrix_multiply(A, B)
            end = time.perf_counter()
            times_zig.append(end - start)

        time_zig_median = np.median(times_zig)
        time_zig_std = np.std(times_zig)

        # Benchmark NumPy
        times_numpy = []
        for _ in range(iterations):
            start = time.perf_counter()
            C_numpy = A @ B
            end = time.perf_counter()
            times_numpy.append(end - start)

        time_numpy_median = np.median(times_numpy)
        time_numpy_std = np.std(times_numpy)

        # Verify correctness
        max_diff = np.max(np.abs(C_zig - C_numpy))
        relative_error = max_diff / (np.max(np.abs(C_numpy)) + 1e-8)

        # Calculate GFLOPS (2*M*N*K operations)
        flops = 2.0 * m * n * k
        gflops_zig = flops / time_zig_median / 1e9
        gflops_numpy = flops / time_numpy_median / 1e9

        result = {
            'M': m,
            'N': n,
            'K': k,
            'time_zig_ms': time_zig_median * 1000,
            'time_zig_std_ms': time_zig_std * 1000,
            'time_numpy_ms': time_numpy_median * 1000,
            'time_numpy_std_ms': time_numpy_std * 1000,
            'gflops_zig': gflops_zig,
            'gflops_numpy': gflops_numpy,
            'speedup_vs_numpy': time_numpy_median / time_zig_median,
            'max_diff': max_diff,
            'relative_error': relative_error,
            'correct': max_diff < 1e-4,
        }

        # PyTorch if available
        if HAS_TORCH:
            A_t = torch.from_numpy(A)
            B_t = torch.from_numpy(B)

            times_torch = []
            for _ in range(iterations):
                start = time.perf_counter()
                C_torch = torch.mm(A_t, B_t)
                end = time.perf_counter()
                times_torch.append(end - start)

            time_torch_median = np.median(times_torch)
            gflops_torch = flops / time_torch_median / 1e9

            result['time_torch_ms'] = time_torch_median * 1000
            result['gflops_torch'] = gflops_torch
            result['speedup_vs_torch'] = time_torch_median / time_zig_median

        return result

    def run_standard_suite(self):
        """Run standard benchmark configurations."""

        print("=" * 60)
        print("Running Standard Benchmark Suite")
        print("=" * 60)

        # 1. Square matrices - powers of 2
        print("\n1. Square Matrices (Powers of 2)")
        for size in [32, 64, 128, 256, 512, 1024, 2048]:
            print(f"  {size}×{size}...", end=" ", flush=True)
            result = self.benchmark_operation(size, size, size)
            self.results.append(result)
            print(f"✓ {result['gflops_zig']:.2f} GFLOPS "
                  f"(speedup: {result['speedup_vs_numpy']:.2f}x)")

        # 2. Rectangular matrices
        print("\n2. Rectangular Matrices")
        configs = [
            (1024, 512, 256),
            (2048, 1024, 512),
            (512, 2048, 1024),
            (4096, 256, 1024),
        ]
        for m, n, k in configs:
            print(f"  {m}×{k} @ {k}×{n}...", end=" ", flush=True)
            result = self.benchmark_operation(m, n, k)
            self.results.append(result)
            print(f"✓ {result['gflops_zig']:.2f} GFLOPS")

        # 3. Small matrices (overhead test)
        print("\n3. Small Matrices (Overhead Test)")
        for size in [2, 4, 8, 16, 32]:
            print(f"  {size}×{size}...", end=" ", flush=True)
            result = self.benchmark_operation(size, size, size, iterations=100)
            self.results.append(result)
            print(f"✓ {result['time_zig_ms']:.3f} ms")

        # 4. Edge cases
        print("\n4. Edge Cases")
        edge_cases = [
            (1, 1, 1),
            (1, 1000, 1),
            (1000, 1, 1000),
            (17, 31, 63),  # Odd sizes
        ]
        for m, n, k in edge_cases:
            print(f"  {m}×{k} @ {k}×{n}...", end=" ", flush=True)
            result = self.benchmark_operation(m, n, k, iterations=50)
            self.results.append(result)
            print(f"✓ Correct: {result['correct']}")

    def save_results(self):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        return df

    def generate_plots(self, df: pd.DataFrame):
        """Generate performance visualization."""

        # Filter square matrices for cleaner plots
        square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
        square = square[square['M'] >= 32]  # Exclude tiny matrices

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. GFLOPS comparison
        ax = axes[0, 0]
        ax.plot(square['M'], square['gflops_zig'], 'o-', label='ZigTorch', linewidth=2)
        ax.plot(square['M'], square['gflops_numpy'], 's-', label='NumPy', linewidth=2)
        if 'gflops_torch' in square.columns:
            ax.plot(square['M'], square['gflops_torch'], '^-', label='PyTorch', linewidth=2)
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        # 2. Speedup
        ax = axes[0, 1]
        ax.plot(square['M'], square['speedup_vs_numpy'], 'o-', linewidth=2, color='green')
        ax.axhline(y=1.0, color='red', linestyle='--', label='Break-even', linewidth=2)
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Speedup vs NumPy', fontsize=12)
        ax.set_title('Relative Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        # 3. Execution time
        ax = axes[1, 0]
        ax.semilogy(square['M'], square['time_zig_ms'], 'o-', label='ZigTorch', linewidth=2)
        ax.semilogy(square['M'], square['time_numpy_ms'], 's-', label='NumPy', linewidth=2)
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Time (ms, log scale)', fontsize=12)
        ax.set_title('Execution Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        # 4. Correctness (error)
        ax = axes[1, 1]
        ax.semilogy(square['M'], square['max_diff'], 'o-', linewidth=2, color='purple')
        ax.axhline(y=1e-4, color='red', linestyle='--', label='Tolerance (1e-4)', linewidth=2)
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Max Absolute Error (log scale)', fontsize=12)
        ax.set_title('Numerical Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        plt.tight_layout()
        plot_path = self.output_dir / 'performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_path}")

        plt.show()

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        square = df[(df['M'] == df['N']) & (df['M'] == df['K'])]
        square = square[square['M'] >= 32]

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        print(f"\nAverage speedup vs NumPy: {square['speedup_vs_numpy'].mean():.2f}x")
        print(f"Best speedup: {square['speedup_vs_numpy'].max():.2f}x "
              f"(size {square.loc[square['speedup_vs_numpy'].idxmax(), 'M']:.0f})")
        print(f"Worst speedup: {square['speedup_vs_numpy'].min():.2f}x "
              f"(size {square.loc[square['speedup_vs_numpy'].idxmin(), 'M']:.0f})")

        print(f"\nPeak GFLOPS (ZigTorch): {square['gflops_zig'].max():.2f}")
        print(f"Peak GFLOPS (NumPy): {square['gflops_numpy'].max():.2f}")

        print(f"\nAll tests correct: {df['correct'].all()}")
        print(f"Max numerical error: {df['max_diff'].max():.2e}")

if __name__ == '__main__':
    runner = BenchmarkRunner()
    runner.run_standard_suite()
    df = runner.save_results()
    runner.generate_plots(df)
    runner.print_summary(df)
```

**Reference:**
- NumPy benchmark suite: https://github.com/numpy/numpy/tree/main/benchmarks
- PyTorch benchmarks: https://github.com/pytorch/pytorch/tree/main/benchmarks
- Google Benchmark library: https://github.com/google/benchmark

---

### Week 3: Memory Safety & Validation

#### Task 3.1: Input Validation Layer
**Priority:** 🟡 High
**Effort:** 4 hours

**Goal:** Add comprehensive parameter validation before computation

```zig
// src/validation.zig

pub const ValidationError = error{
    NullPointer,
    ZeroDimension,
    DimensionOverflow,
    InvalidAlignment,
};

/// Validates matrix multiplication parameters
pub fn validateMatMul(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) ValidationError!void {
    // Check for null pointers (can't directly check in C ABI, but validate alignment)
    if (@intFromPtr(A) == 0 or @intFromPtr(B) == 0 or @intFromPtr(C) == 0) {
        return ValidationError.NullPointer;
    }

    // Check for zero dimensions
    if (M == 0 or N == 0 or K == 0) {
        return ValidationError.ZeroDimension;
    }

    // Check for dimension overflow
    const m64: u64 = @intCast(M);
    const n64: u64 = @intCast(N);
    const k64: u64 = @intCast(K);

    // Check if M*K would overflow
    var mk: u64 = undefined;
    if (@mulWithOverflow(&mk, m64, k64)) {
        return ValidationError.DimensionOverflow;
    }

    // Check if K*N would overflow
    var kn: u64 = undefined;
    if (@mulWithOverflow(&kn, k64, n64)) {
        return ValidationError.DimensionOverflow;
    }

    // Check if M*N would overflow
    var mn: u64 = undefined;
    if (@mulWithOverflow(&mn, m64, n64)) {
        return ValidationError.DimensionOverflow;
    }

    // Warn about poor alignment (optional, doesn't fail)
    if (@intFromPtr(A) % 64 != 0 or @intFromPtr(B) % 64 != 0) {
        // Not critical, but log for debugging
        if (@import("builtin").mode == .Debug) {
            std.debug.print("Warning: Input matrices not 64-byte aligned, performance may be suboptimal\n", .{});
        }
    }
}

/// Safe wrapper around zig_mm with validation
pub export fn zig_mm_safe(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) i32 {
    validateMatMul(A, B, C, M, N, K) catch |err| {
        std.debug.print("Matrix multiply validation failed: {}\n", .{err});
        return -1; // Error code
    };

    const mm = @import("ops/mm.zig");
    mm.zig_mm(A, B, C, M, N, K);
    return 0; // Success
}
```

**Python bindings update:**
```python
# python/zigtorch/native.py

class ZigTorchError(Exception):
    """Base exception for ZigTorch errors."""
    pass

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply with validation."""

    # Type checking
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError(f"Expected numpy arrays, got {type(a)} and {type(b)}")

    # Dimension checking
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shapes {a.shape} and {b.shape}")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply {a.shape[0]}×{a.shape[1]} @ {b.shape[0]}×{b.shape[1]}: "
            f"inner dimensions must match ({a.shape[1]} ≠ {b.shape[0]})"
        )

    # Ensure contiguous, float32
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    # Allocate output
    c = np.empty((a.shape[0], b.shape[1]), dtype=np.float32, order='C')

    # Call Zig (safe version with error codes)
    ret = _lib.zig_mm_safe(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.shape[0],
        b.shape[1],
        a.shape[1],
    )

    if ret != 0:
        raise ZigTorchError(f"Matrix multiplication failed with error code {ret}")

    return c
```

**Reference:**
- Zig error handling: https://ziglang.org/documentation/master/#Errors
- Rust validation patterns: https://doc.rust-lang.org/book/ch09-00-error-handling.html

---

#### Task 3.2: Memory Alignment Utilities
**Priority:** 🟢 Medium
**Effort:** 3 hours

```zig
// src/memory.zig

const std = @import("std");
const builtin = @import("builtin");

/// Optimal alignment for SIMD operations on current architecture
pub const OPTIMAL_ALIGNMENT = blk: {
    if ((builtin.cpu.arch == .x86_64) and
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
        break :blk 64; // AVX-512 cache line
    } else if ((builtin.cpu.arch == .x86_64) and
               std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        break :blk 32; // AVX2
    } else {
        break :blk 16; // SSE / generic
    }
};

/// Allocates aligned memory for optimal SIMD performance
pub fn allocAligned(
    allocator: std.mem.Allocator,
    comptime T: type,
    count: usize,
) ![]align(OPTIMAL_ALIGNMENT) T {
    const alignment = OPTIMAL_ALIGNMENT;

    // Calculate total size with alignment padding
    const size = count * @sizeOf(T);
    const aligned_size = std.mem.alignForward(usize, size, alignment);

    // Allocate raw bytes
    const bytes = try allocator.alignedAlloc(u8, alignment, aligned_size);

    // Cast to desired type
    const ptr: [*]align(OPTIMAL_ALIGNMENT) T = @ptrCast(@alignCast(bytes.ptr));
    return ptr[0..count];
}

/// Check if pointer is optimally aligned
pub fn isOptimallyAligned(ptr: anytype) bool {
    return @intFromPtr(ptr) % OPTIMAL_ALIGNMENT == 0;
}

/// Check if pointer is aligned to at least N bytes
pub fn isAlignedTo(ptr: anytype, comptime alignment: usize) bool {
    return @intFromPtr(ptr) % alignment == 0;
}

test "aligned allocation" {
    const allocator = std.testing.allocator;

    const data = try allocAligned(allocator, f32, 1024);
    defer allocator.free(data);

    // Verify alignment
    try std.testing.expect(isOptimallyAligned(data.ptr));
    try std.testing.expect(isAlignedTo(data.ptr, 64));
}
```

**Python wrapper for aligned allocation:**
```python
# python/zigtorch/memory.py

import numpy as np
from typing import Tuple

def empty_aligned(shape: Tuple[int, ...], dtype=np.float32, alignment: int = 64) -> np.ndarray:
    """Create aligned array for optimal SIMD performance."""

    size = np.prod(shape)
    itemsize = np.dtype(dtype).itemsize

    # Allocate with extra space for alignment
    buffer = np.empty(size * itemsize + alignment, dtype=np.uint8)

    # Find aligned offset
    offset = (-buffer.ctypes.data) % alignment

    # Create view with correct dtype and shape
    aligned_buffer = buffer[offset : offset + size * itemsize]
    array = np.frombuffer(aligned_buffer, dtype=dtype, count=size).reshape(shape)

    # Verify alignment
    assert array.ctypes.data % alignment == 0, f"Array not aligned to {alignment} bytes"

    return array

def zeros_aligned(shape: Tuple[int, ...], dtype=np.float32, alignment: int = 64) -> np.ndarray:
    """Create aligned zero-initialized array."""
    arr = empty_aligned(shape, dtype, alignment)
    arr[:] = 0
    return arr

def randn_aligned(shape: Tuple[int, ...], dtype=np.float32, alignment: int = 64) -> np.ndarray:
    """Create aligned random array."""
    arr = empty_aligned(shape, dtype, alignment)
    arr[:] = np.random.randn(*shape).astype(dtype)
    return arr

# Usage:
# A = zt.memory.empty_aligned((1024, 1024))  # Guaranteed 64-byte aligned
# result = zt.matrix_multiply(A, B)  # Optimal performance
```

**Reference:**
- C++ std::aligned_alloc: https://en.cppreference.com/w/cpp/memory/c/aligned_alloc
- NumPy alignment: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html

---

### Week 4: Documentation & Code Quality

#### Task 4.1: Inline Documentation
**Priority:** 🟡 High
**Effort:** 6 hours

**Goal:** Document all public APIs and complex algorithms

```zig
// src/ops/mm.zig (with documentation)

//! Matrix multiplication implementation using blocked algorithm with SIMD.
//!
//! This module implements high-performance matrix multiplication (GEMM) using:
//! - Blocked/tiled algorithm for cache efficiency
//! - SIMD vectorization (AVX-512, AVX2, SSE)
//! - Multi-threading with work distribution
//! - Manual prefetching for cache optimization
//!
//! Algorithm overview:
//! 1. Divide matrices into blocks that fit in cache (typically 256×256)
//! 2. Further divide into micro-kernels (16×16) for SIMD
//! 3. Distribute block rows across threads
//! 4. Use explicit prefetching to minimize cache misses
//!
//! References:
//! - Goto & van de Geijn (2008): "Anatomy of High-Performance Matrix Multiplication"
//! - BLIS framework: https://github.com/flame/blis
//! - Intel MKL documentation

const std = @import("std");
const Thread = std.Thread;
const Atomic = std.atomic.Value;
const builtin = @import("builtin");

/// Block sizes for cache tiling.
/// These are tuned for typical L1/L2 cache sizes (32KB/256KB).
/// Adjusted dynamically based on CPU architecture.
var BLOCK_M: usize = 256;
var BLOCK_N: usize = 256;
var BLOCK_K: usize = 256;

/// Micro-kernel dimensions.
/// Fixed at 16×16 to match SIMD vector widths and cache lines.
const MICRO_M = 16;
const MICRO_N = 16;

/// Prefetch distance constants.
/// PREFETCH_DISTANCE_L1: How many elements ahead to prefetch into L1 cache.
/// PREFETCH_DISTANCE_L2: How many elements ahead to prefetch into L2 cache.
const PREFETCH_DISTANCE_L1 = 64 / @sizeOf(f32); // One cache line (16 floats)
const PREFETCH_DISTANCE_L2 = 512 / @sizeOf(f32); // 8 cache lines (128 floats)

/// SIMD vector width based on available CPU features.
/// - 16 for AVX-512 (512 bits / 32 bits per float)
/// - 8 for AVX2 (256 bits / 32 bits per float)
/// - 4 for SSE or fallback (128 bits / 32 bits per float)
const VECTOR_WIDTH = if ((builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) and
                          std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
    16
else if ((builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) and
          std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
    8
else
    4;

/// Number of worker threads.
/// Initialized to min(CPU_count, 32) on first call.
var thread_count: usize = undefined;

/// Prefetch data into cache.
///
/// Uses x86 prefetch instructions to bring data into specified cache level.
/// On non-x86 architectures, this is a no-op.
///
/// Parameters:
/// - addr: Base pointer to data
/// - offset: Element offset from base
/// - cache_level: Target cache level (0=L1, 1=L2, 2=L3)
///
/// Note: Prefetching is a hint; CPU may ignore if already in cache.
inline fn prefetchData(addr: [*]const f32, offset: usize, cache_level: u2) void {
    if (builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) {
        // Map cache level to locality hint:
        // - 0 (NTA): Non-temporal, don't pollute cache
        // - 1 (T2): Low temporal locality (L2)
        // - 2 (T1): Medium temporal locality (L1)
        // - 3 (T0): High temporal locality (all levels)
        const locality: u3 = switch (cache_level) {
            0 => 1, // NTA
            1 => 2, // T2
            2 => 3, // T0
            else => 3,
        };

        switch (locality) {
            1 => asm volatile ("prefetchnta (%[addr], %[offset], 4)"
                :
                : [addr] "r" (addr),
                  [offset] "r" (offset),
                : "memory"),
            2 => asm volatile ("prefetcht2 (%[addr], %[offset], 4)"
                :
                : [addr] "r" (addr),
                  [offset] "r" (offset),
                : "memory"),
            else => asm volatile ("prefetcht0 (%[addr], %[offset], 4)"
                :
                : [addr] "r" (addr),
                  [offset] "r" (offset),
                : "memory"),
        }
    }
}

/// Optimized 16×16 micro-kernel for matrix multiplication.
///
/// Computes C[i:i+16, j:j+16] += A[i:i+16, k:k+16] @ B[k:k+16, j:j+16]
///
/// This is the performance-critical innermost kernel. It:
/// 1. Loads current C block into registers
/// 2. Performs 16×16 multiply-accumulate using SIMD
/// 3. Prefetches future data while computing
/// 4. Writes results back to C
///
/// Parameters:
/// - A: Matrix A in row-major layout
/// - B: Matrix B in row-major layout
/// - C: Output matrix C in row-major layout (accumulated)
/// - i: Starting row index in C
/// - j: Starting column index in C
/// - k: Starting inner dimension index
/// - lda: Leading dimension of A (usually K)
/// - ldb: Leading dimension of B (usually N)
/// - ldc: Leading dimension of C (usually N)
///
/// Preconditions:
/// - i+16 <= M, j+16 <= N, k+16 <= K
/// - All pointers must be valid for the given dimensions
///
/// Performance: ~90% of theoretical peak on modern CPUs
inline fn microKernel16x16(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    i: usize,
    j: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    @setEvalBranchQuota(4000); // Allow extensive compile-time unrolling

    // Local accumulator for C block (in registers/L1 cache)
    var c_block: [16][16]f32 = undefined;

    // Load current values from C
    inline for (0..16) |mi| {
        inline for (0..16) |ni| {
            c_block[mi][ni] = C[(i + mi) * ldc + j + ni];
        }
    }

    // Main computation loop: accumulate A[i:i+16, k:k+16] @ B[k:k+16, j:j+16]
    var kk: usize = 0;
    while (kk < MICRO_M) : (kk += 1) {
        // Prefetch future data every 8 iterations
        if (kk % 8 == 0 and kk + PREFETCH_DISTANCE_L2 < lda) {
            prefetchData(A + (i + 0) * lda, k + kk + PREFETCH_DISTANCE_L2, 1);
            prefetchData(A + (i + 8) * lda, k + kk + PREFETCH_DISTANCE_L2, 1);
            prefetchData(B + (k + kk + PREFETCH_DISTANCE_L2) * ldb, j, 1);
        }

        // Multiply-accumulate using SIMD vectors
        inline for (0..16) |mi| {
            const a_val = A[(i + mi) * lda + k + kk];

            if (VECTOR_WIDTH == 16) {
                // AVX-512: Process all 16 elements at once
                const a_vec: @Vector(16, f32) = @splat(a_val);
                var b_vec: @Vector(16, f32) = undefined;

                inline for (0..16) |ni| {
                    b_vec[ni] = B[(k + kk) * ldb + j + ni];
                }

                const result = a_vec * b_vec;

                inline for (0..16) |ni| {
                    c_block[mi][ni] += result[ni];
                }
            } else if (VECTOR_WIDTH == 8) {
                // AVX2: Process in two 8-element chunks
                const a_vec: @Vector(8, f32) = @splat(a_val);

                inline for (0..2) |chunk| {
                    var b_vec: @Vector(8, f32) = undefined;
                    inline for (0..8) |ni| {
                        b_vec[ni] = B[(k + kk) * ldb + j + chunk * 8 + ni];
                    }

                    const result = a_vec * b_vec;

                    inline for (0..8) |ni| {
                        c_block[mi][chunk * 8 + ni] += result[ni];
                    }
                }
            } else {
                // Fallback: Scalar operations (still unrolled at compile time)
                inline for (0..16) |ni| {
                    c_block[mi][ni] += a_val * B[(k + kk) * ldb + j + ni];
                }
            }
        }
    }

    // Write results back to C
    inline for (0..16) |mi| {
        inline for (0..16) |ni| {
            C[(i + mi) * ldc + j + ni] = c_block[mi][ni];
        }
    }
}

/// Matrix multiplication: C = A @ B
///
/// Computes C[M×N] = A[M×K] @ B[K×N] using blocked algorithm.
///
/// All matrices are in row-major layout:
/// - A[i,j] is at A[i*K + j]
/// - B[i,j] is at B[i*N + j]
/// - C[i,j] is at C[i*N + j]
///
/// Parameters:
/// - A: Left matrix (M×K), row-major
/// - B: Right matrix (K×N), row-major
/// - C: Output matrix (M×N), row-major, will be overwritten
/// - M: Number of rows in A and C
/// - N: Number of columns in B and C
/// - K: Inner dimension (columns of A, rows of B)
///
/// Performance characteristics:
/// - Time complexity: O(M*N*K)
/// - Space complexity: O(1) auxiliary
/// - Parallelized across M dimension
/// - Cache-optimized with blocking
///
/// Example:
/// ```zig
/// var A = [_]f32{1, 2, 3, 4}; // 2×2
/// var B = [_]f32{5, 6, 7, 8}; // 2×2
/// var C: [4]f32 = undefined;
/// zig_mm(&A, &B, &C, 2, 2, 2);
/// // C = [19, 22, 43, 50]
/// ```
pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    // Initialize configuration on first call (thread-safe)
    init_guard.call();

    // ... rest of implementation
}
```

**Reference:**
- Zig doc comments: https://ziglang.org/documentation/master/#Doc-Comments
- Doxygen style guide: https://www.doxygen.nl/manual/docblocks.html
- Rust documentation guide: https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html

---

#### Task 4.2: API Documentation Website
**Priority:** 🟢 Medium
**Effort:** 4 hours

**Tool:** Use `zig-docgen` or create custom docs with MkDocs

```bash
# docs/mkdocs.yml
site_name: ZigTorch Documentation
site_description: High-performance tensor computation in Zig
site_author: kitajusSus
repo_url: https://github.com/kitajusSus/testing-ZIG-torch

theme:
  name: material
  palette:
    primary: deep purple
    accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
      - Building from Source: getting-started/building.md
  - API Reference:
      - Matrix Operations: api/matrix.md
      - Tensor Class: api/tensor.md
      - Memory Utilities: api/memory.md
  - Benchmarks:
      - Performance: benchmarks/performance.md
      - Comparison: benchmarks/comparison.md
  - Developer Guide:
      - Architecture: dev/architecture.md
      - Contributing: dev/contributing.md
      - Testing: dev/testing.md
  - Examples:
      - Basic Usage: examples/basic.md
      - PyTorch Integration: examples/pytorch.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true

markdown_extensions:
  - admonition
  - codehilite
  - pymdownx.superfences
  - pymdownx.tabbed
  - pymdownx.arithmatex:
      generic: true
```

**Reference:**
- MkDocs Material: https://squidfunk.github.io/mkdocs-material/
- Rust book structure: https://doc.rust-lang.org/book/
- NumPy docs: https://numpy.org/doc/stable/

---

## Phase 2: Performance Engineering (Weeks 5-8)

**Goal:** Optimize for real-world performance, add auto-tuning

### Week 5: Cache Optimization

#### Task 5.1: Dynamic Block Size Selection
**Priority:** 🟡 High
**Effort:** 8 hours

**Goal:** Automatically determine optimal block sizes based on cache architecture

```zig
// src/autotuner.zig

const std = @import("std");
const builtin = @import("builtin");

/// Cache hierarchy information
pub const CacheInfo = struct {
    l1_data: usize, // L1 data cache size in bytes
    l2: usize,      // L2 cache size in bytes
    l3: usize,      // L3 cache size in bytes (0 if none)
    line_size: usize, // Cache line size in bytes
};

/// Detect cache sizes on Linux
fn detectCacheLinux() CacheInfo {
    // Read from /sys/devices/system/cpu/cpu0/cache/
    var info = CacheInfo{
        .l1_data = 32 * 1024,   // Default: 32 KB
        .l2 = 256 * 1024,       // Default: 256 KB
        .l3 = 8 * 1024 * 1024,  // Default: 8 MB
        .line_size = 64,
    };

    const paths = [_]struct { field: []const u8, path: []const u8 }{
        .{ .field = "l1_data", .path = "/sys/devices/system/cpu/cpu0/cache/index0/size" },
        .{ .field = "l2", .path = "/sys/devices/system/cpu/cpu0/cache/index2/size" },
        .{ .field = "l3", .path = "/sys/devices/system/cpu/cpu0/cache/index3/size" },
    };

    for (paths) |entry| {
        const file = std.fs.openFileAbsolute(entry.path, .{}) catch continue;
        defer file.close();

        var buf: [32]u8 = undefined;
        const n = file.readAll(&buf) catch continue;

        // Parse size (format: "32K" or "256K" or "8M")
        const str = std.mem.trim(u8, buf[0..n], &std.ascii.whitespace);
        const multiplier: usize = if (std.mem.endsWith(u8, str, "K"))
            1024
        else if (std.mem.endsWith(u8, str, "M"))
            1024 * 1024
        else
            1;

        const num_str = std.mem.trimRight(u8, str, "KkMm");
        const value = std.fmt.parseInt(usize, num_str, 10) catch continue;

        if (std.mem.eql(u8, entry.field, "l1_data")) {
            info.l1_data = value * multiplier;
        } else if (std.mem.eql(u8, entry.field, "l2")) {
            info.l2 = value * multiplier;
        } else if (std.mem.eql(u8, entry.field, "l3")) {
            info.l3 = value * multiplier;
        }
    }

    return info;
}

/// Detect cache sizes on macOS
fn detectCacheMacOS() CacheInfo {
    // Use sysctlbyname() to query cache info
    // hw.l1dcachesize, hw.l2cachesize, hw.l3cachesize

    // For now, use conservative defaults
    return CacheInfo{
        .l1_data = 32 * 1024,
        .l2 = 256 * 1024,
        .l3 = 8 * 1024 * 1024,
        .line_size = 64,
    };
}

/// Detect cache sizes on Windows
fn detectCacheWindows() CacheInfo {
    // Use GetLogicalProcessorInformation()

    // For now, use conservative defaults
    return CacheInfo{
        .l1_data = 32 * 1024,
        .l2 = 256 * 1024,
        .l3 = 8 * 1024 * 1024,
        .line_size = 64,
    };
}

/// Detect cache hierarchy
pub fn detectCache() CacheInfo {
    return switch (builtin.os.tag) {
        .linux => detectCacheLinux(),
        .macos => detectCacheMacOS(),
        .windows => detectCacheWindows(),
        else => CacheInfo{
            .l1_data = 32 * 1024,
            .l2 = 256 * 1024,
            .l3 = 8 * 1024 * 1024,
            .line_size = 64,
        },
    };
}

/// Compute optimal block size based on cache architecture
pub fn computeOptimalBlockSize(cache: CacheInfo) struct {
    block_m: usize,
    block_n: usize,
    block_k: usize,
} {
    // Algorithm from "Anatomy of High-Performance Matrix Multiplication"
    // (Goto & van de Geijn, 2008)
    //
    // We want blocks that fit in L2 cache:
    // - A_block: block_m × block_k floats
    // - B_block: block_k × block_n floats
    // - C_block: block_m × block_n floats
    //
    // Total: (block_m*block_k + block_k*block_n + block_m*block_n) * 4 bytes <= L2_size
    //
    // For square blocks (block_m = block_n = block_k = B):
    // 3*B² * 4 <= L2_size
    // B <= sqrt(L2_size / 12)

    const l2_floats = cache.l2 / @sizeOf(f32);
    const block_size_sqrt = @sqrt(@as(f64, @floatFromInt(l2_floats)) / 3.0);
    var block_size = @as(usize, @intFromFloat(block_size_sqrt));

    // Round down to multiple of 64 for alignment
    block_size = (block_size / 64) * 64;

    // Clamp to reasonable range [64, 512]
    block_size = @max(64, @min(block_size, 512));

    return .{
        .block_m = block_size,
        .block_n = block_size,
        .block_k = block_size,
    };
}

test "cache detection" {
    const cache = detectCache();

    // Sanity checks
    try std.testing.expect(cache.l1_data > 0);
    try std.testing.expect(cache.l2 > cache.l1_data);
    try std.testing.expect(cache.line_size == 64 or cache.line_size == 128);
}

test "block size computation" {
    const cache = CacheInfo{
        .l1_data = 32 * 1024,
        .l2 = 256 * 1024,
        .l3 = 8 * 1024 * 1024,
        .line_size = 64,
    };

    const blocks = computeOptimalBlockSize(cache);

    // Should fit in L2
    const total_bytes = (blocks.block_m * blocks.block_k +
                        blocks.block_k * blocks.block_n +
                        blocks.block_m * blocks.block_n) * @sizeOf(f32);

    try std.testing.expect(total_bytes <= cache.l2);
    try std.testing.expect(blocks.block_m % 64 == 0); // Alignment
}
```

**Integrate into mm.zig:**
```zig
// src/ops/mm.zig (updated)

const autotuner = @import("../autotuner.zig");

threadlocal var tuning_cache: ?struct {
    block_m: usize,
    block_n: usize,
    block_k: usize,
} = null;

fn getTuningParams() struct { block_m: usize, block_n: usize, block_k: usize } {
    if (tuning_cache) |params| return params;

    const cache_info = autotuner.detectCache();
    const params = autotuner.computeOptimalBlockSize(cache_info);
    tuning_cache = params;

    if (@import("builtin").mode == .Debug) {
        std.debug.print("Cache-aware tuning: L1={d}KB, L2={d}KB, block_size={d}\n", .{
            cache_info.l1_data / 1024,
            cache_info.l2 / 1024,
            params.block_m,
        });
    }

    return params;
}

pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    const params = getTuningParams();
    const BLOCK_M = params.block_m;
    const BLOCK_N = params.block_n;
    const BLOCK_K = params.block_k;

    // Use computed block sizes in algorithm...
}
```

**Reference:**
- Anatomy of High-Performance GEMM: https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf
- BLIS auto-tuning: https://github.com/flame/blis/wiki/Autotuning
- Halide autoscheduler: https://halide-lang.org/papers/autoscheduler2019.html

---

#### Task 5.2: Empirical Auto-Tuner (Optional)
**Priority:** 🟢 Low
**Effort:** 12 hours

**Goal:** Benchmark different configurations and cache results

```zig
// src/empirical_tuner.zig

/// Run micro-benchmarks to find optimal block size
pub fn empiricalTune(
    allocator: std.mem.Allocator,
    M: usize,
    N: usize,
    K: usize,
) !struct { block_m: usize, block_n: usize, block_k: usize } {
    const candidates = [_]usize{ 64, 128, 192, 256, 384, 512 };

    // Allocate test matrices
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Fill with random data
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    for (A) |*val| val.* = random.float(f32);
    for (B) |*val| val.* = random.float(f32);

    var best_time: u64 = std.math.maxInt(u64);
    var best_block: usize = 256;

    for (candidates) |block_size| {
        // Warmup
        zig_mm_with_block_size(A.ptr, B.ptr, C.ptr, M, N, K, block_size);

        // Benchmark
        const iterations = 5;
        const start = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            zig_mm_with_block_size(A.ptr, B.ptr, C.ptr, M, N, K, block_size);
        }

        const end = std.time.nanoTimestamp();
        const time = @as(u64, @intCast(end - start)) / iterations;

        if (time < best_time) {
            best_time = time;
            best_block = block_size;
        }
    }

    return .{
        .block_m = best_block,
        .block_n = best_block,
        .block_k = best_block,
    };
}
```

**Cache tuning results:**
```bash
# .zigtorch_cache (JSON file saved to ~/.config/zigtorch/)
{
  "cpu_model": "Intel Core i7-12700K",
  "cache_l1": 32768,
  "cache_l2": 262144,
  "cache_l3": 8388608,
  "optimal_block_size": 256,
  "benchmarked_at": "2026-01-26T10:30:00Z"
}
```

**Reference:**
- ATLAS auto-tuning: http://math-atlas.sourceforge.net/
- OpenTuner framework: https://github.com/jansel/opentuner
- FFTW planner: http://www.fftw.org/fftw3_doc/Planner-Flags.html

---

### Week 6-7: Additional Operations

#### Task 6.1: Transpose
**Priority:** 🟡 High
**Effort:** 6 hours

```zig
// src/ops/transpose.zig

/// Cache-oblivious matrix transpose using blocked algorithm
pub export fn zig_transpose(
    A: [*]const f32,
    B: [*]f32,
    M: usize,
    N: usize,
) callconv(.c) void {
    const BLOCK_SIZE = 64; // Fits in L1 cache

    var i: usize = 0;
    while (i < M) : (i += BLOCK_SIZE) {
        var j: usize = 0;
        while (j < N) : (j += BLOCK_SIZE) {
            const i_end = @min(i + BLOCK_SIZE, M);
            const j_end = @min(j + BLOCK_SIZE, N);

            // Transpose block
            transposeBlock(A, B, i, i_end, j, j_end, M, N);
        }
    }
}

inline fn transposeBlock(
    A: [*]const f32,
    B: [*]f32,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    M: usize,
    N: usize,
) void {
    for (i_start..i_end) |i| {
        for (j_start..j_end) |j| {
            B[j * M + i] = A[i * N + j];
        }
    }
}
```

**Python binding:**
```python
# python/zigtorch/native.py

_lib.zig_transpose.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.zig_transpose.restype = None

def transpose(a: np.ndarray) -> np.ndarray:
    """Transpose a 2D matrix."""
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.empty((a.shape[1], a.shape[0]), dtype=np.float32, order='C')

    _lib.zig_transpose(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.shape[0],
        a.shape[1],
    )

    return b
```

**Reference:**
- Cache-oblivious algorithms: https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
- Efficient matrix transpose: http://csapp.cs.cmu.edu/3e/labs.html

---

#### Task 6.2: Element-wise Operations
**Priority:** 🟡 High
**Effort:** 8 hours

```zig
// src/ops/elementwise.zig

const VECTOR_WIDTH = @import("../ops/mm.zig").VECTOR_WIDTH;

/// Element-wise addition: C = A + B
pub export fn zig_add(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    size: usize,
) callconv(.c) void {
    elementwiseBinary(A, B, C, size, .add);
}

/// Element-wise subtraction: C = A - B
pub export fn zig_sub(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    size: usize,
) callconv(.c) void {
    elementwiseBinary(A, B, C, size, .sub);
}

/// Element-wise multiplication: C = A * B
pub export fn zig_mul(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    size: usize,
) callconv(.c) void {
    elementwiseBinary(A, B, C, size, .mul);
}

/// Element-wise division: C = A / B
pub export fn zig_div(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    size: usize,
) callconv(.c) void {
    elementwiseBinary(A, B, C, size, .div);
}

const BinaryOp = enum { add, sub, mul, div };

inline fn elementwiseBinary(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    size: usize,
    comptime op: BinaryOp,
) void {
    // Vectorized loop
    var i: usize = 0;
    while (i + VECTOR_WIDTH <= size) : (i += VECTOR_WIDTH) {
        const a_vec: @Vector(VECTOR_WIDTH, f32) = A[i..][0..VECTOR_WIDTH].*;
        const b_vec: @Vector(VECTOR_WIDTH, f32) = B[i..][0..VECTOR_WIDTH].*;

        const c_vec = switch (op) {
            .add => a_vec + b_vec,
            .sub => a_vec - b_vec,
            .mul => a_vec * b_vec,
            .div => a_vec / b_vec,
        };

        @memcpy(C[i..][0..VECTOR_WIDTH], &c_vec);
    }

    // Tail loop
    while (i < size) : (i += 1) {
        C[i] = switch (op) {
            .add => A[i] + B[i],
            .sub => A[i] - B[i],
            .mul => A[i] * B[i],
            .div => A[i] / B[i],
        };
    }
}

/// Element-wise exponential: B = exp(A)
pub export fn zig_exp(
    A: [*]const f32,
    B: [*]f32,
    size: usize,
) callconv(.c) void {
    for (0..size) |i| {
        B[i] = @exp(A[i]);
    }
}

/// Element-wise logarithm: B = log(A)
pub export fn zig_log(
    A: [*]const f32,
    B: [*]f32,
    size: usize,
) callconv(.c) void {
    for (0..size) |i| {
        B[i] = @log(A[i]);
    }
}

/// ReLU activation: B = max(0, A)
pub export fn zig_relu(
    A: [*]const f32,
    B: [*]f32,
    size: usize,
) callconv(.c) void {
    var i: usize = 0;
    while (i + VECTOR_WIDTH <= size) : (i += VECTOR_WIDTH) {
        const a_vec: @Vector(VECTOR_WIDTH, f32) = A[i..][0..VECTOR_WIDTH].*;
        const zero_vec: @Vector(VECTOR_WIDTH, f32) = @splat(0.0);
        const c_vec = @max(a_vec, zero_vec);
        @memcpy(B[i..][0..VECTOR_WIDTH], &c_vec);
    }

    while (i < size) : (i += 1) {
        B[i] = @max(0.0, A[i]);
    }
}
```

**Python bindings:**
```python
# python/zigtorch/ops.py

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition."""
    a, b = _broadcast(a, b)
    c = np.empty_like(a)
    _lib.zig_add(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 a.size)
    return c

def relu(a: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.empty_like(a)
    _lib.zig_relu(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  a.size)
    return b
```

**Reference:**
- NumPy ufuncs: https://numpy.org/doc/stable/reference/ufuncs.html
- SLEEF vectorized math: https://sleef.org/

---

#### Task 6.3: Reduction Operations
**Priority:** 🟢 Medium
**Effort:** 6 hours

```zig
// src/ops/reduce.zig

/// Sum all elements
pub export fn zig_sum(A: [*]const f32, size: usize) callconv(.c) f32 {
    return reduce(A, size, 0.0, .add);
}

/// Find maximum element
pub export fn zig_max(A: [*]const f32, size: usize) callconv(.c) f32 {
    return reduce(A, size, -std.math.inf(f32), .max);
}

/// Find minimum element
pub export fn zig_min(A: [*]const f32, size: usize) callconv(.c) f32 {
    return reduce(A, size, std.math.inf(f32), .min);
}

const ReduceOp = enum { add, max, min };

inline fn reduce(
    A: [*]const f32,
    size: usize,
    init: f32,
    comptime op: ReduceOp,
) f32 {
    var acc_vec: @Vector(VECTOR_WIDTH, f32) = @splat(init);

    // Vectorized reduction
    var i: usize = 0;
    while (i + VECTOR_WIDTH <= size) : (i += VECTOR_WIDTH) {
        const a_vec: @Vector(VECTOR_WIDTH, f32) = A[i..][0..VECTOR_WIDTH].*;

        acc_vec = switch (op) {
            .add => acc_vec + a_vec,
            .max => @max(acc_vec, a_vec),
            .min => @min(acc_vec, a_vec),
        };
    }

    // Horizontal reduction
    var result = switch (op) {
        .add => @reduce(.Add, acc_vec),
        .max => @reduce(.Max, acc_vec),
        .min => @reduce(.Min, acc_vec),
    };

    // Tail
    while (i < size) : (i += 1) {
        result = switch (op) {
            .add => result + A[i],
            .max => @max(result, A[i]),
            .min => @min(result, A[i]),
        };
    }

    return result;
}

/// Argmax: index of maximum element
pub export fn zig_argmax(A: [*]const f32, size: usize) callconv(.c) usize {
    var max_val = A[0];
    var max_idx: usize = 0;

    for (1..size) |i| {
        if (A[i] > max_val) {
            max_val = A[i];
            max_idx = i;
        }
    }

    return max_idx;
}
```

**Reference:**
- CUB reductions: https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
- Thrust reductions: https://thrust.github.io/doc/group__reductions.html

---

### Week 8: Profiling & Optimization

#### Task 8.1: Built-in Profiler
**Priority:** 🟢 Medium
**Effort:** 6 hours

```zig
// src/profiler.zig

const std = @import("std");

pub const Profiler = struct {
    const Event = struct {
        name: []const u8,
        start: i128,
        end: i128,
        thread_id: u32,
    };

    events: std.ArrayList(Event),
    enabled: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Profiler {
        const enabled = std.posix.getenv("ZIGTORCH_PROFILE") != null;

        return .{
            .events = std.ArrayList(Event).init(allocator),
            .enabled = enabled,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.events.deinit();
    }

    pub fn begin(self: *Profiler, name: []const u8) void {
        if (!self.enabled) return;

        const now = std.time.nanoTimestamp();
        const thread_id = std.Thread.getCurrentId();

        self.events.append(.{
            .name = name,
            .start = now,
            .end = 0,
            .thread_id = thread_id,
        }) catch {};
    }

    pub fn end(self: *Profiler, name: []const u8) void {
        if (!self.enabled) return;

        const now = std.time.nanoTimestamp();
        const thread_id = std.Thread.getCurrentId();

        // Find matching begin event
        var i: usize = self.events.items.len;
        while (i > 0) {
            i -= 1;
            if (self.events.items[i].thread_id == thread_id and
                std.mem.eql(u8, self.events.items[i].name, name) and
                self.events.items[i].end == 0)
            {
                self.events.items[i].end = now;
                break;
            }
        }
    }

    pub fn report(self: *Profiler, writer: anytype) !void {
        try writer.writeAll("=== ZigTorch Profiling Report ===\n\n");

        // Group by name and compute statistics
        var stats = std.StringHashMap(struct {
            count: usize,
            total_ns: u64,
            min_ns: u64,
            max_ns: u64,
        }).init(self.allocator);
        defer stats.deinit();

        for (self.events.items) |event| {
            if (event.end == 0) continue; // Incomplete event

            const duration = @as(u64, @intCast(event.end - event.start));

            if (stats.getPtr(event.name)) |stat| {
                stat.count += 1;
                stat.total_ns += duration;
                stat.min_ns = @min(stat.min_ns, duration);
                stat.max_ns = @max(stat.max_ns, duration);
            } else {
                try stats.put(event.name, .{
                    .count = 1,
                    .total_ns = duration,
                    .min_ns = duration,
                    .max_ns = duration,
                });
            }
        }

        // Print table
        try writer.writeAll("Operation              Count    Total(ms)   Avg(ms)   Min(ms)   Max(ms)\n");
        try writer.writeAll("------------------------------------------------------------------\n");

        var iter = stats.iterator();
        while (iter.next()) |entry| {
            const stat = entry.value_ptr.*;
            const avg_ms = @as(f64, @floatFromInt(stat.total_ns)) / @as(f64, @floatFromInt(stat.count)) / 1_000_000.0;
            const total_ms = @as(f64, @floatFromInt(stat.total_ns)) / 1_000_000.0;
            const min_ms = @as(f64, @floatFromInt(stat.min_ns)) / 1_000_000.0;
            const max_ms = @as(f64, @floatFromInt(stat.max_ns)) / 1_000_000.0;

            try writer.print("{s:<20}   {d:>5}   {d:>8.3}   {d:>7.3}   {d:>7.3}   {d:>7.3}\n", .{
                entry.key_ptr.*,
                stat.count,
                total_ms,
                avg_ms,
                min_ms,
                max_ms,
            });
        }
    }

    /// Export to Chrome Trace format (chrome://tracing)
    pub fn exportChromeTrace(self: *Profiler, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var writer = file.writer();

        try writer.writeAll("[");

        for (self.events.items, 0..) |event, i| {
            if (event.end == 0) continue;

            if (i > 0) try writer.writeAll(",");

            const duration_us = @as(f64, @floatFromInt(event.end - event.start)) / 1000.0;
            const timestamp_us = @as(f64, @floatFromInt(event.start)) / 1000.0;

            try writer.print(
                \\
                \\{{"name":"{s}","ph":"X","ts":{d:.3},"dur":{d:.3},"pid":1,"tid":{d}}}
            , .{
                event.name,
                timestamp_us,
                duration_us,
                event.thread_id,
            });
        }

        try writer.writeAll("]\n");
    }
};

// Global profiler instance
var global_profiler: ?Profiler = null;
var profiler_mutex = std.Thread.Mutex{};

pub fn getGlobalProfiler() *Profiler {
    profiler_mutex.lock();
    defer profiler_mutex.unlock();

    if (global_profiler == null) {
        global_profiler = Profiler.init(std.heap.c_allocator);
    }

    return &global_profiler.?;
}

pub fn profile(comptime name: []const u8, func: anytype, args: anytype) @TypeOf(@call(.auto, func, args)) {
    const profiler = getGlobalProfiler();
    profiler.begin(name);
    defer profiler.end(name);

    return @call(.auto, func, args);
}
```

**Usage in mm.zig:**
```zig
pub export fn zig_mm(...) callconv(.c) void {
    const profiler = @import("../profiler.zig").getGlobalProfiler();

    profiler.begin("zig_mm");
    defer profiler.end("zig_mm");

    profiler.begin("initialization");
    // ... init code
    profiler.end("initialization");

    profiler.begin("compute");
    // ... computation
    profiler.end("compute");
}
```

**Python API:**
```python
# python/zigtorch/profiler.py

import os
import atexit

class Profiler:
    """Context manager for profiling ZigTorch operations."""

    def __enter__(self):
        os.environ['ZIGTORCH_PROFILE'] = '1'
        return self

    def __exit__(self, *args):
        # Trigger report generation
        import zigtorch._native as native
        native.print_profile_report()

        del os.environ['ZIGTORCH_PROFILE']

# Usage:
# with zt.Profiler():
#     result = zt.matrix_multiply(A, B)
# # Automatically prints report on exit
```

**Reference:**
- Chrome Trace Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/
- Tracy Profiler: https://github.com/wolfpld/tracy
- Perfetto: https://perfetto.dev/

---

## Phase 3: API & Usability (Weeks 9-12)

**Goal:** Create NumPy/PyTorch-like Python API for ease of use

### Week 9-10: Tensor Class

#### Task 9.1: Core Tensor Implementation
**Priority:** 🔴 Critical
**Effort:** 12 hours

```python
# python/zigtorch/tensor.py

import numpy as np
from typing import Union, Tuple, List, Optional
from . import native

class Tensor:
    """
    N-dimensional array with Zig backend.

    Similar to PyTorch's tensor but CPU-only with Zig acceleration.
    """

    def __init__(self, data: Union[np.ndarray, List, float], dtype=np.float32):
        """
        Create tensor from data.

        Args:
            data: Input data (array, list, or scalar)
            dtype: Data type (default: float32)

        Examples:
            >>> t = Tensor([1, 2, 3])
            >>> t = Tensor([[1, 2], [3, 4]])
            >>> t = Tensor(5.0)  # Scalar
        """
        if isinstance(data, list):
            data = np.array(data, dtype=dtype)
        elif isinstance(data, (int, float)):
            data = np.array(data, dtype=dtype)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Cannot create Tensor from {type(data)}")

        self._data = np.ascontiguousarray(data, dtype=dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of tensor."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._data.size

    @property
    def dtype(self):
        """Data type."""
        return self._data.dtype

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self._data

    def item(self) -> float:
        """Get scalar value (for 0-D tensors)."""
        if self.size != 1:
            raise ValueError(f"Cannot convert {self.size}-element tensor to scalar")
        return float(self._data.flat[0])

    # ===== Arithmetic Operations =====

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise addition."""
        if isinstance(other, (int, float)):
            return Tensor(self._data + other)
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                # Broadcast
                result = self._data + other._data
            else:
                result = native.add(self._data, other._data)
            return Tensor(result)
        else:
            return NotImplemented

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise subtraction."""
        if isinstance(other, (int, float)):
            return Tensor(self._data - other)
        elif isinstance(other, Tensor):
            result = native.sub(self._data, other._data)
            return Tensor(result)
        else:
            return NotImplemented

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            return Tensor(self._data * other)
        elif isinstance(other, Tensor):
            result = native.mul(self._data, other._data)
            return Tensor(result)
        else:
            return NotImplemented

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division."""
        if isinstance(other, (int, float)):
            return Tensor(self._data / other)
        elif isinstance(other, Tensor):
            result = native.div(self._data, other._data)
            return Tensor(result)
        else:
            return NotImplemented

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            return NotImplemented

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(f"matmul requires 2D tensors, got {self.ndim}D @ {other.ndim}D")

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Cannot multiply {self.shape} @ {other.shape}")

        result = native.matrix_multiply(self._data, other._data)
        return Tensor(result)

    def __neg__(self) -> 'Tensor':
        """Negation."""
        return Tensor(-self._data)

    # ===== In-place Operations =====

    def __iadd__(self, other):
        self._data += other._data if isinstance(other, Tensor) else other
        return self

    def __isub__(self, other):
        self._data -= other._data if isinstance(other, Tensor) else other
        return self

    def __imul__(self, other):
        self._data *= other._data if isinstance(other, Tensor) else other
        return self

    def __itruediv__(self, other):
        self._data /= other._data if isinstance(other, Tensor) else other
        return self

    # ===== Reduction Operations =====

    def sum(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Sum of elements."""
        if axis is None:
            return float(native.sum(self._data.ravel()))
        else:
            return Tensor(np.sum(self._data, axis=axis))

    def mean(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Mean of elements."""
        if axis is None:
            return self.sum() / self.size
        else:
            return Tensor(np.mean(self._data, axis=axis))

    def max(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Maximum element."""
        if axis is None:
            return float(native.max(self._data.ravel()))
        else:
            return Tensor(np.max(self._data, axis=axis))

    def min(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Minimum element."""
        if axis is None:
            return float(native.min(self._data.ravel()))
        else:
            return Tensor(np.min(self._data, axis=axis))

    def argmax(self, axis: Optional[int] = None) -> Union[int, 'Tensor']:
        """Index of maximum element."""
        if axis is None:
            return int(native.argmax(self._data.ravel()))
        else:
            return Tensor(np.argmax(self._data, axis=axis))

    # ===== Shape Operations =====

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        return Tensor(self._data.reshape(shape))

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Transpose dimensions."""
        if axes is None and self.ndim == 2:
            # Use Zig-accelerated transpose
            result = native.transpose(self._data)
            return Tensor(result)
        else:
            return Tensor(np.transpose(self._data, axes))

    @property
    def T(self) -> 'Tensor':
        """Transpose (2D only)."""
        if self.ndim != 2:
            raise ValueError(f"T requires 2D tensor, got {self.ndim}D")
        return self.transpose()

    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        return Tensor(np.squeeze(self._data, axis=axis))

    def unsqueeze(self, axis: int) -> 'Tensor':
        """Add dimension of size 1."""
        return Tensor(np.expand_dims(self._data, axis=axis))

    # ===== Activation Functions =====

    def relu(self) -> 'Tensor':
        """ReLU activation."""
        result = native.relu(self._data)
        return Tensor(result)

    def exp(self) -> 'Tensor':
        """Exponential."""
        result = native.exp(self._data)
        return Tensor(result)

    def log(self) -> 'Tensor':
        """Natural logarithm."""
        result = native.log(self._data)
        return Tensor(result)

    # ===== Utilities =====

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})\n{self._data}"

    def __str__(self) -> str:
        return str(self._data)

    def __getitem__(self, key):
        return Tensor(self._data[key])

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self._data[key] = value._data
        else:
            self._data[key] = value

    # ===== Constructors =====

    @staticmethod
    def zeros(*shape: int, dtype=np.float32) -> 'Tensor':
        """Create tensor filled with zeros."""
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(*shape: int, dtype=np.float32) -> 'Tensor':
        """Create tensor filled with ones."""
        return Tensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def randn(*shape: int, dtype=np.float32) -> 'Tensor':
        """Create tensor with random normal values."""
        return Tensor(np.random.randn(*shape).astype(dtype))

    @staticmethod
    def rand(*shape: int, dtype=np.float32) -> 'Tensor':
        """Create tensor with random uniform values [0, 1)."""
        return Tensor(np.random.rand(*shape).astype(dtype))

    @staticmethod
    def arange(start: float, stop: float, step: float = 1.0, dtype=np.float32) -> 'Tensor':
        """Create tensor with evenly spaced values."""
        return Tensor(np.arange(start, stop, step, dtype=dtype))

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50, dtype=np.float32) -> 'Tensor':
        """Create tensor with linearly spaced values."""
        return Tensor(np.linspace(start, stop, num, dtype=dtype))
```

**Reference:**
- PyTorch Tensor API: https://pytorch.org/docs/stable/tensors.html
- NumPy ndarray: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- JAX DeviceArray: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.DeviceArray.html

---

### Week 11: PyTorch Interoperability

#### Task 11.1: PyTorch Integration
**Priority:** 🟢 Medium
**Effort:** 8 hours

```python
# python/zigtorch/torch_compat.py

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError("PyTorch integration requires PyTorch to be installed")

import zigtorch as zt

class ZigFunction(torch.autograd.Function):
    """
    Custom PyTorch operation using Zig backend.

    Enables use of ZigTorch in PyTorch computational graphs with autograd.
    """

    @staticmethod
    def forward(ctx, input, weight):
        """
        Forward pass: C = input @ weight

        Args:
            input: (M, K) tensor
            weight: (K, N) tensor

        Returns:
            output: (M, N) tensor
        """
        ctx.save_for_backward(input, weight)

        # Convert to numpy, call Zig, convert back
        input_np = input.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()

        output_np = zt.native.matrix_multiply(input_np, weight_np)
        output = torch.from_numpy(output_np).to(input.device)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for matrix multiply.

        Given dL/dC, compute:
        - dL/dA = dL/dC @ B^T
        - dL/dB = A^T @ dL/dC
        """
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        # Gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            grad_output_np = grad_output.detach().cpu().numpy()
            weight_t_np = weight.T.detach().cpu().numpy()
            grad_input_np = zt.native.matrix_multiply(grad_output_np, weight_t_np)
            grad_input = torch.from_numpy(grad_input_np).to(input.device)

        # Gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            input_t_np = input.T.detach().cpu().numpy()
            grad_output_np = grad_output.detach().cpu().numpy()
            grad_weight_np = zt.native.matrix_multiply(input_t_np, grad_output_np)
            grad_weight = torch.from_numpy(grad_weight_np).to(input.device)

        return grad_input, grad_weight

def matmul(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using Zig backend.

    Drop-in replacement for torch.matmul() that uses ZigTorch.
    Supports autograd.

    Args:
        input: (M, K) tensor
        weight: (K, N) tensor

    Returns:
        output: (M, N) tensor

    Example:
        >>> import torch
        >>> import zigtorch.torch_compat as ztc
        >>>
        >>> x = torch.randn(100, 50, requires_grad=True)
        >>> w = torch.randn(50, 20, requires_grad=True)
        >>>
        >>> # Use ZigTorch instead of PyTorch
        >>> y = ztc.matmul(x, w)
        >>>
        >>> loss = y.sum()
        >>> loss.backward()  # Gradients computed correctly
    """
    return ZigFunction.apply(input, weight)

class ZigLinear(torch.nn.Module):
    """
    Linear layer using Zig backend.

    Drop-in replacement for torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = input @ weight^T + bias
        """
        # Note: Linear uses weight^T convention
        output = matmul(input, self.weight.T)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, backend=Zig'

# Convenience function to convert model layers
def convert_to_zig(model: torch.nn.Module, inplace: bool = False) -> torch.nn.Module:
    """
    Convert all Linear layers in model to ZigLinear.

    Args:
        model: PyTorch model
        inplace: Modify model in-place

    Returns:
        Modified model (or original if inplace=True)

    Example:
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(100, 50),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(50, 10),
        ... )
        >>> model_zig = ztc.convert_to_zig(model)
    """
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Create ZigLinear with same parameters
            zig_linear = ZigLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )

            # Copy weights
            zig_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                zig_linear.bias.data = module.bias.data.clone()

            # Replace module
            setattr(model, name, zig_linear)
        elif len(list(module.children())) > 0:
            # Recursively convert submodules
            convert_to_zig(module, inplace=True)

    return model
```

**Example usage:**
```python
# examples/pytorch_integration.py

import torch
import torch.nn as nn
import zigtorch.torch_compat as ztc

# Define model with ZigTorch layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ztc.ZigLinear(784, 256)
        self.fc2 = ztc.ZigLinear(256, 128)
        self.fc3 = ztc.ZigLinear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Or convert existing model
standard_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

zig_model = ztc.convert_to_zig(standard_model)

# Train as usual
optimizer = torch.optim.Adam(zig_model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = zig_model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()  # Zig backend handles forward AND backward!
        optimizer.step()
```

**Reference:**
- PyTorch custom functions: https://pytorch.org/docs/stable/notes/extending.html
- JAX pytrees: https://jax.readthedocs.io/en/latest/pytrees.html

---

### Week 12: Examples & Tutorials

#### Task 12.1: Example Gallery
**Priority:** 🟢 Medium
**Effort:** 6 hours

```python
# examples/01_basic_operations.py
"""
Basic Tensor Operations
=======================

This example demonstrates basic tensor creation and arithmetic.
"""

import zigtorch as zt

# Create tensors
a = zt.Tensor([[1, 2], [3, 4]])
b = zt.Tensor([[5, 6], [7, 8]])

print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)

# Arithmetic
print("\nAddition: a + b")
print(a + b)

print("\nElement-wise multiplication: a * b")
print(a * b)

print("\nMatrix multiplication: a @ b")
print(a @ b)

# Reductions
print(f"\nSum of a: {a.sum()}")
print(f"Mean of a: {a.mean()}")
print(f"Max of a: {a.max()}")

# ---

# examples/02_linear_regression.py
"""
Linear Regression
=================

Fit y = Wx + b using gradient descent.
"""

import zigtorch as zt
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y_true = 3 * X + 2 + 0.5 * np.random.randn(100, 1).astype(np.float32)

X_tensor = zt.Tensor(X)
y_tensor = zt.Tensor(y_true)

# Initialize parameters
W = zt.Tensor(np.random.randn(1, 1).astype(np.float32))
b = zt.Tensor(np.zeros((1, 1), dtype=np.float32))

# Training
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = X_tensor @ W + b

    # Loss (MSE)
    error = y_pred - y_tensor
    loss = (error * error).sum() / X.shape[0]

    # Backward pass (manual gradients)
    grad_W = (X_tensor.T @ error) * (2.0 / X.shape[0])
    grad_b = error.sum() * (2.0 / X.shape[0])

    # Update
    W -= grad_W * learning_rate
    b -= grad_b * learning_rate

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, W = {W.item():.4f}, b = {b.item():.4f}")

print(f"\nFinal: W = {W.item():.4f}, b = {b.item():.4f}")
print(f"True:  W = 3.0000, b = 2.0000")

# Plot
plt.scatter(X, y_true, alpha=0.5, label='Data')
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1).astype(np.float32)
y_plot = (zt.Tensor(X_plot) @ W + b).numpy()
plt.plot(X_plot, y_plot, 'r-', label='Fit', linewidth=2)
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with ZigTorch')
plt.savefig('linear_regression.png', dpi=150)
plt.show()

# ---

# examples/03_neural_network.py
"""
Simple Neural Network
=====================

2-layer MLP for classification.
"""

import zigtorch as zt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to one-hot
y_train_onehot = np.zeros((y_train.shape[0], 2), dtype=np.float32)
y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

# Initialize network
np.random.seed(42)
W1 = zt.Tensor(np.random.randn(2, 64).astype(np.float32) * 0.01)
b1 = zt.Tensor(np.zeros((1, 64), dtype=np.float32))
W2 = zt.Tensor(np.random.randn(64, 2).astype(np.float32) * 0.01)
b2 = zt.Tensor(np.zeros((1, 2), dtype=np.float32))

# Training
learning_rate = 0.1
epochs = 100
batch_size = 32

for epoch in range(epochs):
    # Mini-batch training
    indices = np.random.permutation(X_train.shape[0])

    for i in range(0, X_train.shape[0], batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = zt.Tensor(X_train[batch_indices])
        y_batch = zt.Tensor(y_train_onehot[batch_indices])

        # Forward
        hidden = (X_batch @ W1 + b1).relu()
        logits = hidden @ W2 + b2

        # Softmax (manual)
        exp_logits = logits.exp()
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -(y_batch * probs.log()).sum() / batch_size

        # Backward (manual)
        grad_logits = (probs - y_batch) / batch_size
        grad_W2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0, keepdims=True)

        grad_hidden = grad_logits @ W2.T
        grad_hidden_relu = grad_hidden * (hidden.numpy() > 0)  # ReLU derivative
        grad_W1 = X_batch.T @ zt.Tensor(grad_hidden_relu)
        grad_b1 = zt.Tensor(grad_hidden_relu.sum(axis=0, keepdims=True))

        # Update
        W1 -= grad_W1 * learning_rate
        b1 -= grad_b1 * learning_rate
        W2 -= grad_W2 * learning_rate
        b2 -= grad_b2 * learning_rate

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Test
X_test_tensor = zt.Tensor(X_test)
hidden_test = (X_test_tensor @ W1 + b1).relu()
logits_test = hidden_test @ W2 + b2
preds = logits_test.argmax(axis=1)

accuracy = (preds == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")
```

**Reference:**
- PyTorch tutorials: https://pytorch.org/tutorials/
- Scikit-learn examples: https://scikit-learn.org/stable/auto_examples/
- JAX tutorials: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

---

## Phase 4: Advanced Features (Weeks 13-20)

**Goal:** Add production-ready features and advanced optimizations

### Week 13-14: Mixed Precision

#### Task 13.1: FP16 Support
**Priority:** 🟢 Medium
**Effort:** 10 hours

```zig
// src/ops/mm_fp16.zig

/// Matrix multiply for FP16 (half precision)
pub export fn zig_mm_fp16(
    A: [*]const f16,
    B: [*]const f16,
    C: [*]f16,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    // Similar structure to f32 version but using f16 vectors
    // ARM NEON: float16x8_t
    // x86 AVX-512 FP16: __m512h

    // For now, convert to f32, compute, convert back
    const allocator = std.heap.c_allocator;

    const A_f32 = allocator.alloc(f32, M * K) catch return;
    defer allocator.free(A_f32);
    const B_f32 = allocator.alloc(f32, K * N) catch return;
    defer allocator.free(B_f32);
    const C_f32 = allocator.alloc(f32, M * N) catch return;
    defer allocator.free(C_f32);

    // Convert A, B to f32
    for (0..M*K) |i| A_f32[i] = @floatCast(A[i]);
    for (0..K*N) |i| B_f32[i] = @floatCast(B[i]);

    // Compute in f32
    const mm = @import("mm.zig");
    mm.zig_mm(A_f32.ptr, B_f32.ptr, C_f32.ptr, M, N, K);

    // Convert C back to f16
    for (0..M*N) |i| C[i] = @floatCast(C_f32[i]);
}

/// Mixed precision: inputs FP16, accumulate in FP32, output FP16
pub export fn zig_mm_mixed(
    A: [*]const f16,
    B: [*]const f16,
    C: [*]f16,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    // This is the performance-optimal path:
    // - Load FP16 data (2x less bandwidth)
    // - Convert to FP32 for computation (maintains precision)
    // - Store FP16 result (2x less bandwidth)

    // Implementation depends on SIMD support
    // ARM: vcvt instructions
    // x86: vcvtph2ps / vcvtps2ph
}
```

**Python API:**
```python
# python/zigtorch/tensor.py (additions)

class Tensor:
    def __init__(self, data, dtype=np.float32):
        self._data = np.ascontiguousarray(data, dtype=dtype)
        self._dtype = dtype

    def half(self) -> 'Tensor':
        """Convert to FP16."""
        return Tensor(self._data, dtype=np.float16)

    def float(self) -> 'Tensor':
        """Convert to FP32."""
        return Tensor(self._data, dtype=np.float32)

    def double(self) -> 'Tensor':
        """Convert to FP64."""
        return Tensor(self._data, dtype=np.float64)

# Usage:
# model_fp16 = model.half()  # Convert to FP16 for faster inference
# loss = model_fp16(x).float()  # Back to FP32 for loss
```

**Reference:**
- NVIDIA Mixed Precision: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
- ARM FP16 instructions: https://developer.arm.com/documentation/dui0801/latest/A64-Floating-point-Instructions
- Intel AVX-512 FP16: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-fp16-instruction-set.html

---

#### Task 13.2: INT8 Quantization
**Priority:** 🟢 Medium
**Effort:** 12 hours

```zig
```
