const std = @import("std");
const Thread = std.Thread;
const Atomic = std.atomic.Value;
const builtin = @import("builtin");

// Block sizes adaptable to cache characteristics
var BLOCK_M: usize = 256;
var BLOCK_N: usize = 256;
var BLOCK_K: usize = 256;
const MICRO_M = 16;
const MICRO_N = 16;

// Prefetch constants for different cache levels
const PREFETCH_DISTANCE_L1 = 64 / @sizeOf(f32); // One cache line ahead
const PREFETCH_DISTANCE_L2 = 512 / @sizeOf(f32); // Multiple cache lines ahead

// Fix: use hasFeature instead of directly accessing fields
const VECTOR_WIDTH = if ((builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
    16
else if ((builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2))
    8
else
    4;

// Parametr dostrajania ilości wątków
var thread_count: usize = undefined;

fn initThreadCount() void {
    thread_count = @min(Thread.getCpuCount() catch 4, 32); // Większa maksymalna liczba wątków dla maszyn o wielu rdzeniach
}

fn determineOptimalBlockSize() void {
    if ((builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
        BLOCK_M = 384;
        BLOCK_N = 384;
        BLOCK_K = 384;
    } else if (builtin.cpu.arch == .aarch64) {
        // Slightly different sizes may work better on ARM
        BLOCK_M = 192;
        BLOCK_N = 256;
        BLOCK_K = 256;
    }
}

fn initGlobals() void {
    initThreadCount();
    determineOptimalBlockSize();
}

var init_guard = std.once(initGlobals);

// Improved prefetching with explicit cache level targeting
inline fn prefetchData(addr: [*]const f32, offset: usize, cache_level: u2) void {
    if (builtin.cpu.arch == .x86 or builtin.cpu.arch == .x86_64) {
        // Cache level: 0=L1, 1=L2, 2=L3
        const locality: u3 = switch (cache_level) {
            0 => 1, // NTA (non-temporal)
            1 => 2, // Low locality
            2 => 3, // High locality
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

// Zoptymalizowany mikrokernel do mnożenia bloków 16x16 z wykorzystaniem SIMD i prefetchingu
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
    @setEvalBranchQuota(4000);
    var c_block: [16][16]f32 = undefined;

    // Załaduj aktualne wartości z macierzy C do bloku
    inline for (0..16) |mi| {
        inline for (0..16) |ni| {
            c_block[mi][ni] = C[(i + mi) * ldc + j + ni];
        }
    }

    // Główna pętla obliczeniowa z prefetchingiem
    var kk: usize = 0;
    while (kk < MICRO_M) : (kk += 1) {
        // Prefetching przyszłych danych
        if (kk % 8 == 0 and kk + PREFETCH_DISTANCE_L2 < lda) {
            prefetchData(A + (i + 0) * lda, k + kk + PREFETCH_DISTANCE_L2, 1);
            prefetchData(A + (i + 8) * lda, k + kk + PREFETCH_DISTANCE_L2, 1);
            prefetchData(B + (k + kk + PREFETCH_DISTANCE_L2) * ldb, j, 1);
        }

        // Implementacja mnożenia z wykorzystaniem wektorów SIMD
        inline for (0..16) |mi| {
            const a_val = A[(i + mi) * lda + k + kk];

            if (VECTOR_WIDTH == 16) {
                // Wykorzystanie AVX-512 gdy dostępne
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
                // Wykorzystanie AVX2
                const a_vec: @Vector(8, f32) = @splat(a_val);

                // Process in two 8-element chunks
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
                // Standardowe podejście, działa na wszystkich architekturach
                inline for (0..16) |ni| {
                    c_block[mi][ni] += a_val * B[(k + kk) * ldb + j + ni];
                }
            }
        }
    }

    // Zapisanie wyników do macierzy wynikowej
    inline for (0..16) |mi| {
        inline for (0..16) |ni| {
            C[(i + mi) * ldc + j + ni] = c_block[mi][ni];
        }
    }
}

// Ulepszona funkcja mnożenia bloków macierzy z poprawnym obsługiwaniem niepełnych bloków
fn blockMultiply(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    i: usize,
    j: usize,
    k: usize,
    block_m: usize,
    block_n: usize,
    block_k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    // Pętla po mikro-blokach z lepszym układem pętli dla cache
    var ii: usize = 0;
    while (ii < block_m) : (ii += MICRO_M) {
        const mi_end = @min(MICRO_M, block_m - ii);

        if (mi_end == MICRO_M) {
            // Pełne bloki w wymiarze M
            var jj: usize = 0;
            while (jj < block_n) : (jj += MICRO_N) {
                const ni_end = @min(MICRO_N, block_n - jj);

                if (ni_end == MICRO_N) {
                    // Pełne bloki zarówno w M jak i N - używamy zoptymalizowanego mikrokernela
                    var kk: usize = 0;
                    while (kk < block_k) : (kk += MICRO_M) {
                        microKernel16x16(A, B, C, i + ii, j + jj, k + kk, lda, ldb, ldc);
                    }
                } else {
                    // Niepełne bloki w wymiarze N - używamy ogólnego podejścia
                    for (0..mi_end) |mi| {
                        for (0..ni_end) |ni| {
                            var sum: f32 = 0;
                            for (0..block_k) |ki| {
                                sum += A[(i + ii + mi) * lda + k + ki] * B[(k + ki) * ldb + j + jj + ni];
                            }
                            C[(i + ii + mi) * ldc + j + jj + ni] += sum;
                        }
                    }
                }
            }
        } else {
            // Niepełne bloki w wymiarze M - używamy ogólnego podejścia
            for (0..mi_end) |mi| {
                for (0..block_n) |ni| {
                    var sum: f32 = 0;
                    for (0..block_k) |ki| {
                        sum += A[(i + ii + mi) * lda + k + ki] * B[(k + ki) * ldb + j + ni];
                    }
                    C[(i + ii + mi) * ldc + j + ni] += sum;
                }
            }
        }
    }
}

// Ulepszona implementacja bariery
const Barrier = struct {
    count: Atomic(usize),
    total: usize,
    phase: Atomic(usize),

    fn init(count: usize) Barrier {
        return Barrier{
            .count = Atomic(usize).init(0),
            .total = count,
            .phase = Atomic(usize).init(0),
        };
    }

    fn wait(self: *Barrier) void {
        const phase = self.phase.load(.monotonic);
        const prev = self.count.fetchAdd(1, .release);

        if (prev + 1 == self.total) {
            self.count.store(0, .monotonic);
            self.phase.store(phase + 1, .release);
        } else {
            while (self.phase.load(.acquire) == phase) {
                // Fix: handle the error with catch
                Thread.yield() catch {};
            }
        }
    }
};

// Struktura kontekstu dla wątków
const ThreadContext = struct {
    id: usize,
    start_row: usize,
    end_row: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
    barrier: *Barrier,
};

fn worker(context: *ThreadContext) void {
    const A = context.A;
    const B = context.B;
    const C = context.C;
    // const M = context.M; // Unused
    const N = context.N;
    const K = context.K;
    const start_row = context.start_row;
    const end_row = context.end_row;

    for (start_row..end_row) |i| {
        for (0..N) |j| {
            C[i * N + j] = 0;
        }
    }

    context.barrier.wait();

    var i: usize = start_row;
    while (i < end_row) : (i += BLOCK_M) {
        const block_m = @min(BLOCK_M, end_row - i);

        var j: usize = 0;
        while (j < N) : (j += BLOCK_N) {
            const block_n = @min(BLOCK_N, N - j);

            var k: usize = 0;
            while (k < K) : (k += BLOCK_K) {
                const block_k = @min(BLOCK_K, K - k);

                blockMultiply(
                    A,
                    B,
                    C,
                    i,
                    j,
                    k,
                    block_m,
                    block_n,
                    block_k,
                    K,
                    N,
                    N,
                );
            }
        }
    }

    context.barrier.wait();
}

pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    init_guard.call();
    var effective_threads: usize = 1;

    const total_elements = M * N * K;
    if (total_elements > 1024 * 1024) { // > 1M elementów
        effective_threads = thread_count;
    } else if (total_elements > 256 * 256) { // > 65k elementów
        effective_threads = @max(thread_count / 2, 2);
    } else {
        effective_threads = 1; // Małe macierze - single thread
    }

    if (effective_threads == 1 or M < 64 or N < 64 or K < 64) {
        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }

    var barrier = Barrier.init(effective_threads);

    const base_rows_per_thread = M / effective_threads;
    const extra_rows = M % effective_threads;

    var start_row: usize = 0;

    var contexts: [32]ThreadContext = undefined; //
    var threads: [32]Thread = undefined;
    var threads_created: usize = 0;

    for (0..effective_threads) |t| {
        const rows_for_this_thread = base_rows_per_thread +
            if (t < extra_rows) @as(usize, 1) else @as(usize, 0);
        const end_row = start_row + rows_for_this_thread;

        contexts[t] = ThreadContext{
            .id = t,
            .start_row = start_row,
            .end_row = end_row,
            .A = A,
            .B = B,
            .C = C,
            .M = M,
            .N = N,
            .K = K,
            .barrier = &barrier,
        };

        threads[t] = Thread.spawn(.{}, worker, .{&contexts[t]}) catch |err| {
            std.debug.print("Error spawning thread {d}: {any}\n", .{ t, err });
            effective_threads = t;
            threads_created = t;

            // W przypadku błędu przy tworzeniu pierwszego wątku, wykonaj obliczenia jednowątkowo
            if (t == 0) {
                for (0..M) |i| {
                    for (0..N) |j| {
                        var sum: f32 = 0;
                        for (0..K) |k| {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
                return;
            }
            break;
        };

        threads_created += 1;
        start_row = end_row;
    }

    if (threads_created < effective_threads) {
        for (start_row..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    for (0..threads_created) |t| {
        threads[t].join();
    }
}

test "zig_mm small correctness 2x3 * 3x2" {
    var A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var C = [_]f32{0} ** 4;

    zig_mm(&A, &B, &C, 2, 2, 3);

    try std.testing.expectApproxEqAbs(58, C[0], 1e-4);
    try std.testing.expectApproxEqAbs(64, C[1], 1e-4);
    try std.testing.expectApproxEqAbs(139, C[2], 1e-4);
    try std.testing.expectApproxEqAbs(154, C[3], 1e-4);
}

test "zig_mm identity 4x4" {
    var A = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var B = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    var C = [_]f32{0} ** 16;

    zig_mm(&A, &B, &C, 4, 4, 4);

    try std.testing.expectApproxEqAbs(1, C[0], 1e-4);
    try std.testing.expectApproxEqAbs(6, C[5], 1e-4);
    try std.testing.expectApproxEqAbs(11, C[10], 1e-4);
    try std.testing.expectApproxEqAbs(16, C[15], 1e-4);
}
