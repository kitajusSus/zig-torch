const std = @import("std");
const Thread = std.Thread;
const Atomic = std.atomic.Atomic;

// Hardware-specific optimization constants (tunable)
const L1_CACHE_SIZE = 32 * 1024;
const L2_CACHE_SIZE = 256 * 1024;
const CACHE_LINE_SIZE = 64;
const VECTOR_SIZE = 8; // 8 floats = 32 bytes, good for AVX/AVX2

// Blocking parameters (adjust based on hardware)
const BLOCK_M = 128;
const BLOCK_N = 128;
const BLOCK_K = 128;
const MICRO_M = 8;
const MICRO_N = 8;

// Thread handling
const MAX_THREADS = 16;
var thread_count: usize = undefined;

// Initialize thread count based on available CPU cores
fn initThreadCount() void {
    thread_count = std.math.min(
        Thread.getCpuCount() catch 4,
        MAX_THREADS
    );
}

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

// Simple barrier implementation for thread synchronization
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
        const phase = self.phase.load(.Monotonic);
        const prev = self.count.fetchAdd(1, .Release);
        
        if (prev + 1 == self.total) {
            // Last thread to arrive resets the barrier
            self.count.store(0, .Monotonic);
            self.phase.store(phase + 1, .Release);
        } else {
            // Other threads wait
            while (self.phase.load(.Acquire) == phase) {
                Thread.yield();
            }
        }
    }
};

// Optimized micro-kernel for 8x8 block multiplication using SIMD
inline fn microKernel8x8(
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
    var c_block: [8][8]f32 = .{
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
    };

    // Inner loop processes 8 values at once using SIMD
    var kk: usize = 0;
    while (kk < 8) : (kk += 1) {
        // Load A and B rows/columns for vectorization
        const a_row0 = A[(i + 0) * lda + k + kk];
        const a_row1 = A[(i + 1) * lda + k + kk];
        const a_row2 = A[(i + 2) * lda + k + kk];
        const a_row3 = A[(i + 3) * lda + k + kk];
        const a_row4 = A[(i + 4) * lda + k + kk];
        const a_row5 = A[(i + 5) * lda + k + kk];
        const a_row6 = A[(i + 6) * lda + k + kk];
        const a_row7 = A[(i + 7) * lda + k + kk];

        const b_col: @Vector(8, f32) = .{
            B[(k + kk) * ldb + j + 0],
            B[(k + kk) * ldb + j + 1],
            B[(k + kk) * ldb + j + 2],
            B[(k + kk) * ldb + j + 3],
            B[(k + kk) * ldb + j + 4],
            B[(k + kk) * ldb + j + 5],
            B[(k + kk) * ldb + j + 6],
            B[(k + kk) * ldb + j + 7],
        };

        // Update each element using vectorization
        const a0_vec: @Vector(8, f32) = @splat(a_row0);
        const a1_vec: @Vector(8, f32) = @splat(a_row1);
        const a2_vec: @Vector(8, f32) = @splat(a_row2);
        const a3_vec: @Vector(8, f32) = @splat(a_row3);
        const a4_vec: @Vector(8, f32) = @splat(a_row4);
        const a5_vec: @Vector(8, f32) = @splat(a_row5);
        const a6_vec: @Vector(8, f32) = @splat(a_row6);
        const a7_vec: @Vector(8, f32) = @splat(a_row7);

        // SIMD multiply-add for all rows
        const c0_vec = a0_vec * b_col;
        const c1_vec = a1_vec * b_col;
        const c2_vec = a2_vec * b_col;
        const c3_vec = a3_vec * b_col;
        const c4_vec = a4_vec * b_col;
        const c5_vec = a5_vec * b_col;
        const c6_vec = a6_vec * b_col;
        const c7_vec = a7_vec * b_col;

        // Store results back to c_block
        inline for (0..8) |col| {
            c_block[0][col] += c0_vec[col];
            c_block[1][col] += c1_vec[col];
            c_block[2][col] += c2_vec[col];
            c_block[3][col] += c3_vec[col];
            c_block[4][col] += c4_vec[col];
            c_block[5][col] += c5_vec[col];
            c_block[6][col] += c6_vec[col];
            c_block[7][col] += c7_vec[col];
        }
    }

    // Write results back to C matrix
    for (0..8) |mi| {
        for (0..8) |ni| {
            C[(i + mi) * ldc + j + ni] += c_block[mi][ni];
        }
    }
}

// Block multiply for a panel of the matrix
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
    // Iterate over micro-blocks within the block
    var ii: usize = 0;
    while (ii < block_m - (block_m % MICRO_M)) : (ii += MICRO_M) {
        var jj: usize = 0;
        while (jj < block_n - (block_n % MICRO_N)) : (jj += MICRO_N) {
            // Process each micro-block with the micro kernel
            var kk: usize = 0;
            while (kk < block_k - (block_k % 8)) : (kk += 8) {
                microKernel8x8(
                    A,
                    B,
                    C,
                    i + ii,
                    j + jj,
                    k + kk,
                    lda,
                    ldb,
                    ldc,
                );
            }
            
            // Handle remaining k (if block_k is not multiple of 8)
            if (block_k % 8 != 0) {
                var rem_k: usize = block_k - (block_k % 8);
                while (rem_k < block_k) : (rem_k += 1) {
                    for (0..MICRO_M) |mi| {
                        for (0..MICRO_N) |ni| {
                            C[(i + ii + mi) * ldc + j + jj + ni] += 
                                A[(i + ii + mi) * lda + k + rem_k] * 
                                B[(k + rem_k) * ldb + j + jj + ni];
                        }
                    }
                }
            }
        }
        
        // Handle remaining n (if block_n is not multiple of MICRO_N)
        if (block_n % MICRO_N != 0) {
            const jj = block_n - (block_n % MICRO_N);
            for (0..MICRO_M) |mi| {
                for (jj..block_n) |ni| {
                    for (0..block_k) |kk| {
                        C[(i + ii + mi) * ldc + j + ni] += 
                            A[(i + ii + mi) * lda + k + kk] * 
                            B[(k + kk) * ldb + j + ni];
                    }
                }
            }
        }
    }
    
    // Handle remaining m (if block_m is not multiple of MICRO_M)
    if (block_m % MICRO_M != 0) {
        const ii = block_m - (block_m % MICRO_M);
        for (ii..block_m) |mi| {
            for (0..block_n) |ni| {
                for (0..block_k) |kk| {
                    C[(i + mi) * ldc + j + ni] += 
                        A[(i + mi) * lda + k + kk] * 
                        B[(k + kk) * ldb + j + ni];
                }
            }
        }
    }
}

// Worker function executed by each thread
fn worker(context: *ThreadContext) void {
    const A = context.A;
    const B = context.B;
    const C = context.C;
    const M = context.M;
    const N = context.N;
    const K = context.K;
    const start_row = context.start_row;
    const end_row = context.end_row;

    // Initialize the result to zero for this thread's portion
    for (start_row..end_row) |i| {
        for (0..N) |j| {
            C[i * N + j] = 0;
        }
    }

    // Wait for all threads to finish initialization
    context.barrier.wait();

    // Block by block multiplication
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
                    A, B, C,
                    i, j, k,
                    block_m, block_n, block_k,
                    K, N, N,
                );
            }
        }
    }

    // Wait for all threads to complete
    context.barrier.wait();
}

// Main entry point for matrix multiplication
export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    // Initialize thread count if not already done
    initThreadCount();
    
    // Use fewer threads for small matrices
    const effective_threads = if (M < 256 or N < 256 or K < 256)
        1
    else if (M < 512 or N < 512 or K < 512)
        @min(thread_count, 2)
    else
        thread_count;
    
    // If single-threaded, simplify the computation
    if (effective_threads == 1) {
        for (0..M) |i| {
            for (0..N) |j| {
                C[i * N + j] = 0;
                for (0..K) |k| {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
        return;
    }
    
    // Create barrier for thread synchronization
    var barrier = Barrier.init(effective_threads);
    
    // Allocate memory for contexts and threads
    var contexts: [MAX_THREADS]ThreadContext = undefined;
    var threads: [MAX_THREADS]Thread = undefined;
    
    // Calculate rows per thread (divide work evenly)
    const base_rows_per_thread = M / effective_threads;
    const extra_rows = M % effective_threads;
    
    // Start position for first thread
    var start_row: usize = 0;
    
    // Spawn worker threads
    for (0..effective_threads) |t| {
        // Calculate work range for this thread
        const rows_for_this_thread = base_rows_per_thread + 
            if (t < extra_rows) @as(usize, 1) else @as(usize, 0);
        const end_row = start_row + rows_for_this_thread;
        
        // Configure thread context
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
        
        // Spawn thread
        threads[t] = Thread.spawn(.{}, worker, .{&contexts[t]}) catch unreachable;
        
        // Update start for next thread
        start_row = end_row;
    }
    
    // Wait for all threads to complete
    for (0..effective_threads) |t| {
        threads[t].join();
    }
}
