
const std = @import("std");
const builtin = @import("builtin");
const Thread = std.Thread;

// Optimization constants
const L1_CACHE_SIZE = 32 * 1024;
const L2_CACHE_SIZE = 256 * 1024;
const CACHE_LINE_SIZE = 64;
const ThreadCount = 6;
const VECTOR_SIZE = 8;
const BLOCK_M = 128;
const BLOCK_N = 128;
const BLOCK_K = 128;
const MICRO_M = 8;
const MICRO_N = 8;

const ThreadContext = struct {
    start_row: usize,
    end_row: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
};

// Zoptymalizowany micro-kernel 8x8 z wykorzystaniem SIMD
inline fn microKernel8x8(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    i: usize,
    j: usize,
    k: usize,
    N: usize,
    K: usize,
) void {
    var c: [8][8]f32 = .{
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
        .{0} ** 8,
    };

    var kk: usize = 0;
    while (kk < 8) : (kk += 1) {
        const a0: @Vector(8, f32) = .{ 
            A[(i + 0) * K + k + kk], 
            A[(i + 1) * K + k + kk], 
            A[(i + 2) * K + k + kk], 
            A[(i + 3) * K + k + kk], 
            A[(i + 4) * K + k + kk], 
            A[(i + 5) * K + k + kk], 
            A[(i + 6) * K + k + kk], 
            A[(i + 7) * K + k + kk] 
        };
        const b0: @Vector(8, f32) = .{ 
            B[(k + kk) * N + j + 0], 
            B[(k + kk) * N + j + 1], 
            B[(k + kk) * N + j + 2], 
            B[(k + kk) * N + j + 3], 
            B[(k + kk) * N + j + 4], 
            B[(k + kk) * N + j + 5], 
            B[(k + kk) * N + j + 6], 
            B[(k + kk) * N + j + 7] 
        };

        inline for (0..8) |row| {
            const a_element = a0[row];
            const a_broadcast: @Vector(8, f32) = @splat(a_element);
            const result = a_broadcast * b0;
            inline for (0..8) |col| {
                c[row][col] += result[col];
            }
        }
    }

    inline for (0..8) |row| {
        const vec_c: @Vector(8, f32) = .{ 
            c[row][0], c[row][1], c[row][2], c[row][3], 
            c[row][4], c[row][5], c[row][6], c[row][7] 
        };
        @memcpy(C[(i + row) * N + j ..][0..8], @as([*]const f32, @ptrCast(&vec_c)));
    }
}

fn panelMultiply(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    i: usize,
    j: usize,
    k: usize,
    N: usize,
    K: usize,
    block_m: usize,
    block_n: usize,
    block_k: usize,
) void {
    var ii: usize = 0;
    while (ii < block_m) : (ii += MICRO_M) {
        var jj: usize = 0;
        while (jj < block_n) : (jj += MICRO_N) {
            inline for (0..MICRO_M) |mi| {
                inline for (0..MICRO_N) |ni| {
                    C[(i + ii + mi) * N + j + jj + ni] = 0;
                }
            }

            var kk: usize = 0;
            while (kk < block_k) : (kk += 8) {
                microKernel8x8(
                    A,
                    B,
                    C,
                    i + ii,
                    j + jj,
                    k + kk,
                    N,
                    K,
                );
            }
        }
    }
}

fn worker(context: ThreadContext) void {
    const N = context.N;
    const K = context.K;
    
    const block_m = BLOCK_M;
    const block_n = BLOCK_N;
    const block_k = BLOCK_K;

    var i: usize = context.start_row;
    while (i < context.end_row) : (i += block_m) {
        const actual_block_m = @min(block_m, context.end_row - i);
        
        var j: usize = 0;
        while (j < N) : (j += block_n) {
            const actual_block_n = @min(block_n, N - j);
            
            var k: usize = 0;
            while (k < K) : (k += block_k) {
                const actual_block_k = @min(block_k, K - k);
                
                panelMultiply(
                    context.A,
                    context.B,
                    context.C,
                    i,
                    j,
                    k,
                    N,
                    K,
                    actual_block_m,
                    actual_block_n,
                    actual_block_k,
                );
            }
        }
    }
}

export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    var threads: [ThreadCount]Thread = undefined;
    var contexts: [ThreadCount]ThreadContext = undefined;

    const effective_thread_count = @min(ThreadCount, M / MICRO_M);

    for (0..effective_thread_count) |t| {
        const rows_per_thread = M / effective_thread_count;
        const start_row = t * rows_per_thread;
        const end_row = if (t == effective_thread_count - 1) M else (t + 1) * rows_per_thread;

        contexts[t] = ThreadContext{
            .start_row = start_row,
            .end_row = end_row,
            .A = A,
            .B = B,
            .C = C,
            .M = M,
            .N = N,
            .K = K,
        };

        threads[t] = Thread.spawn(.{}, worker, .{contexts[t]}) catch unreachable;
    }

    for (0..effective_thread_count) |t| {
        threads[t].join();
    }
}

