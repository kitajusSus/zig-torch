const std = @import("std");
pub const BlockSize = 64;

fn worker(start_i: usize, end_i: usize, A: [*]const f32, B: [*]const f32, C: [*]f32, _: usize, N: usize, K: usize) void {
    for (start_i..end_i) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

export fn zig_mm(A: [*]const f32, B: [*]const f32, C: [*]f32, M: usize, N: usize, K: usize) callconv(.C) void {
    const thread_count: usize = 4;
    var handles: [4]?std.Thread = undefined;
    const chunk_size = (M + thread_count - 1) / thread_count;

    for (0..thread_count) |t| {
        const start_i = t * chunk_size;
        const end_i = @min(start_i + chunk_size, M);
        handles[t] = std.Thread.spawn(.{}, worker, .{start_i, end_i, A, B, C, M, N, K}) catch null;
    }

    for (handles) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}
