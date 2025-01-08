const std = @import("std");
pub const BlockSize = 64;

fn worker(
    start_i: usize,
    end_i: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    N: usize,
    K: usize,
) void {
    std.debug.print("Worker started: {d}-{d}\n", .{start_i, end_i});
    
    var i: usize = start_i;
    while (i < end_i) : (i += 1) {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    std.debug.print("Worker finished: {d}-{d}\n", .{start_i, end_i});
}

export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    std.debug.print("Starting multiplication: {d}x{d} * {d}x{d}\n", .{M, K, K, N});

    const thread_count: usize = 2;
    var handles: [2]?std.Thread = undefined;

    const chunk_size = M / thread_count;
    
    for (0..thread_count) |t| {
        const start_i = t * chunk_size;
        const end_i = if (t == thread_count - 1) M else (t + 1) * chunk_size;
        
        std.debug.print("Spawning thread {d}: {d}-{d}\n", .{t, start_i, end_i});
        
        handles[t] = std.Thread.spawn(
            .{},
            worker,
            .{ start_i, end_i, A, B, C, N, K }
        ) catch |err| {
            std.debug.print("Thread spawn failed: {}\n", .{err});
            return;
        };
    }

    std.debug.print("Waiting for threads to finish...\n", .{});
    
    for (handles) |maybe_thread| {
        if (maybe_thread) |thread| {
            thread.join();
        }
    }
    
    std.debug.print("All threads finished\n", .{});
}
