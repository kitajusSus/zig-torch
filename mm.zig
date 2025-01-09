
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
) void {  // Usunąłem !void ponieważ nie rzucamy błędów
    std.debug.print("Worker started: {d}-{d}\n", .{start_i, end_i});
    
    var i: usize = start_i;
    while (i < end_i) : (i += 1) {
        var j: usize = 0;
        while (j < N) : (j += 1) {
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < K) : (k += 1) {
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
    const thread_count: usize = 2;
    
    var t: usize = 0;
    while (t < thread_count) : (t += 1) {
        const start_i = t * (M / thread_count);
        const end_i = if (t == thread_count - 1) M else (t + 1) * (M / thread_count);
        
        worker(start_i, end_i, A, B, C, N, K); // Usunąłem catch ponieważ funkcja nie zwraca błędu
    }
}

