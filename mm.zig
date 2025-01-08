const std = @import("std");
pub const BlockSize = 64;

fn getCpuCount() usize {
    return 4;
}

fn worker(
    start_i: usize,
    end_i: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    if (start_i >= M or end_i > M) {
        std.debug.print("Invalid bounds: start_i={}, end_i={}, M={}\n", .{start_i, end_i, M});
        return;
    }

    var i: usize = start_i;
    while (i < end_i) : (i += BlockSize) {
        var j: usize = 0;
        while (j < N) : (j += BlockSize) {
            var k: usize = 0;
            while (k < K) : (k += BlockSize) {
                const i_end = @min(i + BlockSize, end_i);
                const j_end = @min(j + BlockSize, N);
                const k_end = @min(k + BlockSize, K);

                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    const rowA = ii * K;
                    const rowC = ii * N;
                    
                    var jj: usize = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = 0.0;
                        var kk: usize = k;
                        
                        while (kk < k_end) : (kk += 1) {
                            if (rowA + kk >= M * K or kk * N + jj >= K * N) {
                                std.debug.print("Index out of bounds\n", .{});
                                return;
                            }
                            sum += A[rowA + kk] * B[kk * N + jj];
                        }
                        if (rowC + jj >= M * N) {
                            std.debug.print("Output index out of bounds\n", .{});
                            return;
                        }
                        C[rowC + jj] = sum;
                    }
                }
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
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
