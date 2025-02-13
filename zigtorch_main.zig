const std = @import("std");

pub const BlockSize = 64;

fn getCpuCount() usize {
    return std.Thread.getCpuCount() catch 1;
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
    _ = M;
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

                        const length = k_end - k;
                        const unroll4 = length - (length % 4);

                        while ((kk - k) < unroll4) : (kk += 4) {
                            sum +=
                                A[rowA + kk + 0] * B[(kk + 0) * N + jj] +
                                A[rowA + kk + 1] * B[(kk + 1) * N + jj] +
                                A[rowA + kk + 2] * B[(kk + 2) * N + jj] +
                                A[rowA + kk + 3] * B[(kk + 3) * N + jj];
                        }

                        while (kk < k_end) : (kk += 1) {
                            sum += A[rowA + kk] * B[kk * N + jj];
                        }

                        C[rowC + jj] += sum;
                    }
                }
            }
        }
    }
}

export fn mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    const num_threads = getCpuCount();
    const chunk_size = M / num_threads;

    var threads = std.ArrayList(std.Thread).init(std.heap.page_allocator);
    defer threads.deinit();

    for (0..num_threads) |thread_id| {
        const start_i = thread_id * chunk_size;
        const end_i = if (thread_id == num_threads - 1) M else start_i + chunk_size;

        const thread = std.Thread.spawn(.{}, worker, .{
            start_i, end_i, A, B, C, M, N, K,
        }) catch @panic("Failed to spawn thread");
        threads.append(thread) catch @panic("Failed to append thread");
    }

    for (threads.items) |thread| {
        thread.join();
    }
}
