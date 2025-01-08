const std = @import("std");
pub const BlockSize = 64;

export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    var i: usize = 0;
    while (i < M) : (i += BlockSize) {
        var j: usize = 0;
        while (j < N) : (j += BlockSize) {
            var k: usize = 0;
            while (k < K) : (k += BlockSize) {
                const i_end = @min(i + BlockSize, M);
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
                        
                        // Dodajemy loop unrolling
                        const unroll_factor = 4;
                        const remaining = k_end - k;
                        const unroll_limit = k + (remaining - (remaining % unroll_factor));
                        
                        while (kk < unroll_limit) : (kk += unroll_factor) {
                            sum += A[rowA + kk] * B[kk * N + jj] +
                                  A[rowA + kk + 1] * B[(kk + 1) * N + jj] +
                                  A[rowA + kk + 2] * B[(kk + 2) * N + jj] +
                                  A[rowA + kk + 3] * B[(kk + 3) * N + jj];
                        }
                        
                        // Pozostałe elementy
                        while (kk < k_end) : (kk += 1) {
                            sum += A[rowA + kk] * B[kk * N + jj];
                        }
                        
                        C[rowC + jj] = sum;
                    }
                }
            }
        }
    }
}
