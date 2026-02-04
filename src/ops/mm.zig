const std = @import("std");
const BLOCK_SIZE = 64;
pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    for (0..M * N) |idx| {
        C[idx] = 0;
    }
    var i_0: usize = 0;
    while (i_0 < M) : (i_0 += BLOCK_SIZE) {
        const i_end = @min(i_0 + BLOCK_SIZE, M);

        var k0: usize = 0;
        while (k0 < K) : (k0 += BLOCK_SIZE) {
            const k_end = @min(k0 + BLOCK_SIZE, K);

            var j0: usize = 0;
            while (j0 < N) : (j0 += BLOCK_SIZE) {
                const j_end = @min(j0 + BLOCK_SIZE, N);
                multiplyBlock(A, B, C, i_0, i_end, j0, j_end, k0, k_end, K, N);
            }
        }
    }
}

inline fn multiplyBlock(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
    K: usize,
    N: usize,
) void {
    const VEC_SIZE = 8;

    for (i_start..i_end) |i| {
        for (k_start..k_end) |k| {
            const a_val = A[i * K + k];
            const a_vec: @Vector(VEC_SIZE, f32) = @splat(a_val);

            var j = j_start;
            while (j + VEC_SIZE <= j_end) : (j += VEC_SIZE) {
                var b_vec: @Vector(VEC_SIZE, f32) = undefined;
                inline for (0..VEC_SIZE) |vi| {
                    b_vec[vi] = B[k * N + j + vi];
                }

                var c_vec: @Vector(VEC_SIZE, f32) = undefined;
                inline for (0..VEC_SIZE) |vi| {
                    c_vec[vi] = C[i * N + j + vi];
                }

                c_vec += a_vec * b_vec;
                inline for (0..VEC_SIZE) |vi| {
                    C[i * N + j + vi] = c_vec[vi];
                }
            }
            while (j < j_end) : (j += 1) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
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
