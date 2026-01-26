const std = @import("std");

/// Matrix multiplication: C = A × B
/// A is M×K matrix, B is K×N matrix, C is M×N matrix (row-major order)
pub export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.c) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
