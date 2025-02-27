const std = @import("std");
const mm = @import("mm.zig");

// Export matrix multiplication function for Python integration
export fn matrix_multiply(
    A_ptr: [*]const f32,
    B_ptr: [*]const f32,
    C_ptr: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    // Forward the call to the optimized implementation
    mm.zig_mm(A_ptr, B_ptr, C_ptr, M, N, K);
}
