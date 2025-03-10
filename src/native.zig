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

export fn tensor_add(
    a: [*]const f32,
    b: [*]const f32,
    result: [*]f32,
    size: usize,
) void {
    for (0..size) |i| {
        result[i] = a[i] + b[i];
    }
}

const add = @import("add.zig");

export fn addMatricesFloat(a_ptr: [*]const f32, b_ptr: [*]const f32, result_ptr: [*]f32, len: usize) void {
    add.addMatrices(a_ptr, b_ptr, result_ptr, len);
}
