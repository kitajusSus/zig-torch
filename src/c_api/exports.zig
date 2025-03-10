// src/c_api/exports.zig
const std = @import("std");
const mm = @import("../mm.zig");

// Export matrix multiplication function with C calling convention
export fn zigtorch_matrix_multiply(
    A_ptr: [*]const f32,
    B_ptr: [*]const f32,
    C_ptr: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    mm.zig_mm(A_ptr, B_ptr, C_ptr, M, N, K);
}

// Export tensor addition with C calling convention
export fn zigtorch_tensor_add(
    a: [*]const f32,
    b: [*]const f32,
    result: [*]f32,
    size: usize,
) callconv(.C) void {
    for (0..size) |i| {
        result[i] = a[i] + b[i];
    }
}

// Add a version function
export fn zigtorch_get_version() callconv(.C) [*:0]const u8 {
    return "ZigTorch 0.1.0";
}
