// src/add.zig
const std = @import("std");

/// Add two matrices element-wise
pub fn addMatrices(a_ptr: [*]const f32, b_ptr: [*]const f32, result_ptr: [*]f32, len: usize) void {
    for (0..len) |i| {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}
