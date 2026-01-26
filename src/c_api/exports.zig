const std = @import("std");
const mm = @import("../ops/mm.zig");
const add = @import("../ops/add.zig");

pub export fn zigtorch_matrix_multiply(
    A_ptr: [*]const f32,
    B_ptr: [*]const f32,
    C_ptr: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    mm.zig_mm(A_ptr, B_ptr, C_ptr, M, N, K);
}

pub export fn zigtorch_tensor_add(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    result_ptr: [*]f32,
    size: usize,
) callconv(.C) void {
    add.addMatrices(a_ptr, b_ptr, result_ptr, size);
}

pub export fn zigtorch_version() callconv(.C) [*:0]const u8 {
    return "ZigTorch 0.1.0";
}
