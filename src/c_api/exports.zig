// src/c_api/exports.zig
const std = @import("std");
const mm = @import("../ops/mm.zig");

// Eksportowana funkcja mnożenia macierzy
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

// Funkcja dodawania tensorów
pub export fn zigtorch_tensor_add(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    result_ptr: [*]f32,
    size: usize,
) callconv(.C) void {
    for (0..size) |i| {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

// Eksportuj informacje o wersji
pub export fn zigtorch_get_version() callconv(.C) [*:0]const u8 {
    return "ZigTorch 0.1.0";
}
