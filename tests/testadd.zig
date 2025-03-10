// tests/testadd.zig
const std = @import("std");
const testing = std.testing;
const add = @import("../src/add.zig");

test "matrix addition" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var result = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    add.addMatrices(&a, &b, &result, 4);

    try testing.expectEqual(@as(f32, 6.0), result[0]);
    try testing.expectEqual(@as(f32, 8.0), result[1]);
    try testing.expectEqual(@as(f32, 10.0), result[2]);
    try testing.expectEqual(@as(f32, 12.0), result[3]);
}
