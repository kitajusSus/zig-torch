// tests/core/mm_test.zig
const std = @import("std");
const mm = @import("../../src/mm.zig");
const testing = std.testing;

test "matrix multiplication 2x2" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    mm.zig_mm(&a, &b, &c, 2, 2, 2);

    try testing.expectEqual(@as(f32, 19.0), c[0]);
    try testing.expectEqual(@as(f32, 22.0), c[1]);
    try testing.expectEqual(@as(f32, 43.0), c[2]);
    try testing.expectEqual(@as(f32, 50.0), c[3]);
}

test "matrix multiplication 3x3" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    const b = [_]f32{ 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    var c = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    mm.zig_mm(&a, &b, &c, 3, 3, 3);

    // Verify results
    try testing.expectEqual(@as(f32, 30.0), c[0]);
    try testing.expectEqual(@as(f32, 24.0), c[1]);
    try testing.expectEqual(@as(f32, 18.0), c[2]);
    // ... check other values ...
}
