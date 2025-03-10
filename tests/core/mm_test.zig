const std = @import("std");
const testing = std.testing;
const mm = @import("zigtorch").ops.mm;

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
