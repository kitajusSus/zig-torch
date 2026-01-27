const std = @import("std");

pub const ops = struct {
    pub const mm = @import("ops/mm.zig");
    pub const add = @import("ops/add.zig");
};

comptime {
    _ = @import("c_api/exports.zig");
}

test {
    std.testing.refAllDecls(@This());
}
