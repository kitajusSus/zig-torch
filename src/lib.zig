const std = @import("std");

const c_api = @import("c_api/exports.zig");

pub const ops = c_api.ops;

comptime {
    _ = c_api;
}
