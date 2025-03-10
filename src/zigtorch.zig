// src/zigtorch.zig
const std = @import("std");

pub const ops = struct {
    pub const mm = @import("ops/mm.zig");
    // adding new operations here
};

// verson
pub const VERSION = "0.1.0";

// info
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;

    pub fn toString() []const u8 {
        return "0.1.0";
    }
};
