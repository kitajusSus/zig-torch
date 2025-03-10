// src/zigtorch.zig - Main library entry point
const std = @import("std");

// Export core functionality
pub const ops = struct {
    pub const mm = @import("core/ops/mm.zig");
    // other operations
};

// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;

    pub fn versionString() []const u8 {
        return "0.1.0";
    }
};
