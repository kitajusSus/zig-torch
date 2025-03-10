const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "native",
        .root_source_file = .{ .path = "src/native.zig" },
        .target = target,
        .optimize = optimize,
    });

    lib.linkLibC();
    b.installArtifact(lib);
}
