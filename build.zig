const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zigtorch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mm.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Dodaj ZigTorch jako moduł
    // const zigtorch_dep = b.dependency("zigtorch", .{
    //     .target = target,
    //     .optimize = optimize,
    // });
    // const zigtorch_mod = zigtorch_dep.module("zigtorch");
    // exe.addModule("zigtorch", zigtorch_mod);

    b.installArtifact(exe);
}
