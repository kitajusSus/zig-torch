// build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Static library for Zig users
    const static_lib = b.addStaticLibrary(.{
        .name = "zigtorch",
        .root_source_file = .{ .path = "src/zigtorch.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(static_lib);

    // 2. Shared library for C/Python users
    const shared_lib = b.addSharedLibrary(.{
        .name = "zigtorch",
        .root_source_file = .{ .path = "src/c_api/exports.zig" },
        .target = target,
        .optimize = optimize,
    });
    shared_lib.linkLibC();
    b.installArtifact(shared_lib);

    // 3. Tests
    const lib_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/zigtorch.zig" },
        .target = target,
        .optimize = optimize,
    });

    const mm_tests = b.addTest(.{
        .root_source_file = .{ .path = "tests/core/mm_test.zig" },
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&lib_tests.step);
    test_step.dependOn(&mm_tests.step);

    // 4. Benchmarks
    const mm_bench = b.addExecutable(.{
        .name = "mm_benchmark",
        .root_source_file = .{ .path = "tests/benchmark/mm_bench.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(mm_bench);

    const run_bench = b.addRunArtifact(mm_bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
