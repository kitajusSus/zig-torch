const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shared library
    const lib = b.addSharedLibrary(.{
        .name = "zigtorch",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });

    b.installArtifact(lib);

    // Main tests
    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run Zig tests");
    const run_main_tests = b.addRunArtifact(main_tests);
    test_step.dependOn(&run_main_tests.step);

    // Core tests with mm module
    const core_tests = b.addTest(.{
        .root_source_file = b.path("core/mm_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    core_tests.root_module.addImport("mm", b.createModule(.{
        .root_source_file = b.path("src/ops/mm.zig"),
    }));

    const core_test_step = b.step("test-core", "Run core tests");
    const run_core_tests = b.addRunArtifact(core_tests);
    core_test_step.dependOn(&run_core_tests.step);

    // Benchmark executable
    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_source_file = b.path("benchmarks/mm_bench.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_exe.root_module.addImport("mm", b.createModule(.{
        .root_source_file = b.path("src/ops/mm.zig"),
    }));

    b.installArtifact(bench_exe);

    const bench_step = b.step("bench", "Run benchmark");
    const run_bench = b.addRunArtifact(bench_exe);
    bench_step.dependOn(&run_bench.step);
}
