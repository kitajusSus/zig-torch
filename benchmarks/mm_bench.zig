const std = @import("std");
const mm = @import("root/src/ops/mm.zig");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("Matrix Multiplication Benchmark\n\n", .{});

    try benchmark(64, 64, 64, 100);
    try benchmark(128, 128, 128, 50);
    try benchmark(256, 256, 256, 20);
    try benchmark(512, 512, 512, 10);
    try benchmark(1024, 1024, 1024, 5);
}

fn benchmark(m: usize, n: usize, k: usize, iterations: usize) !void {
    const stdout = std.io.getStdOut().writer();
    const allocator = std.heap.page_allocator;

    var a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);

    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);

    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    var prng = std.rand.DefaultPrng.init(0);
    var rand = prng.random();

    for (0..a.len) |i| {
        a[i] = rand.float(f32);
    }

    for (0..b.len) |i| {
        b[i] = rand.float(f32);
    }

    mm.zig_mm(a.ptr, b.ptr, c.ptr, m, n, k);

    var timer = try std.time.Timer.start();
    var total_time: u64 = 0;

    for (0..iterations) |_| {
        timer.reset();
        mm.zig_mm(a.ptr, b.ptr, c.ptr, m, n, k);
        total_time += timer.read();
    }

    const avg_time_ns = total_time / iterations;
    const avg_time_ms = @as(f64, @floatFromInt(avg_time_ns)) / 1_000_000.0;

    const flops = 2 * m * n * k;
    const flops_per_sec = @as(f64, @floatFromInt(flops)) / (@as(f64, @floatFromInt(avg_time_ns)) / 1_000_000_000.0);
    const gflops = flops_per_sec / 1_000_000_000.0;

    try stdout.print("Matrix size: [{d}x{d}]x[{d}x{d}]\n", .{ m, k, k, n });
    try stdout.print("  Average time: {d:.3} ms\n", .{avg_time_ms});
    try stdout.print("  Performance: {d:.2} GFLOPS\n\n", .{gflops});
}
