const std = @import("std");
const mm = @import("mm");

pub fn main() !void {
    std.debug.print("Matrix Multiplication Benchmark\n\n", .{});

    benchmark(64, 64, 64, 100);
    benchmark(128, 128, 128, 50);
    benchmark(256, 256, 256, 20);
    benchmark(512, 512, 512, 10);
    benchmark(1024, 1024, 1024, 5);
}

fn benchmark(m: usize, n: usize, k: usize, iterations: usize) void {
    const allocator = std.heap.page_allocator;

    const a = allocator.alloc(f32, m * k) catch {
        std.debug.print("Failed to allocate memory for matrix A\n", .{});
        return;
    };
    defer allocator.free(a);

    const b = allocator.alloc(f32, k * n) catch {
        std.debug.print("Failed to allocate memory for matrix B\n", .{});
        return;
    };
    defer allocator.free(b);

    const c = allocator.alloc(f32, m * n) catch {
        std.debug.print("Failed to allocate memory for matrix C\n", .{});
        return;
    };
    defer allocator.free(c);

    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();

    for (0..a.len) |i| {
        a[i] = rand.float(f32);
    }

    for (0..b.len) |i| {
        b[i] = rand.float(f32);
    }

    mm.zig_mm(a.ptr, b.ptr, c.ptr, m, n, k);

    var timer = std.time.Timer.start() catch {
        std.debug.print("Failed to start timer\n", .{});
        return;
    };
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

    std.debug.print("Matrix size: [{d}x{d}]x[{d}x{d}]\n", .{ m, k, k, n });
    std.debug.print("  Average time: {d:.3} ms\n", .{avg_time_ms});
    std.debug.print("  Performance: {d:.2} GFLOPS\n\n", .{gflops});
}
