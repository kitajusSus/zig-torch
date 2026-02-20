const std = @import("std");
const mm = @import("mm");
const Io = std.Io; // new interface for time measurment 20.02.2026
//  link https://codeberg.org/ziglang/zig/commit/922ab8b8bc3b6dc14da9393b65ca2601f9a82728
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var threaded = Io.Threaded.init(.{ .gpa = allocator });
    defer threaded.deinit();
    const io = threaded.io();

    std.debug.print("Matrix Multiplication Benchmark\n\n", .{});

    benchmark(io, 64, 64, 64, 100);
    benchmark(io, 128, 128, 128, 50);
    benchmark(io, 256, 256, 256, 20);
    benchmark(io, 512, 512, 512, 10);
    benchmark(io, 1024, 1024, 1024, 5);
}

fn benchmark(io: Io, m: usize, n: usize, k: usize, iterations: usize) void {
    const allocator = std.heap.page_allocator;

    const a = allocator.alloc(f32, m * k) catch return;
    defer allocator.free(a);

    const b = allocator.alloc(f32, k * n) catch return;
    defer allocator.free(b);

    const c = allocator.alloc(f32, m * n) catch return;
    defer allocator.free(c);

    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();

    for (a) |*i| i.* = rand.float(f32);
    for (b) |*i| i.* = rand.float(f32);

    // Warmup
    mm.zig_mm(a.ptr, b.ptr, c.ptr, m, n, k);

    var total_time_ns: u64 = 0;

    for (0..iterations) |_| {
        const start = Io.Clock.awake.now(io);

        mm.zig_mm(a.ptr, b.ptr, c.ptr, m, n, k);

        const duration = start.untilNow(io, .awake);
        total_time_ns += @intCast(duration.toNanoseconds());
    }

    const avg_time_ns = total_time_ns / iterations;
    const avg_time_ms = @as(f64, @floatFromInt(avg_time_ns)) / 1_000_000.0;

    const flops = 2 * m * n * k;
    const flops_per_sec = @as(f64, @floatFromInt(flops)) / (@as(f64, @floatFromInt(avg_time_ns)) / 1_000_000_000.0);
    const gflops = flops_per_sec / 1_000_000_000.0;
    //  new interace is more explicit but easier to read and use.
    std.debug.print("Matrix size: [{d}x{d}]x[{d}x{d}]\n", .{ m, k, k, n });
    std.debug.print("  Average time: {d:.3} ms\n", .{avg_time_ms});
    std.debug.print("  Performance: {d:.2} GFLOPS\n\n", .{gflops});
}
