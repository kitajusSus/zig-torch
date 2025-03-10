// examples/neural_network.zig
const std = @import("std");
const zigtorch = @import("zigtorch");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const allocator = std.heap.page_allocator;

    // Define matrices for a simple neural network
    // Input layer (2 neurons) to hidden layer (3 neurons)
    const weights1 = [_]f32{
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
    };

    // Hidden layer (3 neurons) to output layer (1 neuron)
    const weights2 = [_]f32{
        0.7, 0.8, 0.9,
    };

    // Input data
    const input = [_]f32{ 1.0, 2.0 };

    // Allocate buffers for intermediate and final results
    var hidden = [_]f32{ 0.0, 0.0, 0.0 };
    var output = [_]f32{0.0};

    // Forward pass through the network
    // Input layer to hidden layer: hidden = weights1 * input
    zigtorch.ops.mm.zig_mm(&weights1, &input, &hidden, 3, 2, 1);

    // Apply ReLU activation
    for (0..hidden.len) |i| {
        hidden[i] = if (hidden[i] > 0) hidden[i] else 0;
    }

    // Hidden layer to output layer: output = weights2 * hidden
    zigtorch.ops.mm.zig_mm(&weights2, &hidden, &output, 1, 3, 1);

    try stdout.print("Simple Neural Network Example\n", .{});
    try stdout.print("Input: [{d}, {d}]\n", .{ input[0], input[1] });
    try stdout.print("Hidden layer: [{d:.4}, {d:.4}, {d:.4}]\n", .{ hidden[0], hidden[1], hidden[2] });
    try stdout.print("Output: {d:.4}\n", .{output[0]});
}
