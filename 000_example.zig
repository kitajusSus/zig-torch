// japierdole kocham zig i ziggersów
const std = @import("std");

/// Multiplication of two matrices
fn mm(A: []const f32, B: []const f32, I: usize, J: usize, K: usize) ![]const f32 {
    const allocator = std.heap.page_allocator;
    // Allocates memory for the resulting matrix
    var C = try allocator.alloc(f32, I * J);
    // Matrix multiplication
    var i: usize = 0;
    while (i < I) : (i += 1) {
        var j: usize = 0;
        while (j < J) : (j += 1) {
            var sum: f32 = 0;
            var k: usize = 0;
            while (k < K) : (k += 1) {
                sum += A[i * K + k] * B[k * J + j];
            }
            C[i * J + j] = sum;
        }
    }
    return C;
}

pub fn main() !void {
    const cos: [2 * 3]f32 = [_]f32{1, 2, 3, 4, 5, 6}; // 2x3 matrix
    const inne: [3 * 2]f32 = [_]f32{1, 2, 3, 4, 5, 6}; // 3x2 matrix

    // using  mm() and TRY TO handle potential allocation error
    const wynik = try mm(cos[0..], inne[0..], 2, 2, 3);

    //RESULT
    std.debug.print("Wynik: [{d}]\n", .{wynik});

    // FreeING the memory 
    const allocator = std.heap.page_allocator;
    allocator.free(wynik);
}

