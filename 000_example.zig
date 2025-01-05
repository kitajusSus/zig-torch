const std = @import("std");


/// Multiplications of two marices

fn mm( A: [*]const f32, B: [*]const f32,  I: usize, J: usize,  K: usize ) ![]f32 {
    const allocator = std.heap.page_allocator;
    var C = try allocator.alloc(f32, I*J);
    defer allocator.free(C);
    var i: usize = 0;
    while (i<I) : (i += 1) {
        var j: usize = 0;
        while (j<J) : (j += 1){
            var sum: f32 = 0;
            var k: usize = 0;
            while (k<K) : (k+=1){
                sum += A[i * K + k] *  B[k*J + j];
            } 
            C[i*J + j] = sum; 
        }
    }
    return C;
}

pub fn main() !void {
    const cos: [2*3]f32 = [_]f32{1, 2, 3, 4, 5, 6};
    const inne: [3*2]f32 = [_]f32{1, 2, 3, 4, 5, 6};
    const wynik = try mm(cos[0..], inne[0..],2,2,3) catch unreachable;
    std.debug.print("C: {d}\n", .{wynik});

}
