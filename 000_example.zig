const std = @import("std")


/// Multiplications of two marices

fn mm( A: [*]const f32, B: [*]const f32 ) !f32 {
    const I: u16;
    const J: u16;
    const K: u16;
    var C: []f32 = 0;
    var i: usize = 0;
    while (i<I) : (i += 1) {
        var j: usize = 0;
        while (j<J) : (j += 1){
            var sum: f32 = 0;
            var k: usize = 0;
            while (k<K) : (k+=1){
                sum += A[i * K + k] *  B[k*J +j];
            } 
            C[i*N + j] = sum; 
        }
    }
    return C;
}

pub fn main() void {


}
