// japierdole kocham zig i ziggersów
const std = @import("std");

/// Rozmiar bloku do cache blocking:
/// - 64 bywa dobrym punktem startowym, ale warto testować 32, 128, itp.
pub const BlockSize = 64;
fn getCpuCount() usize {
    // std.os.sysconf(.nprocessors_onln) zwraca !i64, więc obsługujemy błąd przez `catch`.
    // Jeśli wystąpi błąd -> używamy -1.
    //const raw_count = std.os.sysconf(.nprocessors_onln) catch -1;

    // Jeśli raw_count jest <= 0, ustawiamy 1:
    //if (raw_count < 1) {
        return 4;
    //} else {
        // Tutaj powinna zadziałać wbudowana funkcja @intCast(typ, wartość).
        // Jeśli nadal masz komunikat 'expected 1 argument, found 2',
        // sprawdź, czy nie masz konfliktu nazw w kodzie lub błędu składni wyżej.
      //  return @as(usize, @intCast(raw_count));
    //}
}
fn worker(
    start_i: usize,
    end_i: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    // Przechodzimy "po blokach" w wymiarze i, j, k.
    // Wewnątrz pętli robimy unrolling w wymiarze k.
    _ = M;
    var i: usize = start_i;
    while (i < end_i) : (i += BlockSize) {
        var j: usize = 0;
        while (j < N) : (j += BlockSize) {
            var k: usize = 0;
            while (k < K) : (k += BlockSize) {
                const i_end = @min(i + BlockSize, end_i);
                const j_end = @min(j + BlockSize, N);
                const k_end = @min(k + BlockSize, K);

                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    const rowA = ii * K;
                    const rowC = ii * N;

                    var jj: usize = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = 0.0;
                        var kk: usize = k;
                        // Tu zaczynamy unrolling w pętli "po K":
                        const length = k_end - k;
                        const unroll4 = length - (length % 4);

                        // Najpierw sumujemy pełne "czwórki"
                        while ((kk - k) < unroll4) : (kk += 4) {
                            sum += A[rowA + kk + 0] * B[(kk + 0) * N + jj]
                                 + A[rowA + kk + 1] * B[(kk + 1) * N + jj]
                                 + A[rowA + kk + 2] * B[(kk + 2) * N + jj]
                                 + A[rowA + kk + 3] * B[(kk + 3) * N + jj];
                        }

                        // Resztka (gdy length nie jest wielokrotnością 4)
                        while (kk < k_end) : (kk += 1) {
                            sum += A[rowA + kk] * B[kk * N + jj];
                        }

                        // Dodajemy wynik do C (być może w C już coś było, więc +=)
                        // Jeśli chcesz nadpisać (zamiast dodawać) użyj: C[rowC + jj] = sum;
                        C[rowC + jj] += sum;
                    }
                }
            }
        }
    }
}


export fn zig_mm(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    const thread_count = getCpuCount();
    var handles: [4]?std.Thread = undefined;

    const chunk_size = (M + thread_count - 1) / thread_count;

    for (0..thread_count) |t| {
        const start_i = t * chunk_size;
        const end_i = if (start_i + chunk_size <= M) start_i + chunk_size else M;

        if (start_i < end_i) {
            handles[t] = std.Thread.spawn(
                .{},
                worker,
                .{start_i, end_i, A, B, C, M, N, K},
            ) catch |err| blk: {
                std.debug.print("failed do spawn thread: {}\n", .{err});
                break :blk null;
            };
        } else {
            handles[t] = null;
        }
    }

    for (handles) |maybe_thread| {
        if (maybe_thread) |thread| {
            thread.join();
        }
    }
}

