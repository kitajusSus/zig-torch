// japierdole kocham zig
const std = @import("std");

/// Rozmiar bloku do cache blocking:
/// - 64 defaultowo, ale warto testować 32, 128, itp.
pub const BlockSize = 64;
fn getCpuCount() usize {
    // std.os.sysconf(.nprocessors_onln) zwraca !i64, więc obsługujemy błąd przez `catch`.
    // Jeśli wystąpi błąd -> używamy -1.
    //const raw_count = std.os.sysconf(.nprocessors_onln) catch -1;
    // na początku myslałem ze to ma sens ale juz pogodze sie z faktem ze bedzie stale 4 rdzenie wykorzystywać
    // *DO POPRAWY*
    // Jeśli raw_count jest <= 0, ustawiamy 1:
    //if (raw_count < 1) {
        return 4;
    //} else {
        // Tutaj powinna zadziałać wbudowana funkcja @intCast(typ, wartość) ale intcast niedziała trzeba szukać rozwiązania bo do kurwy nedzy .
       
      //  return @as(usize, @intCast(raw_count));
    //}
    //
}
//pracuś bedzie robił robote i nosił bloki w kopalni
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
    // wzorowałem się na metodzie `cache blocking` lub `loop blocking`, czyli podział macierzy na mniejsze macierze (blocki) które mieszczą się wygodnie w ram procesora (cache) 
    // rozmiar bloku defaulotowo jest ustawione na 64 look at BlockSize [musi byc to liczba  2^n] 
    // Wewnątrz pętli robimy unrolling w wymiarze k. czyli petla po literze k jest rozpakowywana w celu zmniejszenia liczby potrzebnych iteracji i do tego stosowana jest nasza wielowątkowosc która ustawiłem na 4, sumuje 4 elementy na raz i zwieksza wydajnosc. 
    _ = M; // wyrzucało blad, wiec zrobiłem pusta zmienna 
    var i: usize = start_i;
    // petla po i czyli ilosc kolumn w macierzy A
    while (i < end_i) : (i += BlockSize) {
        var j: usize = 0;
        //petla po j czyli ilosc wierszy A i ilosc kolumn B
        while (j < N) : (j += BlockSize) {
            var k: usize = 0;
            //petla k czyli ilosc wierszy B ponieważ macierze wcale nie musza być kwadratami, 
            while (k < K) : (k += BlockSize) {
                const i_end = @min(i + BlockSize, end_i); //@min zwraca mniejsza wartosc z tych 2 tak by nie uciec po za macierz bo po co liczyć cos co mnie nie interesuje?
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
                        // Tu zaczynamy unrolling tzw: rozpakowywanie w pętli "po K":
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
                        // mozna albo nadpisać albo dodawać
                        // widze roznice w logice i chyba -> jest logiczniejsze ale nie wiem C[rowC + jj] = sum;
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
    const thread_count = getCpuCount(); //defaultowo 4
    var handles: [thread_count]?std.Thread = undefined;

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

