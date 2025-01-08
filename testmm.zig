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
                            sum += A[rowA + kk + 0] * B[(kk + 0) * N + jj] + A[rowA + kk + 1] * B[(kk + 1) * N + jj] + A[rowA + kk + 2] * B[(kk + 2) * N + jj] + A[rowA + kk + 3] * B[(kk + 3) * N + jj];
                        }

                        // Resztka (gdy length nie jest wielokrotnością 4)
                        while (kk < k_end) : (kk += 1) {
                            sum += A[rowA + kk] * B[kk * N + jj];
                        }

                        // Dodajemy wynik do C (być może w C już coś było, więc +=)
                        // mozna albo nadpisać albo dodawać
                        // widze roznice w logice i chyba -> jest logiczniejsze ale nie wiem C[rowC + jj] = sum;
                        std.debug.print("Sum = {}, rowC = {}, jj = {}\n", .{ sum, rowC, jj });
                        C[rowC + jj] = sum;
                    }
                }
            }
        }
    }
}

// Przykład użycia funkcji printMatrix po wywołaniu worker
fn printMatrix(C: [*]const f32, M: usize, N: usize) void {
    for (C[0 .. M * N]) |value| {
        std.debug.print("{}, ", .{value});
    }
    std.debug.print("\n", .{});
    std.debug.print("\n", .{});
}

pub fn main() void {
    const M = 4;
    const N = 4;
    const K = 4;
    var A: [M * K]f32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var B: [K * N]f32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var C: [M * N]f32 = [_]f32{0} ** (M * N);

    worker(0, M, &A, &B, &C, M, N, K);
    printMatrix(&C, M, N);
}
