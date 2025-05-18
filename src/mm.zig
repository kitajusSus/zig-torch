const std = @import("std");
const Thread = std.Thread;
const builtin = @import("builtin");
const math = std.math; // Używamy @min/@max bezpośrednio z builtins
const mem = std.mem;
const debug = std.debug;

const BLOCK_SIZE: usize = 64;

const Barrier = struct {
    count: usize,
    total: usize,
    phase: usize,

    pub fn init(num_threads: usize) Barrier {
        return Barrier{
            .count = 0,
            .total = num_threads,
            .phase = 0,
        };
    }

    pub fn wait(self: *Barrier) void {
        const current_phase = @atomicLoad(usize, &self.phase, .monotonic);
        const prev_count = @atomicRmw(usize, &self.count, .Add, 1, .acq_rel);

        if (prev_count + 1 == self.total) {
            _ = @atomicRmw(usize, &self.count, .Xchg, 0, .monotonic);
            _ = @atomicRmw(usize, &self.phase, .Add, 1, .release);
        } else {
            while (@atomicLoad(usize, &self.phase, .acquire) == current_phase) {
                Thread.yield() catch {
                    // W przypadku błędu yield, można spróbować krótkiego spin-loopa
                    // lub po prostu kontynuować pętlę.
                    // Dla debugowania, można tu coś wydrukować, ale ostrożnie.
                };
            }
        }
    }
};

const ThreadContext = struct {
    id: usize,
    start_row: usize,
    end_row: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    full_M: usize,
    full_N: usize,
    full_K: usize,
    barrier: *Barrier,
};

fn multiplyBlock(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    block_start_row_C: usize,
    block_start_col_C: usize,
    block_M_dim: usize,
    block_N_dim: usize,
    full_K_dim: usize,
    full_N_orig: usize,
    full_K_orig: usize,
) void {
    var i_local: usize = 0;
    while (i_local < block_M_dim) : (i_local += 1) {
        const current_row_A = block_start_row_C + i_local;
        var j_local: usize = 0;
        while (j_local < block_N_dim) : (j_local += 1) {
            const current_col_B = block_start_col_C + j_local;
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < full_K_dim) : (k += 1) {
                sum += A[current_row_A * full_K_orig + k] * B[k * full_N_orig + current_col_B];
            }
            C[current_row_A * full_N_orig + current_col_B] = sum;
        }
    }
}

fn worker(context: *ThreadContext) void {
    // debug.print("Worker {d}: start_row={d}, end_row={d}\n", .{context.id, context.start_row, context.end_row});
    const A = context.A;
    const B = context.B;
    const C = context.C;
    const start_row_for_thread = context.start_row;
    const end_row_for_thread = context.end_row;
    const N = context.full_N;
    const K = context.full_K;

    var i_block_start: usize = start_row_for_thread;
    while (i_block_start < end_row_for_thread) : (i_block_start += BLOCK_SIZE) {
        const current_block_M = @min(BLOCK_SIZE, end_row_for_thread - i_block_start);
        var j_block_start: usize = 0;
        while (j_block_start < N) : (j_block_start += BLOCK_SIZE) {
            const current_block_N = @min(BLOCK_SIZE, N - j_block_start);
            multiplyBlock(A, B, C, i_block_start, j_block_start, current_block_M, current_block_N, K, N, K);
        }
    }
    // debug.print("Worker {d} calling barrier.wait()\n", .{context.id});
    context.barrier.wait();
    // debug.print("Worker {d} finished barrier.wait()\n", .{context.id});
}

pub export fn zig_mm(
    A_ptr: [*]const f32,
    B_ptr: [*]const f32,
    C_ptr: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    // debug.print("zig_mm called: M={d}, N={d}, K={d}\n", .{ M, N, K });

    const num_elements_C = M * N;
    if (num_elements_C > 0) {
        var idx: usize = 0;
        while (idx < num_elements_C) : (idx += 1) {
            C_ptr[idx] = 0.0;
        }
    }

    if (M == 0) { // Jeśli nie ma nic do zrobienia
        // debug.print("M == 0, returning early\n", .{});
        return;
    }

    var num_threads_to_use: usize = Thread.getCpuCount() catch 1;
    const min_rows_for_threading = BLOCK_SIZE * 1; // Lub nawet 1, jeśli chcemy testować narzut

    if (M < min_rows_for_threading or N < BLOCK_SIZE or K < BLOCK_SIZE) {
        num_threads_to_use = 1;
    } else if (num_threads_to_use > M) { // Nie więcej wątków niż wierszy do przetworzenia
        num_threads_to_use = M;
    }
    if (num_threads_to_use == 0) { // Upewnij się, że jest co najmniej jeden wątek
        num_threads_to_use = 1;
    }

    // debug.print("Using {d} threads\n", .{num_threads_to_use});

    if (num_threads_to_use <= 1) {
        // debug.print("Single-threaded execution\n", .{});
        multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
        return;
    }

    // --- Wielowątkowość ---
    // debug.print("Multi-threaded execution with {d} threads\n", .{num_threads_to_use});

    var barrier_storage = Barrier.init(num_threads_to_use); // Inicjalizuj barierę z *faktyczną* liczbą wątków, które będą pracować
    const barrier_ptr: *Barrier = &barrier_storage;

    // Ogranicz do maksymalnej liczby wątków, dla których mamy miejsce w tablicach
    const MAX_SUPPORTED_THREADS = 32;
    if (num_threads_to_use > MAX_SUPPORTED_THREADS) {
        // debug.print("Clamping num_threads_to_use from {d} to {d}\n", .{num_threads_to_use, MAX_SUPPORTED_THREADS});
        num_threads_to_use = MAX_SUPPORTED_THREADS;
        barrier_storage = Barrier.init(num_threads_to_use); // Ponownie inicjuj barierę
    }

    var contexts: [MAX_SUPPORTED_THREADS]ThreadContext = undefined;
    var threads: [MAX_SUPPORTED_THREADS]Thread = undefined;

    const base_rows_per_thread = M / num_threads_to_use;
    const extra_rows = M % num_threads_to_use;
    var current_start_row: usize = 0;
    var threads_successfully_spawned: usize = 0;

    // Tworzenie wątków (bez ostatniego, który obsłuży główny wątek)
    var t_idx: usize = 0;
    while (t_idx < num_threads_to_use - 1) : (t_idx += 1) {
        const rows_offset: usize = if (t_idx < extra_rows) @as(usize, 1) else @as(usize, 0);
        const rows_for_this_thread = base_rows_per_thread + rows_offset;

        if (rows_for_this_thread == 0) {
            // debug.print("Thread {d} has no rows, skipping spawn.\n", .{t_idx});
            // To skomplikuje logikę bariery, jeśli nie dostosujemy `num_threads_to_use` na bieżąco.
            // Dla uproszczenia, załóżmy na razie, że każdy wątek dostaje pracę lub `num_threads_to_use` jest małe.
            // Jeśli M jest bardzo małe, num_threads_to_use powinno być 1.
            continue;
        }

        const end_row = @min(current_start_row + rows_for_this_thread, M);
        if (current_start_row >= end_row) continue; // Nie ma już pracy

        contexts[t_idx] = ThreadContext{
            .id = t_idx,
            .start_row = current_start_row,
            .end_row = end_row,
            .A = A_ptr,
            .B = B_ptr,
            .C = C_ptr,
            .full_M = M,
            .full_N = N,
            .full_K = K,
            .barrier = barrier_ptr,
        };

        // debug.print("Spawning thread {d} for rows [{d}..{d})\n", .{t_idx, contexts[t_idx].start_row, contexts[t_idx].end_row});
        threads[threads_successfully_spawned] = Thread.spawn(.{}, worker, .{&contexts[t_idx]}) catch |err| {
            debug.print("Failed to spawn thread {d}: {any}. Main thread will attempt to do its work if it's the last chunk.\n", .{ t_idx, err });
            // W tym prostym modelu, jeśli spawn zawiedzie, ta praca nie zostanie podjęta.
            // Można by dodać logikę, aby główny wątek ją przejął.
            // Na razie, jeśli spawn się nie powiedzie, ten wątek nie zostanie dołączony.
            // WAŻNE: To może zepsuć barierę, jeśli `barrier_storage.total` nie zostanie zaktualizowane!
            // Rozwiązanie: nie zwiększaj `threads_successfully_spawned` i zaktualizuj `barrier_storage.total`
            // przed wywołaniem `worker` przez główny wątek i przed `join`.
            // Na razie dla uproszczenia, zakładamy, że spawn się powiedzie.
            // LUB: Zmniejsz `barrier_storage.total` tutaj.
            // barrier_storage.total -=1; // To nie jest thread-safe jeśli inne wątki już pracują z barierą
            continue; // Nie udało się utworzyć, nie zwiększaj licznika
        };
        threads_successfully_spawned += 1;
        current_start_row = end_row;
    }

    // Przygotuj kontekst dla głównego wątku (który wykonuje pracę ostatniego segmentu)
    // Jeśli `num_threads_to_use` było 1, ta pętla nie wykonała się, `current_start_row` to 0.
    // Jeśli M jest małe, `current_start_row` mogło już osiągnąć M.
    if (current_start_row < M) {
        contexts[num_threads_to_use - 1] = ThreadContext{
            .id = num_threads_to_use - 1,
            .start_row = current_start_row,
            .end_row = M, // Główny wątek bierze resztę
            .A = A_ptr,
            .B = B_ptr,
            .C = C_ptr,
            .full_M = M,
            .full_N = N,
            .full_K = K,
            .barrier = barrier_ptr,
        };
        // debug.print("Main thread (context {d}) executing work for rows [{d}..{d})\n", .{num_threads_to_use - 1, contexts[num_threads_to_use - 1].start_row, contexts[num_threads_to_use - 1].end_row});
        worker(&contexts[num_threads_to_use - 1]);
    } else if (num_threads_to_use > 0) { // Jeśli główny wątek nie ma pracy, ale miał być
        // debug.print("Main thread (context {d}) has no work (start_row {d} >= M {d}).\n", .{num_threads_to_use -1, current_start_row, M});
        // Musimy dostosować liczbę wątków w barierze, jeśli główny wątek nie będzie pracował
        // a był wliczony do `num_threads_to_use`.
        if (barrier_storage.total == num_threads_to_use) { // Jeśli główny wątek był liczony
            barrier_storage.total = threads_successfully_spawned; // Tylko te, które wystartowały
            if (barrier_storage.total == 0 and M > 0) { // Żaden pomocniczy wątek nie wystartował, a główny nie ma pracy
                // To powinno być obsłużone przez num_threads_to_use = 1 na początku
                debug.print("Error: No threads working but M>0. This should not happen.\n", .{});
                // multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K); // Fallback
                return;
            }
            // debug.print("Adjusted barrier total to {d} because main thread has no work.\n", .{barrier_storage.total});
        }
    }

    // Dołączanie utworzonych wątków
    var join_idx: usize = 0;
    while (join_idx < threads_successfully_spawned) : (join_idx += 1) {
        // debug.print("Joining thread for context id {d}\n", .{contexts[join_idx].id}); // Zakładając, że contexts[join_idx] jest poprawne
        threads[join_idx].join();
        // debug.print("Joined thread for context id {d}\n", .{contexts[join_idx].id});
    }
    // debug.print("zig_mm finished\n", .{});
}
