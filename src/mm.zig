const std = @import("std");
const Thread = std.Thread;
const builtin = @import("builtin");
// const math = std.math; // Używamy @min/@max z builtins
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
    //debug.print("Worker {d}: start_row={d}, end_row={d}\n", .{context.id, context.start_row, context.end_row});
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
}

pub export fn zig_mm(
    A_ptr: [*]const f32,
    B_ptr: [*]const f32,
    C_ptr: [*]f32,
    M: usize,
    N: usize,
    K: usize,
) callconv(.C) void {
    //debug.print("zig_mm called: M={d}, N={d}, K={d}\n", .{ M, N, K });

    const num_elements_C = M * N;
    if (num_elements_C > 0) {
        var idx: usize = 0;
        while (idx < num_elements_C) : (idx += 1) {
            C_ptr[idx] = 0.0;
        }
    }

    if (M == 0) {
        //debug.print("M == 0, returning early\n", .{});
        return;
    }

    const num_threads_initial_request: usize = Thread.getCpuCount() catch 1;
    const min_rows_for_threading = BLOCK_SIZE * 1;

    var final_num_threads_to_use: usize = 0; // Zadeklaruj bez inicjalizacji

    if (M < min_rows_for_threading or N < BLOCK_SIZE or K < BLOCK_SIZE) {
        final_num_threads_to_use = 1;
    } else if (num_threads_initial_request > M) {
        final_num_threads_to_use = M;
    } else if (num_threads_initial_request == 0) {
        final_num_threads_to_use = 1;
    } else {
        final_num_threads_to_use = num_threads_initial_request;
    }

    //debug.print("Calculated final_num_threads_to_use: {d}\n", .{final_num_threads_to_use});

    if (final_num_threads_to_use <= 1) {
        //debug.print("Single-threaded execution path chosen.\n", .{});
        multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
        return;
    }

    //debug.print("Multi-threaded execution path with initially {d} threads\n", .{final_num_threads_to_use});

    const MAX_SUPPORTED_THREADS = 32;
    if (final_num_threads_to_use > MAX_SUPPORTED_THREADS) {
        //debug.print("Clamping final_num_threads_to_use from {d} to {d}\n", .{final_num_threads_to_use, MAX_SUPPORTED_THREADS});
        final_num_threads_to_use = MAX_SUPPORTED_THREADS;
    }

    var contexts: [MAX_SUPPORTED_THREADS]ThreadContext = undefined;
    var threads: [MAX_SUPPORTED_THREADS]Thread = undefined;

    var temp_actual_working_threads: usize = 0;
    var temp_current_start_row_for_calc: usize = 0;
    const temp_base_rows_per_thread = M / final_num_threads_to_use;
    const temp_extra_rows = M % final_num_threads_to_use;

    for (0..final_num_threads_to_use) |i_calc| {
        const rows_offset_calc: usize = if (i_calc < temp_extra_rows) @as(usize, 1) else @as(usize, 0);
        const rows_for_this_thread_calc = temp_base_rows_per_thread + rows_offset_calc;

        if (rows_for_this_thread_calc == 0) continue;

        const end_row_calc = @min(temp_current_start_row_for_calc + rows_for_this_thread_calc, M);
        if (temp_current_start_row_for_calc >= end_row_calc) continue;

        temp_actual_working_threads += 1;
        temp_current_start_row_for_calc = end_row_calc;
    }

    if (temp_actual_working_threads == 0 and M > 0) {
        //debug.print("No actual working threads after distribution (M > 0). Forcing single thread.\n", .{});
        multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
        return;
    }
    // Jeśli M == 0, temp_actual_working_threads będzie 0, co jest ok, bo wróciliśmy na początku.

    //debug.print("Actual working threads determined: {d}\n", .{temp_actual_working_threads});

    // Jeśli po kalkulacji wyszło, że tylko 1 wątek ma pracować, idź ścieżką jednowątkową
    if (temp_actual_working_threads <= 1 and M > 0) { // Dodano M > 0
        //debug.print("Fallback to single-threaded after calculating actual workers.\n", .{});
        multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
        return;
    }
    if (temp_actual_working_threads == 0) { // Jeśli M było 0, temp_actual_working_threads będzie 0
        return;
    }

    // Krok 2: Inicjalizuj barierę z *faktyczną* liczbą pracujących wątków
    var barrier_storage = Barrier.init(temp_actual_working_threads);
    const barrier_ptr: *Barrier = &barrier_storage;

    // Krok 3: Uruchom wątki pomocnicze i przygotuj główny wątek
    var threads_spawned_count: usize = 0; // Licznik faktycznie utworzonych wątków pomocniczych
    var main_thread_context_idx: usize = 0; // Indeks w contexts dla głównego wątku (ustawimy go)
    var main_thread_has_work: bool = false;

    var current_start_row_for_dispatch: usize = 0;
    var current_active_context_idx: usize = 0; // Indeks dla tablicy contexts[] i threads[]

    for (0..final_num_threads_to_use) |original_slot_idx_dispatch| {
        const rows_offset_dispatch: usize = if (original_slot_idx_dispatch < temp_extra_rows) @as(usize, 1) else @as(usize, 0);
        const rows_for_this_slot_dispatch = temp_base_rows_per_thread + rows_offset_dispatch;
        if (rows_for_this_slot_dispatch == 0) continue;

        const actual_start_row_for_dispatch = current_start_row_for_dispatch;
        const actual_end_row_for_dispatch = @min(actual_start_row_for_dispatch + rows_for_this_slot_dispatch, M);
        if (actual_start_row_for_dispatch >= actual_end_row_for_dispatch) continue;

        contexts[current_active_context_idx] = ThreadContext{
            .id = current_active_context_idx,
            .start_row = actual_start_row_for_dispatch,
            .end_row = actual_end_row_for_dispatch,
            .A = A_ptr,
            .B = B_ptr,
            .C = C_ptr,
            .full_M = M,
            .full_N = N,
            .full_K = K,
            .barrier = barrier_ptr,
        };

        // Pierwsze `temp_actual_working_threads - 1` kontekstów dla wątków pomocniczych
        // Ostatni kontekst dla głównego wątku
        if (current_active_context_idx < temp_actual_working_threads - 1) {
            //debug.print("Spawning thread (active_ctx_idx {d}) for rows [{d}..{d})\n", .{current_active_context_idx, contexts[current_active_context_idx].start_row, contexts[current_active_context_idx].end_row});
            if (threads_spawned_count < threads.len) { // Upewnij się, że nie wychodzisz poza tablicę threads
                threads[threads_spawned_count] = Thread.spawn(.{}, worker, .{&contexts[current_active_context_idx]}) catch |err| {
                    debug.print("CRITICAL: Failed to spawn thread for active_ctx_idx {d}: {any}. Execution will be single-threaded.\n", .{ current_active_context_idx, err });
                    multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
                    return; // Awaryjne wyjście do jednowątkowego
                };
                threads_spawned_count += 1;
            } else {
                debug.print("CRITICAL ERROR: Ran out of space in 'threads' array. This should not happen.\n", .{});
                multiplyBlock(A_ptr, B_ptr, C_ptr, 0, 0, M, N, K, N, K);
                return;
            }
        } else {
            main_thread_context_idx = current_active_context_idx;
            main_thread_has_work = true;
            //debug.print("Main thread will be context {d} for rows [{d}..{d})\n", .{contexts[main_thread_context_idx].id, contexts[main_thread_context_idx].start_row, contexts[main_thread_context_idx].end_row});
        }

        current_active_context_idx += 1;
        current_start_row_for_dispatch = actual_end_row_for_dispatch;
        if (current_active_context_idx >= temp_actual_working_threads) break; // Przydzielono pracę dla wszystkich pracujących wątków
    }

    // Główny wątek wykonuje swoją pracę
    if (main_thread_has_work) {
        //debug.print("Main thread (id {d}) actually working on rows [{d}..{d})\n", .{contexts[main_thread_context_idx].id, contexts[main_thread_context_idx].start_row, contexts[main_thread_context_idx].end_row});
        worker(&contexts[main_thread_context_idx]);
    }

    // Dołączanie utworzonych wątków
    var join_idx: usize = 0;
    while (join_idx < threads_spawned_count) : (join_idx += 1) {
        //debug.print("Joining thread for context id {d}\n", .{contexts[join_idx].id}); // Uwaga: contexts[join_idx].id będzie poprawne
        threads[join_idx].join();
    }
    //debug.print("zig_mm finished\n", .{});
}
