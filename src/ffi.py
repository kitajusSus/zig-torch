import ctypes
import numpy as np
import os
import platform
import time

def get_lib_path(lib_base_name="mm"):
    """Znajduje ścieżkę do biblioteki dynamicznej."""
    system = platform.system()
    
    if system == "Windows":
        lib_filename = f"lib{lib_base_name}.dll"
    elif system == "Darwin":  # macOS
        lib_filename = f"lib{lib_base_name}.dylib"
    else:  # Linux i inne
        lib_filename = f"lib{lib_base_name}.so"
        
    # Zakładamy, że biblioteka jest w tym samym katalogu co skrypt,
    # lub w bieżącym katalogu roboczym.
    # Sprawdź najpierw katalog skryptu.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path_script_dir = os.path.join(script_dir, lib_filename)

    if os.path.exists(lib_path_script_dir):
        return lib_path_script_dir
    
    # Jeśli nie ma w katalogu skryptu, sprawdź bieżący katalog roboczy.
    lib_path_cwd = os.path.join(os.getcwd(), lib_filename)
    if os.path.exists(lib_path_cwd):
        return lib_path_cwd

    # Jeśli używasz `zig build` z plikiem build.zig, biblioteka może być w zig-out/lib
    # Możesz dodać tę logikę, jeśli potrzebujesz:
    # zig_out_path = os.path.join(os.getcwd(), "zig-out", "lib", lib_filename)
    # if os.path.exists(zig_out_path):
    #    return zig_out_path

    raise FileNotFoundError(
        f"Nie znaleziono biblioteki: '{lib_filename}'\n"
        f"Sprawdzono: '{lib_path_script_dir}' oraz '{lib_path_cwd}'.\n"
        f"Upewnij się, że skompilowałeś 'mm.zig' (np. `zig build-lib -dynamic mm.zig`)\n"
        f"lub dostosuj ścieżkę w funkcji get_lib_path()."
    )

# Załaduj bibliotekę
try:
    zig_lib_path = get_lib_path() # Domyślna nazwa bazowa to "mm"
    print(f"Ładowanie biblioteki z: {zig_lib_path}")
    zig_lib = ctypes.CDLL(zig_lib_path)
except Exception as e:
    print(f"Błąd podczas ładowania biblioteki Zig: {e}")
    exit(1)

# Zdefiniuj sygnaturę funkcji zig_mm z Zig
# pub export fn zig_mm(
#     A_ptr: [*]const f32,
#     B_ptr: [*]const f32,
#     C_ptr: [*]f32,
#     M: usize,
#     N: usize,
#     K: usize,
# ) callconv(.C) void
try:
    zig_mm_func = zig_lib.zig_mm
    zig_mm_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A_ptr
        ctypes.POINTER(ctypes.c_float),  # B_ptr
        ctypes.POINTER(ctypes.c_float),  # C_ptr (output)
        ctypes.c_size_t,                 # M
        ctypes.c_size_t,                 # N
        ctypes.c_size_t                  # K
    ]
    zig_mm_func.restype = None  # void
except AttributeError:
    print(f"BŁĄD: Nie znaleziono funkcji 'zig_mm' w bibliotece {zig_lib_path}.")
    print("Sprawdź, czy funkcja jest poprawnie zadeklarowana jako 'pub export' w kodzie Zig.")
    exit(1)

# Funkcja opakowująca wywołanie Zigdef multiply_matrices_with_zig(A_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
def multiply_matrices_with_zig(A_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
#    Mnoży dwie macierze NumPy (A * B) używając biblioteki Zig.
 #   Macierze muszą być typu float32 i C-ciągłe.
    
    if not A_np.flags['C_CONTIGUOUS']:
        A_np = np.ascontiguousarray(A_np, dtype=np.float32)
    if not B_np.flags['C_CONTIGUOUS']:
        B_np = np.ascontiguousarray(B_np, dtype=np.float32)

    if A_np.dtype != np.float32:
        print("Konwersja macierzy A do float32")
        A_np = A_np.astype(np.float32)
    if B_np.dtype != np.float32:
        print("Konwersja macierzy B do float32")
        B_np = B_np.astype(np.float32)

    M, K_A = A_np.shape
    K_B, N = B_np.shape

    if K_A != K_B:
        raise ValueError(
            f"Niezgodne wymiary macierzy do mnożenia: "
            f"A ma {K_A} kolumn, a B ma {K_B} wierszy."
        )
    K = K_A
    
    # Utwórz macierz wynikową C (zaalokowaną przez Python/Numpy)
    # Zostanie wypełniona przez funkcję Zig
    C_np = np.zeros((M, N), dtype=np.float32) # Musi być float32
    
    # Uzyskaj wskaźniki ctypes do danych numpy
    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Wywołaj funkcję Zig
   # print(f"Wywoływanie zig_mm z M={M}, N={N}, K={K}")
    start_time = time.perf_counter()
    zig_mm_func(A_ptr, B_ptr, C_ptr, M, N, K)
    end_time = time.perf_counter()
    
    duration_ms = (end_time - start_time) * 1000
    print(f"Multiply with zig ({M}x{K}) x ({K}x{N}) took : {duration_ms:.3f} ms")
    
    return C_np

# Główna część skryptu
if __name__ == "__main__":

    # M_dim, K_dim, N_dim = 512, 256, 1024
    # M_dim, K_dim, N_dim = 128, 128, 128
    M_dim, K_dim, N_dim = 50, 50, 50 # Większy test
    # M_dim, K_dim, N_dim = 2,3,4 # B. mały test
    #M_dim, K_dim, N_dim = 65, 65, 65 # Test na granicy przełączania liczby wątków

    print(f"Creating Matrix A ({M_dim}x{K_dim}) and B ({K_dim}x{N_dim}) type float32...")
    # Użyj np.float32, ponieważ tego oczekuje funkcja Zig
    matrix_a_np = np.random.rand(M_dim, K_dim).astype(np.float32)
    matrix_b_np = np.random.rand(K_dim, N_dim).astype(np.float32)

    print("\n ZIG MULTIPLYING : !!!!!!!!!!!!!!!!!")

    result_zig_np = multiply_matrices_with_zig(matrix_a_np, matrix_b_np)
    #print(result_zig_np)

    # Dla bardzo dużych macierzy, mnożenie w Numpy też może chwilę potrwać.
    print("\n Multiply with Numpy (for comparison):")
    start_time_np = time.perf_counter()
    result_numpy_np = np.dot(matrix_a_np, matrix_b_np)
    end_time_np = time.perf_counter()
    duration_np_ms = (end_time_np - start_time_np) * 1000
    print(f"Time for numpy: {duration_np_ms:.3f} ms")

    print("\nChecking...")
    # comparison for floats
    # `atol` (absolute tolerance) 
    # `rtol` (relative tolerance) 
    if np.allclose(result_zig_np, result_numpy_np, rtol=1e-5, atol=1e-5):
        print("ZIG IS NOT MULTYPLYING ONLY ZEROES!!!! IT IS WORKING")
    else:
        diff = np.abs(result_zig_np - result_numpy_np)
        print("ERROS")
        print(f"  MAX ABSOLUTE DIFF: =  {np.max(diff)}")
        print(f"  MEAN DIFF: =   {np.mean(diff)}")
        # Pokaż kilka pierwszych elementów, gdzie jest różnica
        # for i in range(M_dim):
        #     for j in range(N_dim):
        #         if not np.isclose(result_zig_np[i,j], result_numpy_np[i,j], rtol=1e-5, atol=1e-5):
        #             print(f"  Różnica w C[{i},{j}]: Zig={result_zig_np[i,j]}, Numpy={result_numpy_np[i,j]}")
        #             if i > 5 : # Pokaż tylko kilka przykładów
        #                 exit(1)


    print(f"\n Zig matrix  ( first 5x5 OR LESS):")
    rows_to_show = min(5, M_dim)
    cols_to_show = min(5, N_dim)
    print(result_zig_np[:rows_to_show, :cols_to_show])
    
    print(f"Numpy Matrix ( first 5x5 OR LESS):")
    print(result_numpy_np[:rows_to_show, :cols_to_show])


    print("\nTest FFI zakończony.")
