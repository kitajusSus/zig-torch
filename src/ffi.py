import ctypes
import numpy as np
import os
import platform
import time


def get_lib_path(lib_base_name="zigtorch"):
    system = platform.system()

    if system == "Windows":
        lib_filename = f"{lib_base_name}.dll"
    elif system == "Darwin":
        lib_filename = f"lib{lib_base_name}.dylib"
    else:
        lib_filename = f"lib{lib_base_name}.so"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, lib_filename),
        os.path.join(os.getcwd(), lib_filename),
        os.path.join(os.getcwd(), "zig-out", "lib", lib_filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Nie znaleziono biblioteki: '{lib_filename}'\n"
        f"Sprawdzono: {candidates}\n"
        f"Zbuduj bibliotekę (np. `zig build -Doptimize=ReleaseFast`)\n"
        f"lub dostosuj ścieżkę w get_lib_path()."
    )


zig_lib_path = get_lib_path()
zig_lib = ctypes.CDLL(zig_lib_path)

zig_mm_func = zig_lib.zig_mm
zig_mm_func.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
zig_mm_func.restype = None


def multiply_matrices_with_zig(A_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
    A_np = np.ascontiguousarray(A_np, dtype=np.float32)
    B_np = np.ascontiguousarray(B_np, dtype=np.float32)

    M, K_A = A_np.shape
    K_B, N = B_np.shape

    if K_A != K_B:
        raise ValueError(
            f"Niezgodne wymiary macierzy do mnożenia: "
            f"A ma {K_A} kolumn, a B ma {K_B} wierszy."
        )
    K = K_A

    C_np = np.zeros((M, N), dtype=np.float32)

    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    start_time = time.perf_counter()
    zig_mm_func(A_ptr, B_ptr, C_ptr, M, N, K)
    duration_ms = (time.perf_counter() - start_time) * 1000
    print(f"Multiply with zig ({M}x{K}) x ({K}x{N}) took : {duration_ms:.3f} ms")

    return C_np


if __name__ == "__main__":
    M_dim, K_dim, N_dim = 50, 50, 50
    matrix_a_np = np.random.rand(M_dim, K_dim).astype(np.float32)
    matrix_b_np = np.random.rand(K_dim, N_dim).astype(np.float32)

    result_zig_np = multiply_matrices_with_zig(matrix_a_np, matrix_b_np)

    start_time_np = time.perf_counter()
    result_numpy_np = np.dot(matrix_a_np, matrix_b_np)
    duration_np_ms = (time.perf_counter() - start_time_np) * 1000
    print(f"Time for numpy: {duration_np_ms:.3f} ms")

    if np.allclose(result_zig_np, result_numpy_np, rtol=1e-5, atol=1e-5):
        print("ZIG IS NOT MULTYPLYING ONLY ZEROES!!!! IT IS WORKING")
    else:
        diff = np.abs(result_zig_np - result_numpy_np)
        print("ERROS")
        print(f"  MAX ABSOLUTE DIFF: =  {np.max(diff)}")
        print(f"  MEAN DIFF: =   {np.mean(diff)}")

    rows_to_show = min(5, M_dim)
    cols_to_show = min(5, N_dim)
    print(result_zig_np[:rows_to_show, :cols_to_show])
    print(result_numpy_np[:rows_to_show, :cols_to_show])