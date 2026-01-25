import ctypes
import os
import platform
from pathlib import Path
import numpy as np

def _get_lib_name(base: str = "zigtorch") -> str:
    system = platform.system()
    if system == "Windows":
        return f"{base}.dll"
    if system == "Darwin":
        return f"lib{base}.dylib"
    return f"lib{base}.so"

def _get_lib_path() -> str:
    lib_name = _get_lib_name()
    here = Path(__file__).resolve().parent
    candidates = [
        here / lib_name,
        here.parent.parent / "zig-out" / "lib" / lib_name,
        Path.cwd() / "zig-out" / "lib" / lib_name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Could not find {lib_name} in {candidates}")

_lib = ctypes.CDLL(_get_lib_path())

_lib.zigtorch_matrix_multiply.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.zigtorch_matrix_multiply.restype = None

_lib.zigtorch_tensor_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.zigtorch_tensor_add.restype = None

_lib.zigtorch_version.argtypes = []
_lib.zigtorch_version.restype = ctypes.c_char_p

def version() -> str:
    return _lib.zigtorch_version().decode("utf-8")

def matrix_multiply(a, b):
    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)

    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")
    if a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(f"Matrix dimensions do not match: {a_arr.shape} vs {b_arr.shape}")

    m, k = a_arr.shape
    _, n = b_arr.shape
    c_arr = np.zeros((m, n), dtype=np.float32)

    _lib.zigtorch_matrix_multiply(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m, n, k,
    )
    return c_arr

def tensor_add(a, b):
    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)

    if a_arr.shape != b_arr.shape:
        raise ValueError("Tensors must have the same shape")

    out = np.zeros_like(a_arr, dtype=np.float32)

    _lib.zigtorch_tensor_add(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a_arr.size,
    )
    return out