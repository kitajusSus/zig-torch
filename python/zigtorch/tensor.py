# python/zigtorch/tensor.py
import ctypes
import numpy as np
import os
from pathlib import Path

# Load the shared library
def _get_lib_path():
    # Try to find the library in several common locations
    lib_name = {
        "Windows": "zigtorch.dll",
        "Darwin": "libzigtorch.dylib",  # macOS
        "Linux": "libzigtorch.so",
    }.get(platform.system(), "libzigtorch.so")
    
    possible_paths = [
        # Same directory as this script
        Path(__file__).parent / lib_name,
        # Standard library locations
        Path("/usr/local/lib") / lib_name,
        Path("/usr/lib") / lib_name,
        # Relative to Python package
        Path(__file__).parent.parent.parent / "zig-out" / "lib" / lib_name,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Could not find {lib_name}")

_lib = ctypes.CDLL(_get_lib_path())

# Define function signatures
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

_lib.zigtorch_get_version.argtypes = []
_lib.zigtorch_get_version.restype = ctypes.c_char_p

def get_version():
    """Return the version of the ZigTorch library"""
    return _lib.zigtorch_get_version().decode('utf-8')

def matrix_multiply(a, b):
    """Multiply two matrices using ZigTorch's optimized implementation"""
    # Convert inputs to numpy arrays with correct type and shape
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    
    # Check dimensions
    if a_array.ndim != 2 or b_array.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")
    
    if a_array.shape[1] != b_array.shape[0]:
        raise ValueError(f"Matrix dimensions don't match: {a_array.shape} and {b_array.shape}")
    
    # Create output array
    m, k = a_array.shape
    _, n = b_array.shape
    result = np.zeros((m, n), dtype=np.float32)
    
    # Get data pointers
    a_ptr = a_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the C function
    _lib.zigtorch_matrix_multiply(a_ptr, b_ptr, result_ptr, m, n, k)
    
    return result

def tensor_add(a, b):
    """Add two tensors element-wise using ZigTorch"""
    # Convert inputs to numpy arrays
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    
    # Check shapes
    if a_array.shape != b_array.shape:
        raise ValueError("Tensors must have the same shape")
    
    # Create output array
    result = np.zeros_like(a_array, dtype=np.float32)
    
    # Get data pointers
    a_ptr = a_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the C function
    _lib.zigtorch_tensor_add(a_ptr, b_ptr, result_ptr, a_array.size)
    
    return result