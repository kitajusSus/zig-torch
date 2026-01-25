# zigtorch/native.py
import ctypes
import numpy as np
import os
import platform
from pathlib import Path


def _get_lib_path():
    """Find the ZigTorch shared library in common locations."""
    lib_name = {
        "Windows": "libzigtorch.dll",
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
        # Fallback for development
        Path(__file__).parent.parent / "zig-out" / "lib" / lib_name,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Could not find {lib_name}")


# Load the shared library
_lib_path = _get_lib_path()
_lib = ctypes.CDLL(_lib_path)

# Define function signatures for tensor add
_lib.zigtorch_tensor_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]
_lib.zigtorch_tensor_add.restype = None


def add_matrices(a, b):
    """Add two matrices element-wise"""
    # Convert inputs to numpy arrays with proper dtype
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    
    # Check shapes
    if a_array.shape != b_array.shape:
        raise ValueError("Matrices must have the same shape")
    
    # Create output array
    result = np.zeros_like(a_array, dtype=np.float32)
    
    # Get data pointers
    a_ptr = a_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the C function
    _lib.zigtorch_tensor_add(a_ptr, b_ptr, result_ptr, a_array.size)
    
    return result
