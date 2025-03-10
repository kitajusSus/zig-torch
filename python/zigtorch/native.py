# zigtorch/native.py
import ctypes
import numpy as np
import os

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), '../zig-out/lib/libzigtorch.so')
_lib = ctypes.CDLL(_lib_path)

# Define function signatures
_lib.addMatricesFloat.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]
_lib.addMatricesFloat.restype = None

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
    _lib.addMatricesFloat(a_ptr, b_ptr, result_ptr, a_array.size)
    
    return result
