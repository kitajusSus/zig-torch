import torch
import numpy as np
import os
import ctypes
from pathlib import Path

# Find and load the native library
def _find_lib():
    # Look in package directory first
    package_dir = Path(__file__).parent
    candidates = [
        package_dir / "libnative.so",  # Linux
        package_dir / "libnative.dylib",  # macOS
        package_dir / "libnative.dll",  # Windows
    ]
    
    for lib_path in candidates:
        if lib_path.exists():
            return str(lib_path)
    
    # If not found, raise an error
    raise ImportError("Could not find ZigTorch native library. Please reinstall the package.")

# Load the library
_lib = ctypes.CDLL(_find_lib())

# Configure function signatures
_lib.matrix_multiply.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_size_t,  # M
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # K
]
_lib.matrix_multiply.restype = None

def mm(tensor1, tensor2):
    """
    Matrix multiplication using the Zig implementation.
    
    Args:
        tensor1 (torch.Tensor): First input tensor with shape (M, K)
        tensor2 (torch.Tensor): Second input tensor with shape (K, N)
    
    Returns:
        torch.Tensor: Result tensor with shape (M, N)
    """
    # Validate inputs
    if tensor1.dim() != 2 or tensor2.dim() != 2:
        raise ValueError("Both tensors must be 2-dimensional")
    
    if tensor1.size(1) != tensor2.size(0):
        raise ValueError(
            f"Tensor dimensions mismatch for multiplication: {tensor1.size()} and {tensor2.size()}"
        )
    
    # Ensure contiguous layout and float32 type for both tensors
    tensor1 = tensor1.contiguous().float()
    tensor2 = tensor2.contiguous().float()
    
    # Get dimensions
    M, K = tensor1.size()
    _, N = tensor2.size()
    
    # Create output tensor
    output = torch.empty(M, N, dtype=torch.float32)
    
    # Get pointers to tensor data
    tensor1_ptr = ctypes.cast(tensor1.data_ptr(), ctypes.POINTER(ctypes.c_float))
    tensor2_ptr = ctypes.cast(tensor2.data_ptr(), ctypes.POINTER(ctypes.c_float))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))
    
    # Call the Zig function
    _lib.matrix_multiply(
        tensor1_ptr, 
        tensor2_ptr, 
        output_ptr, 
        M, 
        N, 
        K
    )
    
    return output

# Version information
__version__ = "0.1.0"
