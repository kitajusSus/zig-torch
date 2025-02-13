import numpy as np
from .native import lib

class Tensor:
    def __init__(self, data):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
    
    def __add__(self, other):
        out = np.empty_like(self.data)
        lib.tensor_add(
            self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            other.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.data.size
        )
        return Tensor(out)
