import numpy as np
from ffi import multiply_matrices_with_zig

A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)

C = multiply_matrices_with_zig(A, B)
print(C)
