import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from zigtorch import matrix_multiply

# Create random test matrices
A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)

print("A:\n", A)
print("B:\n", B)

# Compute matrix multiplication using ZigTorch
C_zig = matrix_multiply(A, B)
print("C (ZigTorch A @ B):\n", C_zig)

# Compute matrix multiplication using NumPy for comparison
C_numpy = A @ B
print("C (NumPy A @ B):\n", C_numpy)

# Verify correctness
max_diff = np.max(np.abs(C_zig - C_numpy))
print(f"\nMax difference between ZigTorch and NumPy: {max_diff:.2e}")

assert np.allclose(C_zig, C_numpy, atol=1e-5)
print("✓ Verification successful! ZigTorch matrix multiplication matches NumPy.")
