import sys
import os
import numpy as np

# Add project root to path so we can import zigtorch if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from zigtorch import matrix_multiply

A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)

print("A:\n", A)
print("B:\n", B)

C = matrix_multiply(A, B)
print("C (result):\n", C)

# Verify
assert np.allclose(C, A @ B, atol=1e-5)
print("Verification successful!")
