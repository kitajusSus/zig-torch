import numpy as np
import pytest

from zigtorch import matrix_multiply

@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 3), (3, 2)),
        ((4, 4), (4, 4)),
        ((16, 16), (16, 16)),
    ],
)
def test_zig_mm_matches_numpy(shape_a, shape_b):
    A = np.random.rand(*shape_a).astype(np.float32)
    B = np.random.rand(*shape_b).astype(np.float32)

    zig_out = matrix_multiply(A, B)
    np_out = A @ B

    assert np.allclose(zig_out, np_out, rtol=1e-5, atol=1e-5)

def test_dimension_mismatch_raises():
    A = np.random.rand(2, 3).astype(np.float32)
    B = np.random.rand(4, 2).astype(np.float32)

    with pytest.raises(ValueError):
        matrix_multiply(A, B)