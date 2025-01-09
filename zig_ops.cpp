#include <torch/extension.h>
#include <vector>

// Declaration of the Zig function
extern "C" void zig_mm(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);

// Wrapper function for PyTorch
torch::Tensor mm(torch::Tensor A, torch::Tensor B) {
    // Check tensor types and contiguity
    TORCH_CHECK(A.dtype() == torch::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat, "B must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    // Dimensions
    auto M = A.size(0);
    auto K_ = A.size(1);
    auto K2 = B.size(0);
    auto N = B.size(1);

    TORCH_CHECK(K_ == K2, "Dimension mismatch: A's K != B's K.");

    // Allocate output tensor
    auto C_out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat));

    // Get pointers to data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C_out.data_ptr<float>();

    // Call the Zig function
    zig_mm(A_ptr, B_ptr, C_ptr, M, N, K_);

    return C_out;
}

// Register the module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &mm, "Matrix multiplication via Zig");
}
