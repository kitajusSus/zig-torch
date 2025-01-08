#include <torch/extension.h>
#include <vector>

// Deklaracja funkcji Zig
extern "C" void zig_mm(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);

// Wrapper dla PyTorch
torch::Tensor mm(torch::Tensor A, torch::Tensor B) {
    // Sprawdzenie typów i ciągłości danych
    TORCH_CHECK(A.dtype() == torch::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat, "B must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    auto M = A.size(0);
    auto K_ = A.size(1);
    auto K2 = B.size(0);
    auto N = B.size(1);

    TORCH_CHECK(K_ == K2, "Dimension mismatch: A's K != B's K.");

    // Alokowanie wyjścia
    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat));

    // Pobieranie wskaźników
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Wywołanie funkcji Zig
    zig_mm(A_ptr, B_ptr, C_ptr, M, N, K_);

    return C;
}

// Rejestracja modułu
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &mm, "Matrix multiplication via Zig");
}
