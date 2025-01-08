#include <torch/extension.h>
#include <vector>
#include <cstddef>

extern "C" {
    void zig_mm(const float* A, const float* B, float* C, 
               size_t M, size_t N, size_t K);
}

torch::Tensor mm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2, "Matrix A must be 2-dimensional");
    TORCH_CHECK(b.dim() == 2, "Matrix B must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Incompatible matrix dimensions");
    
    const auto M = a.size(0);
    const auto K = a.size(1);
    const auto N = b.size(1);
    
    auto a_cont = a.contiguous().to(torch::kFloat32);
    auto b_cont = b.contiguous().to(torch::kFloat32);
    auto c = torch::zeros({M, N}, torch::kFloat32);
    
    const float* ptra = a_cont.data_ptr<float>();
    const float* ptrb = b_cont.data_ptr<float>();
    float* ptrc       = c.data_ptr<float>();
    
    zig_mm(ptra, ptrb, ptrc, M, N, K);
    
    return c;
}

PYBIND11_MODULE(zigtorch, m) {
    m.def("mm", &mm, "Optimized matrix multiplication");
}
