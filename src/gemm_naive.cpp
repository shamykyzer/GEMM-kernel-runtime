#include "gemm.h"
#include "tile_driver.h"

namespace tile_runtime {

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    detail::check_gemm_args("gemm_naive", A, B, C);

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

}  // namespace tile_runtime
