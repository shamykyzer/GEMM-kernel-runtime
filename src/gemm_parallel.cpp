#include "gemm.h"

#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tile_runtime {

void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "gemm_parallel: A.cols() != B.rows() — dimension mismatch");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            "gemm_parallel: C dimensions do not match (A.rows() x B.cols())");
    }

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    // Linearize the 2D tile grid into a single loop for MSVC compatibility.
    // MSVC's OpenMP doesn't support collapse() and requires signed loop vars.
    const int tile_rows = static_cast<int>((M + block_size - 1) / block_size);
    const int tile_cols = static_cast<int>((N + block_size - 1) / block_size);
    const int num_tiles = tile_rows * tile_cols;

    #pragma omp parallel for schedule(static)
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const size_t ii = static_cast<size_t>(tile_idx / tile_cols) * block_size;
        const size_t jj = static_cast<size_t>(tile_idx % tile_cols) * block_size;
        const size_t i_end = std::min(ii + block_size, M);
        const size_t j_end = std::min(jj + block_size, N);

        for (size_t kk = 0; kk < K; kk += block_size) {
            const size_t k_end = std::min(kk + block_size, K);

            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
                    float sum = 0.0f;
                    for (size_t k = kk; k < k_end; ++k) {
                        sum += a[i * K + k] * b[k * N + j];
                    }
                    c[i * N + j] += sum;
                }
            }
        }
    }
}

}  // namespace tile_runtime
