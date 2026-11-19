// Scalar GEMM tile kernel.
// Called once per output tile by run_tiles_serial and run_tiles_parallel in tile_driver.cpp.
#include "tile_kernels.h"

#include <cstddef>

namespace tile_runtime {
namespace detail {

using std::size_t;

void tile_kernel_scalar(const float* a, const float* b, float* c,
                        std::size_t K, std::size_t N,
                        std::size_t ii, std::size_t i_end,
                        std::size_t jj, std::size_t j_end,
                        std::size_t block_size) {
    for (size_t kk = 0; kk < K; kk += block_size) {
        const size_t k_end = min_size(kk + block_size, K);

        for (size_t i = ii; i < i_end; ++i) {
            for (size_t j = jj; j < j_end; ++j) {
                float sum = 0.0f;
                for (size_t k = kk; k < k_end; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] += sum;  // accumulate across kk tiles
            }
        }
    }
}

}  // namespace detail
}  // namespace tile_runtime
