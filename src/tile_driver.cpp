// Shared GEMM shape checks and serial and OpenMP tile traversal.
// The GEMM entry points select the tile kernel and traversal mode.
#include "tile_driver.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tile_runtime {
namespace detail {

void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            std::string(kernel) + ": A.cols() != B.rows() (dimension mismatch)");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            std::string(kernel) + ": C dimensions do not match (A.rows() x B.cols())");
    }
}

// Zeroes one output tile: columns [jj, j_end) of rows [ii, i_end).
static void zero_tile(float* c, std::size_t N,
                      std::size_t ii, std::size_t i_end,
                      std::size_t jj, std::size_t j_end) {
    for (std::size_t i = ii; i < i_end; ++i) {
        std::fill(c + i * N + jj, c + i * N + j_end, 0.0f);
    }
}

void run_tiles_serial(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (std::size_t ii = 0; ii < M; ii += block_size) {
        const std::size_t i_end = std::min(ii + block_size, M);
        for (std::size_t jj = 0; jj < N; jj += block_size) {
            const std::size_t j_end = std::min(jj + block_size, N);
            // The tile kernels add into C, so zeroing makes the entry points overwrite it.
            zero_tile(c, N, ii, i_end, jj, j_end);
            kernel(a, b, c, K, N, ii, i_end, jj, j_end, block_size);
        }
    }
}

void run_tiles_parallel(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    // Linearize the 2D tile grid into a single loop for MSVC compatibility.
    // MSVC's OpenMP doesn't support collapse() and requires signed loop vars.
    const int tile_rows = static_cast<int>((M + block_size - 1) / block_size);
    const int tile_cols = static_cast<int>((N + block_size - 1) / block_size);
    const int num_tiles = tile_rows * tile_cols;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const std::size_t ii = static_cast<std::size_t>(tile_idx / tile_cols) * block_size;
        const std::size_t jj = static_cast<std::size_t>(tile_idx % tile_cols) * block_size;
        const std::size_t i_end = std::min(ii + block_size, M);
        const std::size_t j_end = std::min(jj + block_size, N);

        zero_tile(c, N, ii, i_end, jj, j_end);
        kernel(a, b, c, K, N, ii, i_end, jj, j_end, block_size);
    }
}

}  // namespace detail
}  // namespace tile_runtime
