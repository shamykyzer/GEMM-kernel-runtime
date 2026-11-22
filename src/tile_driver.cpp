// Shared GEMM shape checks, CPU support guards, and serial and OpenMP tile traversal.
// The GEMM entry points select the tile kernel and traversal mode.
#include "tile_driver.h"
#include "cpu_features.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tile_runtime {
namespace detail {

// These sets cover what MSVC may emit under /arch:AVX2 and /arch:AVX512,
// even when the intrinsics in a tile kernel use fewer instruction sets.
bool avx2_kernel_supported() {
    const CpuFeatures& f = CpuFeatures::detect();
    return f.avx && f.avx2 && f.fma && f.bmi1 && f.bmi2;
}

bool avx512_kernel_supported() {
    const CpuFeatures& f = CpuFeatures::detect();
    return avx2_kernel_supported() && f.avx512f && f.avx512vl &&
           f.avx512bw && f.avx512dq && f.avx512cd;
}

bool stdsimd_kernel_supported() {
#ifdef TILE_HAS_STD_SIMD
    return avx2_kernel_supported();
#else
    return false;
#endif
}

TileKernel require_avx2_kernel(const char* kernel) {
    if (avx2_kernel_supported()) {
        return tile_kernel_avx2;
    }
    throw std::runtime_error(std::string(kernel) +
        ": this CPU or operating system does not support AVX2, FMA, BMI1 and BMI2");
}

TileKernel require_avx512_kernel(const char* kernel) {
    if (avx512_kernel_supported()) {
        return tile_kernel_avx512;
    }
    throw std::runtime_error(std::string(kernel) +
        ": this CPU or operating system does not support AVX-512 (F, VL, BW, DQ, CD) with AVX2, FMA, BMI1 and BMI2");
}

TileKernel require_stdsimd_kernel(const char* kernel) {
#ifdef TILE_HAS_STD_SIMD
    if (stdsimd_kernel_supported()) {
        return tile_kernel_stdsimd;
    }
    throw std::runtime_error(std::string(kernel) +
        ": this CPU or operating system does not support AVX2, FMA, BMI1 and BMI2");
#else
    throw std::runtime_error(std::string(kernel) +
        ": the std::simd kernel is not part of this build (it is only built with GCC)");
#endif
}

static std::string shape(const Tensor& tensor) {
    return std::to_string(tensor.rows()) + "x" + std::to_string(tensor.cols());
}

void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C) {
    // Compare objects: distinct empty tensors may have null data pointers, and
    // distinct tensors own separate storage. A and B may be the same object.
    if (&C == &A) {
        throw std::invalid_argument(std::string(kernel) +
            ": C is the same tensor as A (the output must not alias an input)");
    }
    if (&C == &B) {
        throw std::invalid_argument(std::string(kernel) +
            ": C is the same tensor as B (the output must not alias an input)");
    }
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(std::string(kernel) + ": A is " + shape(A) +
            " but B is " + shape(B) + " (A.cols() must equal B.rows())");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(std::string(kernel) + ": C is " + shape(C) +
            " but A * B is " + std::to_string(A.rows()) + "x" + std::to_string(B.cols()) +
            " (C must be A.rows() x B.cols())");
    }
}

void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C,
                     std::size_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument(std::string(kernel) + ": block_size must be at least 1");
    }
    check_gemm_args(kernel, A, B, C);
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

    // Return before clamping: with every dimension 0 the clamp would give a
    // block of 0. A block at or above max(M, K, N) already meant one tile and
    // one k chunk, so clamping changes no result, and it keeps M + block - 1
    // from wrapping for huge block sizes.
    if (M == 0 || N == 0) return;
    const std::size_t block = std::min(block_size, std::max({M, K, N}));

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (std::size_t ii = 0; ii < M; ii += block) {
        const std::size_t i_end = std::min(ii + block, M);
        for (std::size_t jj = 0; jj < N; jj += block) {
            const std::size_t j_end = std::min(jj + block, N);
            // The tile kernels add into C, so zeroing makes the entry points overwrite it.
            zero_tile(c, N, ii, i_end, jj, j_end);
            kernel(a, b, c, K, N, ii, i_end, jj, j_end, block);
        }
    }
}

void run_tiles_parallel(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    // Early return and clamp: see run_tiles_serial.
    if (M == 0 || N == 0) return;
    const std::size_t block = std::min(block_size, std::max({M, K, N}));

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    // Linearize the 2D tile grid into a single loop for MSVC compatibility.
    // MSVC's OpenMP doesn't support collapse() and requires signed loop vars.
    // Use 64 bits because int overflows past 2^31 tiles. The product cannot
    // overflow: there are no more tiles than elements of C, which is in memory.
    const std::int64_t tile_rows = static_cast<std::int64_t>((M + block - 1) / block);
    const std::int64_t tile_cols = static_cast<std::int64_t>((N + block - 1) / block);
    const std::int64_t num_tiles = tile_rows * tile_cols;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const std::size_t ii = static_cast<std::size_t>(tile_idx / tile_cols) * block;
        const std::size_t jj = static_cast<std::size_t>(tile_idx % tile_cols) * block;
        const std::size_t i_end = std::min(ii + block, M);
        const std::size_t j_end = std::min(jj + block, N);

        zero_tile(c, N, ii, i_end, jj, j_end);
        kernel(a, b, c, K, N, ii, i_end, jj, j_end, block);
    }
}

}  // namespace detail
}  // namespace tile_runtime
