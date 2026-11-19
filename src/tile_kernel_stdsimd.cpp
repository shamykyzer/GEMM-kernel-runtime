// std::experimental::simd GEMM tile kernel.
// Called once per output tile by run_tiles_serial and run_tiles_parallel in tile_driver.cpp.
#include "tile_kernels.h"

#include <cstddef>
#include <experimental/simd>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "tile_kernel_stdsimd.cpp must be compiled with AVX2 and FMA enabled (see CMakeLists.txt)"
#endif

namespace stdx = std::experimental;

namespace tile_runtime {
namespace detail {

using std::size_t;

void tile_kernel_stdsimd(const float* a, const float* b, float* c,
                         std::size_t K, std::size_t N,
                         std::size_t ii, std::size_t i_end,
                         std::size_t jj, std::size_t j_end,
                         std::size_t block_size) {
    using simd_f = stdx::native_simd<float>;
    constexpr size_t W = simd_f::size();

    for (size_t kk = 0; kk < K; kk += block_size) {
        const size_t k_end = min_size(kk + block_size, K);

        for (size_t i = ii; i < i_end; ++i) {
            size_t j = jj;
            // Vectorized loop
            for (; j + W <= j_end; j += W) {
                simd_f c_vec;
                c_vec.copy_from(&c[i * N + j], stdx::element_aligned);
                for (size_t k = kk; k < k_end; ++k) {
                    simd_f a_val(a[i * K + k]);
                    simd_f b_vec;
                    b_vec.copy_from(&b[k * N + j], stdx::element_aligned);
                    c_vec += a_val * b_vec;
                }
                c_vec.copy_to(&c[i * N + j], stdx::element_aligned);
            }
            // Scalar remainder
            for (; j < j_end; ++j) {
                float sum = 0.0f;
                for (size_t k = kk; k < k_end; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] += sum;
            }
        }
    }
}

}  // namespace detail
}  // namespace tile_runtime
