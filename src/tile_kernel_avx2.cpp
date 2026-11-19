// AVX2 GEMM tile kernel.
// Called once per output tile by run_tiles_serial and run_tiles_parallel in tile_driver.cpp.
#include "tile_kernels.h"

#include <cstddef>
#include <immintrin.h>

#if !defined(__AVX2__) || (!defined(_MSC_VER) && !defined(__FMA__))
#error "tile_kernel_avx2.cpp must be compiled with AVX2 and FMA enabled (see CMakeLists.txt)"
#endif

namespace tile_runtime {
namespace detail {

using std::size_t;

void tile_kernel_avx2(const float* a, const float* b, float* c,
                      std::size_t K, std::size_t N,
                      std::size_t ii, std::size_t i_end,
                      std::size_t jj, std::size_t j_end,
                      std::size_t block_size) {
    for (size_t kk = 0; kk < K; kk += block_size) {
        const size_t k_end = min_size(kk + block_size, K);

        // 4x8 micro-kernel: process 4 rows at a time
        size_t i = ii;
        for (; i + 4 <= i_end; i += 4) {
            size_t j = jj;
            // Vectorized: 8-wide AVX2
            for (; j + 8 <= j_end; j += 8) {
                __m256 c0 = _mm256_loadu_ps(&c[(i + 0) * N + j]);
                __m256 c1 = _mm256_loadu_ps(&c[(i + 1) * N + j]);
                __m256 c2 = _mm256_loadu_ps(&c[(i + 2) * N + j]);
                __m256 c3 = _mm256_loadu_ps(&c[(i + 3) * N + j]);

                for (size_t k = kk; k < k_end; ++k) {
                    __m256 b_vec = _mm256_loadu_ps(&b[k * N + j]);
                    _mm_prefetch(reinterpret_cast<const char*>(&b[(k + 2) * N + j]), _MM_HINT_T0);
                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 0) * K + k]), b_vec, c0);
                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 1) * K + k]), b_vec, c1);
                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 2) * K + k]), b_vec, c2);
                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 3) * K + k]), b_vec, c3);
                }

                _mm256_storeu_ps(&c[(i + 0) * N + j], c0);
                _mm256_storeu_ps(&c[(i + 1) * N + j], c1);
                _mm256_storeu_ps(&c[(i + 2) * N + j], c2);
                _mm256_storeu_ps(&c[(i + 3) * N + j], c3);
            }
            // Scalar cleanup for remaining columns
            for (; j < j_end; ++j) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (size_t k = kk; k < k_end; ++k) {
                    float bv = b[k * N + j];
                    s0 += a[(i + 0) * K + k] * bv;
                    s1 += a[(i + 1) * K + k] * bv;
                    s2 += a[(i + 2) * K + k] * bv;
                    s3 += a[(i + 3) * K + k] * bv;
                }
                c[(i + 0) * N + j] += s0;
                c[(i + 1) * N + j] += s1;
                c[(i + 2) * N + j] += s2;
                c[(i + 3) * N + j] += s3;
            }
        }
        // Remaining rows (1-3)
        for (; i < i_end; ++i) {
            size_t j = jj;
            for (; j + 8 <= j_end; j += 8) {
                __m256 c_vec = _mm256_loadu_ps(&c[i * N + j]);
                for (size_t k = kk; k < k_end; ++k) {
                    __m256 b_vec = _mm256_loadu_ps(&b[k * N + j]);
                    c_vec = _mm256_fmadd_ps(_mm256_set1_ps(a[i * K + k]), b_vec, c_vec);
                }
                _mm256_storeu_ps(&c[i * N + j], c_vec);
            }
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
