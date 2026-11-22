// AVX-512 GEMM tile kernel.
// Called once per output tile by run_tiles_serial and run_tiles_parallel in tile_driver.cpp.
#include "tile_kernels.h"

#include <cstddef>
#include <immintrin.h>

#if !defined(__AVX512F__)
#error "tile_kernel_avx512.cpp must be compiled with AVX-512F enabled (see CMakeLists.txt)"
#endif

namespace tile_runtime {
namespace detail {

using std::size_t;

void tile_kernel_avx512(const float* a, const float* b, float* c,
                        std::size_t K, std::size_t N,
                        std::size_t ii, std::size_t i_end,
                        std::size_t jj, std::size_t j_end,
                        std::size_t block_size) {
    for (size_t kk = 0; kk < K; kk += block_size) {
        const size_t k_end = min_size(kk + block_size, K);

        // 4x16 micro-kernel: process 4 rows at a time
        size_t i = ii;
        for (; i + 4 <= i_end; i += 4) {
            size_t j = jj;
            // Vectorized: 16-wide AVX-512
            for (; j + 16 <= j_end; j += 16) {
                __m512 c0 = _mm512_loadu_ps(&c[(i + 0) * N + j]);
                __m512 c1 = _mm512_loadu_ps(&c[(i + 1) * N + j]);
                __m512 c2 = _mm512_loadu_ps(&c[(i + 2) * N + j]);
                __m512 c3 = _mm512_loadu_ps(&c[(i + 3) * N + j]);

                for (size_t k = kk; k < k_end; ++k) {
                    __m512 b_vec = _mm512_loadu_ps(&b[k * N + j]);
                    _mm_prefetch(reinterpret_cast<const char*>(&b[(k + 2) * N + j]), _MM_HINT_T0);
                    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 0) * K + k]), b_vec, c0);
                    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 1) * K + k]), b_vec, c1);
                    c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 2) * K + k]), b_vec, c2);
                    c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 3) * K + k]), b_vec, c3);
                }

                _mm512_storeu_ps(&c[(i + 0) * N + j], c0);
                _mm512_storeu_ps(&c[(i + 1) * N + j], c1);
                _mm512_storeu_ps(&c[(i + 2) * N + j], c2);
                _mm512_storeu_ps(&c[(i + 3) * N + j], c3);
            }
            // Masked cleanup for remaining columns (1-15)
            if (j < j_end) {
                const __mmask16 mask = static_cast<__mmask16>((1u << (j_end - j)) - 1);
                __m512 c0 = _mm512_maskz_loadu_ps(mask, &c[(i + 0) * N + j]);
                __m512 c1 = _mm512_maskz_loadu_ps(mask, &c[(i + 1) * N + j]);
                __m512 c2 = _mm512_maskz_loadu_ps(mask, &c[(i + 2) * N + j]);
                __m512 c3 = _mm512_maskz_loadu_ps(mask, &c[(i + 3) * N + j]);

                for (size_t k = kk; k < k_end; ++k) {
                    __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[k * N + j]);
                    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 0) * K + k]), b_vec, c0);
                    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 1) * K + k]), b_vec, c1);
                    c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 2) * K + k]), b_vec, c2);
                    c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + 3) * K + k]), b_vec, c3);
                }

                _mm512_mask_storeu_ps(&c[(i + 0) * N + j], mask, c0);
                _mm512_mask_storeu_ps(&c[(i + 1) * N + j], mask, c1);
                _mm512_mask_storeu_ps(&c[(i + 2) * N + j], mask, c2);
                _mm512_mask_storeu_ps(&c[(i + 3) * N + j], mask, c3);
            }
        }
        // Remaining rows (1-3) with masked column handling
        for (; i < i_end; ++i) {
            size_t j = jj;
            for (; j + 16 <= j_end; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(&c[i * N + j]);
                for (size_t k = kk; k < k_end; ++k) {
                    __m512 b_vec = _mm512_loadu_ps(&b[k * N + j]);
                    c_vec = _mm512_fmadd_ps(_mm512_set1_ps(a[i * K + k]), b_vec, c_vec);
                }
                _mm512_storeu_ps(&c[i * N + j], c_vec);
            }
            if (j < j_end) {
                const __mmask16 mask = static_cast<__mmask16>((1u << (j_end - j)) - 1);
                __m512 c_vec = _mm512_maskz_loadu_ps(mask, &c[i * N + j]);
                for (size_t k = kk; k < k_end; ++k) {
                    __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[k * N + j]);
                    c_vec = _mm512_fmadd_ps(_mm512_set1_ps(a[i * K + k]), b_vec, c_vec);
                }
                _mm512_mask_storeu_ps(&c[i * N + j], mask, c_vec);
            }
        }
    }
}

}  // namespace detail
}  // namespace tile_runtime
