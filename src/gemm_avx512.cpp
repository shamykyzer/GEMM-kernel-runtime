#include "gemm.h"

#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace tile_runtime {

void gemm_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "gemm_avx512: A.cols() != B.rows() — dimension mismatch");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            "gemm_avx512: C dimensions do not match (A.rows() x B.cols())");
    }

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (size_t ii = 0; ii < M; ii += block_size) {
        const size_t i_end = std::min(ii + block_size, M);
        for (size_t jj = 0; jj < N; jj += block_size) {
            const size_t j_end = std::min(jj + block_size, N);
            for (size_t kk = 0; kk < K; kk += block_size) {
                const size_t k_end = std::min(kk + block_size, K);

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
    }
}

}  // namespace tile_runtime
