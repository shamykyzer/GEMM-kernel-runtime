#include "gemm.h"

#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tile_runtime {

void gemm_parallel_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "gemm_parallel_simd: A.cols() != B.rows() — dimension mismatch");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            "gemm_parallel_simd: C dimensions do not match (A.rows() x B.cols())");
    }

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

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

            // 4x8 micro-kernel
            size_t i = ii;
            for (; i + 4 <= i_end; i += 4) {
                size_t j = jj;
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
                // Scalar cleanup
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
            // Remaining rows
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
}

}  // namespace tile_runtime
