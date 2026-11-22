#pragma once

#include "tensor.h"
#include <cstddef>

namespace tile_runtime {

// Single-precision matrix multiply (GEMM) kernels. Matrices are row-major:
// A is M x K, B is K x N and C is M x N.
//
// Every function below computes C = A * B and overwrites C: whatever C held
// before the call is discarded, never added to. C must already have shape
// A.rows() x B.cols(); no function resizes it, and a shape mismatch throws
// std::invalid_argument.

// Naive triple-loop GEMM.
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Cache-friendly tiled GEMM.
// block_size controls the tile dimension for ii/jj/kk loops.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized tiled GEMM.
// Parallelizes over independent output tiles.
void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX2+FMA vectorized tiled GEMM (8-wide float, requires AVX2+FMA CPU).
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX-512 vectorized tiled GEMM (16-wide float, requires AVX-512F CPU).
void gemm_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

#ifdef TILE_HAS_STD_SIMD
// Portable SIMD GEMM using std::experimental::simd.
void gemm_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);
#endif

// OpenMP-parallelized AVX2 tiled GEMM (parallel tiles + vectorized inner loop).
void gemm_parallel_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized AVX-512 tiled GEMM (parallel tiles + 16-wide vectorized inner loop).
void gemm_parallel_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

}  // namespace tile_runtime
