#pragma once

#include "tensor.h"
#include <cstddef>

namespace tile_runtime {

// Naive triple-loop GEMM: C = A * B
// Requires: A.cols() == B.rows(), C sized to (A.rows() x B.cols()).
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Cache-friendly tiled GEMM: C = A * B
// block_size controls the tile dimension for ii/jj/kk loops.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized tiled GEMM: C = A * B
// Parallelizes over independent output tiles.
void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX2+FMA vectorized tiled GEMM (8-wide float, requires AVX2+FMA CPU).
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX-512 vectorized tiled GEMM (16-wide float, requires AVX-512F CPU).
void gemm_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// Portable SIMD GEMM using std::experimental::simd.
void gemm_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized AVX2 tiled GEMM (parallel tiles + vectorized inner loop).
void gemm_parallel_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

}  // namespace tile_runtime
