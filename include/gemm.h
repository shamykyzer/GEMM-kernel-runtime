#pragma once

#include "tensor.h"
#include <cstddef>

namespace tile_runtime {

// Single-precision matrix multiply (GEMM) kernels. Matrices are row-major:
// A is M x K, B is K x N and C is M x N.
//
// Every function below computes C = A * B and overwrites C: whatever C held
// before the call is discarded, never added to. C must already have shape
// A.rows() x B.cols(); no function resizes it.
//
// Every function throws std::invalid_argument, before it reads or writes any
// element, when A.cols() != B.rows(), when C is not A.rows() x B.cols(), when
// block_size is 0 (the tiled functions), or when C is the same Tensor object
// as A or B. A and B may be the same object. A block_size larger than the
// matrices is allowed and means a single tile.
//
// The SIMD kernels check the CPU themselves. Called on a machine or in a
// build that cannot run them, they throw std::runtime_error instead of
// crashing. Arguments are checked first, so a bad argument is
// std::invalid_argument on every machine. The gemm_*_supported() functions
// at the end of this header report what will run.

// Naive triple-loop GEMM.
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Cache-friendly tiled GEMM.
// block_size controls the tile dimension for ii/jj/kk loops.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized tiled GEMM.
// Parallelizes over independent output tiles.
void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX2+FMA vectorized tiled GEMM (8-wide float).
// Throws std::runtime_error unless gemm_avx2_supported().
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// AVX-512 vectorized tiled GEMM (16-wide float).
// Throws std::runtime_error unless gemm_avx512_supported().
void gemm_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// Portable SIMD GEMM using std::experimental::simd.
// Throws std::runtime_error unless gemm_simd_supported().
void gemm_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized AVX2 tiled GEMM (parallel tiles + vectorized inner loop).
// Throws std::runtime_error unless gemm_avx2_supported().
void gemm_parallel_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized AVX-512 tiled GEMM (parallel tiles + 16-wide vectorized inner loop).
// Throws std::runtime_error unless gemm_avx512_supported().
void gemm_parallel_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// True when this CPU, its operating system and this build can run the
// kernels named. Cheap to call: the CPU is examined once and the answer is
// cached.
bool gemm_avx2_supported();    // gemm_avx, gemm_parallel_simd
bool gemm_avx512_supported();  // gemm_avx512, gemm_parallel_avx512
bool gemm_simd_supported();    // gemm_simd; false when the std::simd kernel is not built (compilers other than GCC)

}  // namespace tile_runtime
