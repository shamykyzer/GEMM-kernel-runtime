// Internal argument checks, CPU support guards, and tile traversal for GEMM.
// The drivers take a tile kernel so both traversal modes share the same arithmetic.
#pragma once

#include "tensor.h"
#include "tile_kernels.h"

#include <cstddef>

namespace tile_runtime {
namespace detail {

// True when this CPU, its operating system and this build can run the tile kernel.
bool avx2_kernel_supported();
bool avx512_kernel_supported();
bool stdsimd_kernel_supported();

// Returns the tile kernel, or throws std::runtime_error naming `kernel`
// (the public entry point) and the instruction sets it needs.
TileKernel require_avx2_kernel(const char* kernel);
TileKernel require_avx512_kernel(const char* kernel);
TileKernel require_stdsimd_kernel(const char* kernel);

// Throws std::invalid_argument if C is the same Tensor object as A or B, or
// unless A is M x K, B is K x N and C is M x N.
void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C);

// The same checks, and block_size must be at least 1.
void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C,
                     std::size_t block_size);

// Both drivers require check_gemm_args with block_size to have succeeded.

// Zeroes each block_size x block_size output tile and then calls `kernel` on
// it, row by row. The tile kernels add into C, so zeroing the tile first is
// what makes every entry point overwrite C.
void run_tiles_serial(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size);

// The same tiles shared out across OpenMP threads. Each tile is zeroed and
// written by exactly one thread.
void run_tiles_parallel(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size);

}  // namespace detail
}  // namespace tile_runtime
