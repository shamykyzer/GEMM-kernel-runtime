// Internal argument checks and tile traversal used by the GEMM entry points.
// The drivers take a tile kernel so both traversal modes share the same arithmetic.
#pragma once

#include "tensor.h"
#include "tile_kernels.h"

#include <cstddef>

namespace tile_runtime {
namespace detail {

// Throws std::invalid_argument unless A is M x K, B is K x N and C is M x N.
void check_gemm_args(const char* kernel, const Tensor& A, const Tensor& B, const Tensor& C);

// Calls `kernel` once per block_size x block_size output tile, row by row.
void run_tiles_serial(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size);

// The same tiles shared out across OpenMP threads. Each tile is written by exactly one thread.
void run_tiles_parallel(TileKernel kernel, const Tensor& A, const Tensor& B, Tensor& C, std::size_t block_size);

}  // namespace detail
}  // namespace tile_runtime
