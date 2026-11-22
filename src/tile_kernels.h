// Internal tile kernel interface used by the GEMM entry points and tile drivers.
// Shared by the scalar and SIMD tile implementations.
#pragma once

#include <cstddef>

namespace tile_runtime {
namespace detail {

// Computes one output tile: for i in [ii, i_end) and j in [jj, j_end),
//     c[i * N + j] += sum over k of a[i * K + k] * b[k * N + j]
// walking k in chunks of block_size. A is M x K, B is K x N and C is M x N,
// all row-major. A tile kernel must not write outside its tile and must not
// throw.
using TileKernel = void (*)(const float* a, const float* b, float* c,
                            std::size_t K, std::size_t N,
                            std::size_t ii, std::size_t i_end,
                            std::size_t jj, std::size_t j_end,
                            std::size_t block_size);

void tile_kernel_scalar(const float* a, const float* b, float* c,
                        std::size_t K, std::size_t N,
                        std::size_t ii, std::size_t i_end,
                        std::size_t jj, std::size_t j_end,
                        std::size_t block_size);

void tile_kernel_avx2(const float* a, const float* b, float* c,
                      std::size_t K, std::size_t N,
                      std::size_t ii, std::size_t i_end,
                      std::size_t jj, std::size_t j_end,
                      std::size_t block_size);

void tile_kernel_avx512(const float* a, const float* b, float* c,
                        std::size_t K, std::size_t N,
                        std::size_t ii, std::size_t i_end,
                        std::size_t jj, std::size_t j_end,
                        std::size_t block_size);

// Defined only in builds with TILE_HAS_STD_SIMD; reached through require_stdsimd_kernel.
void tile_kernel_stdsimd(const float* a, const float* b, float* c,
                         std::size_t K, std::size_t N,
                         std::size_t ii, std::size_t i_end,
                         std::size_t jj, std::size_t j_end,
                         std::size_t block_size);

// std::min without <algorithm>. The tile kernel files include no standard
// library headers beyond <cstddef> (plus <experimental/simd> in the std::simd
// one), so no general library code is compiled with their instruction set
// where the linker could then pick it for the whole program.
static inline std::size_t min_size(std::size_t x, std::size_t y) { return x < y ? x : y; }

}  // namespace detail
}  // namespace tile_runtime
