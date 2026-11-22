// Public tiled GEMM entry points called by users of gemm.h.
// Each entry point selects the tile kernel and driver listed below.
//
// Entry point            Tile kernel  Driver
// gemm_tiled             scalar       serial
// gemm_parallel          scalar       parallel
// gemm_avx               avx2         serial
// gemm_parallel_simd     avx2         parallel
// gemm_avx512            avx512       serial
// gemm_parallel_avx512   avx512       parallel
// gemm_simd              stdsimd      serial (throws when stdsimd is not built)

#include "gemm.h"
#include "tile_driver.h"

namespace tile_runtime {

void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_tiled", A, B, C, block_size);
    detail::run_tiles_serial(detail::tile_kernel_scalar, A, B, C, block_size);
}

void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_parallel", A, B, C, block_size);
    detail::run_tiles_parallel(detail::tile_kernel_scalar, A, B, C, block_size);
}

void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_avx", A, B, C, block_size);
    const detail::TileKernel kernel = detail::require_avx2_kernel("gemm_avx");
    detail::run_tiles_serial(kernel, A, B, C, block_size);
}

void gemm_parallel_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_parallel_simd", A, B, C, block_size);
    const detail::TileKernel kernel = detail::require_avx2_kernel("gemm_parallel_simd");
    detail::run_tiles_parallel(kernel, A, B, C, block_size);
}

void gemm_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_avx512", A, B, C, block_size);
    const detail::TileKernel kernel = detail::require_avx512_kernel("gemm_avx512");
    detail::run_tiles_serial(kernel, A, B, C, block_size);
}

void gemm_parallel_avx512(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_parallel_avx512", A, B, C, block_size);
    const detail::TileKernel kernel = detail::require_avx512_kernel("gemm_parallel_avx512");
    detail::run_tiles_parallel(kernel, A, B, C, block_size);
}

void gemm_simd(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    detail::check_gemm_args("gemm_simd", A, B, C, block_size);
    const detail::TileKernel kernel = detail::require_stdsimd_kernel("gemm_simd");
    detail::run_tiles_serial(kernel, A, B, C, block_size);
}

bool gemm_avx2_supported() { return detail::avx2_kernel_supported(); }
bool gemm_avx512_supported() { return detail::avx512_kernel_supported(); }
bool gemm_simd_supported() { return detail::stdsimd_kernel_supported(); }

}  // namespace tile_runtime
