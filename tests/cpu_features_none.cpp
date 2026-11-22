// Replaces the real detector in test_gemm_no_simd so every SIMD kernel
// must throw std::runtime_error.
#include "cpu_features.h"

namespace tile_runtime {

const CpuFeatures& CpuFeatures::detect() {
    static const CpuFeatures features{};
    return features;
}

}  // namespace tile_runtime
