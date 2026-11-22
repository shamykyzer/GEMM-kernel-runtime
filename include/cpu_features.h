#pragma once

namespace tile_runtime {

// What this process may use: a flag is true only when the CPU has the
// instruction set and the operating system saves the registers it needs.
struct CpuFeatures {
    bool avx;
    bool avx2;
    bool fma;
    bool bmi1;
    bool bmi2;
    bool avx512f;
    bool avx512vl;
    bool avx512bw;
    bool avx512dq;
    bool avx512cd;

    // Examines the CPU on the first call and returns the cached result
    // after that. Safe to call from several threads.
    static const CpuFeatures& detect();
};

}  // namespace tile_runtime
