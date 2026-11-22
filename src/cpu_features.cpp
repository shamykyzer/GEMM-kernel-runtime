// CPU and operating system feature detection for the library.
// Kept separate and built without SIMD flags so checking support cannot
// itself execute instructions that the process cannot use.
#include "cpu_features.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#include <immintrin.h>
#endif

namespace tile_runtime {
namespace {

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
bool bit(unsigned value, unsigned index) {
    return ((value >> index) & 1u) != 0;
}
#endif

CpuFeatures query() {
    CpuFeatures f{};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const unsigned max_leaf = static_cast<unsigned>(regs[0]);
    if (max_leaf < 1) {
        return f;
    }

    __cpuid(regs, 1);
    const unsigned leaf1_ecx = static_cast<unsigned>(regs[2]);
    const bool osxsave = bit(leaf1_ecx, 27);

    // XCR0 says which register state the operating system saves and
    // restores. Reading it is only legal when OSXSAVE is set.
    bool ymm_enabled = false;
    bool zmm_enabled = false;
    if (osxsave) {
        const unsigned long long xcr0 = _xgetbv(0);
        ymm_enabled = (xcr0 & 0x06u) == 0x06u;                 // XMM and YMM
        zmm_enabled = ymm_enabled && (xcr0 & 0xE0u) == 0xE0u;  // opmask, ZMM0-15 upper halves, ZMM16-31
    }

    unsigned leaf7_ebx = 0;
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        leaf7_ebx = static_cast<unsigned>(regs[1]);
    }

    f.avx      = ymm_enabled && bit(leaf1_ecx, 28);
    f.fma      = ymm_enabled && bit(leaf1_ecx, 12);
    f.avx2     = ymm_enabled && bit(leaf7_ebx, 5);
    f.bmi1     = bit(leaf7_ebx, 3);
    f.bmi2     = bit(leaf7_ebx, 8);
    f.avx512f  = zmm_enabled && bit(leaf7_ebx, 16);
    f.avx512dq = zmm_enabled && bit(leaf7_ebx, 17);
    f.avx512cd = zmm_enabled && bit(leaf7_ebx, 28);
    f.avx512bw = zmm_enabled && bit(leaf7_ebx, 30);
    f.avx512vl = zmm_enabled && bit(leaf7_ebx, 31);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    f.avx      = __builtin_cpu_supports("avx");
    f.avx2     = __builtin_cpu_supports("avx2");
    f.fma      = __builtin_cpu_supports("fma");
    f.bmi1     = __builtin_cpu_supports("bmi");
    f.bmi2     = __builtin_cpu_supports("bmi2");
    f.avx512f  = __builtin_cpu_supports("avx512f");
    f.avx512vl = __builtin_cpu_supports("avx512vl");
    f.avx512bw = __builtin_cpu_supports("avx512bw");
    f.avx512dq = __builtin_cpu_supports("avx512dq");
    f.avx512cd = __builtin_cpu_supports("avx512cd");
#endif
    return f;
}

}  // namespace

const CpuFeatures& CpuFeatures::detect() {
    static const CpuFeatures features = query();
    return features;
}

}  // namespace tile_runtime
